import os
import json
import time
import warnings
from typing import Dict, Any, List

import numpy as np
import torch
from torch import nn
import foolbox as fb
from foolbox.attacks import LinfProjectedGradientDescentAttack
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights

# 屏蔽多余警告
warnings.filterwarnings("ignore")

# ====== 配置区（只保留一份！） ======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PGD 参数：eps=8/255
EPS = 8 / 255.0
STEPS = 10
ALPHA = 2 / 255.0

BATCH_SIZE = 16
NUM_IMAGES = 64
RANDOM_START = True

TORCH_HOME = r"C:\Users\28983\.cache\torch"
OUT_JSON = "pgd_report_resnet50.json"


def make_inputs(num_images: int) -> torch.Tensor:
    """
    生成随机RGB图像张量，shape=[num_images, 3, 64, 64]，值域[0,1]。
    """
    torch.manual_seed(0)
    x = torch.rand(num_images, 3, 64, 64, dtype=torch.float32)  # [0,1]
    return x


@torch.no_grad()
def predict_pseudo_labels(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    使用模型对原图预测类别，作为伪标签 pseudo label。
    """
    logits = model(x)
    return logits.argmax(dim=1)


def main():
    os.environ["TORCH_HOME"] = TORCH_HOME

    # 1) 加载 ResNet50
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights).to(DEVICE).eval()

    # 2) Foolbox 包装
    fmodel = fb.PyTorchModel(model, bounds=(0.0, 1.0), device=DEVICE)

    # 3) 输入数据
    x_all = make_inputs(NUM_IMAGES).to(DEVICE)  # [N,3,64,64]
    assert x_all.ndim == 4 and x_all.shape[1:] == (3, 64, 64)
    assert float(x_all.min()) >= 0.0 and float(x_all.max()) <= 1.0

    # 4) 生成伪标签
    y_all = predict_pseudo_labels(model, x_all)  # [N]

    # 5) 定义 PGD攻击（使用完整类名！！！）
    attack = LinfProjectedGradientDescentAttack(
        steps=STEPS,
        random_start=RANDOM_START,
    )

    results: List[Dict[str, Any]] = []
    success = 0
    t0 = time.time()

    for start in tqdm(range(0, NUM_IMAGES, BATCH_SIZE), desc="Attacking"):
        end = min(start + BATCH_SIZE, NUM_IMAGES)
        x = x_all[start:end]
        y = y_all[start:end]

        # 只运行一次攻击！新版foolbox返回三个值
        adv, _, _ = attack(fmodel, x, y, epsilons=EPS)

        logits_adv = model(adv)
        y_adv = logits_adv.argmax(dim=1)

        # 对抗成功：预测类别发生变化
        is_success = (y_adv != y).detach().cpu().numpy().astype(bool)
        success += int(is_success.sum())

        for i in range(end - start):
            results.append({
                "idx": start + i,
                "pseudo_label": int(y[i].detach().cpu().item()),
                "adv_label": int(y_adv[i].detach().cpu().item()),
                "success": bool(is_success[i]),
            })

    elapsed = time.time() - t0
    success_rate = success / NUM_IMAGES

    report = {
        "attack": {
            "type": "LinfPGD",
            "eps": EPS,
            "steps": STEPS,
            "alpha_abs_stepsize": ALPHA,
            "random_start": RANDOM_START,
            "bounds": [0.0, 1.0],
        },
        "data": {
            "num_images": NUM_IMAGES,
            "input_shape": [NUM_IMAGES, 3, 64, 64],
            "value_range": [0.0, 1.0],
            "pseudo_label_strategy": "model_predicted_class_on_original_images",
        },
        "model": {
            "name": "torchvision.resnet50",
            "weights": "ResNet50_Weights.IMAGENET1K_V1",
            "device": DEVICE,
        },
        "metrics": {
            "success_count": success,
            "total": NUM_IMAGES,
            "success_rate": success_rate,
            "elapsed_sec": elapsed,
        },
        "per_sample": results,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("===== DONE =====")
    print(f"Success rate: {success_rate:.4f} ({success}/{NUM_IMAGES})")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Report saved to: {OUT_JSON}")


if __name__ == "__main__":
    main()