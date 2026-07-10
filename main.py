"""
项目主运行流程
串联数据、模型、对抗攻击、结果统计，供团队联调使用
"""
import torch
import json
from torchvision.models import resnet50, ResNet50_Weights
# 导入你写好的攻击生成工具（核心，调用你的任务代码）
from attack_generator import AttackGenerator


def create_dummy_data(num_samples=64, batch_size=16):
    """模拟数据集，后续替换真实图片"""
    data = torch.rand(num_samples, 3, 64, 64)
    labels = torch.randint(0, 1000, (num_samples,))
    return data, labels


def main():
    # 1. 设备初始化（你无显卡，自动CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前运行设备：", device)

    # 2. 加载模型
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    print("ResNet50 模型加载完成")

    # 3. 初始化你写的攻击生成器
    attacker = AttackGenerator(eps=8/255)
    print("攻击工具初始化完成，支持FGSM、PGD")

    # 4. 加载数据（临时随机数据，队友后续替换真实数据集）
    all_imgs, all_labels = create_dummy_data(num_samples=64)
    all_imgs = all_imgs.to(device)
    all_labels = all_labels.to(device)

    # 5. 主循环：批量生成对抗样本
    total_success = 0
    total_num = len(all_imgs)
    batch = 16
    report_data = []

    for start in range(0, total_num, batch):
        end = min(start + batch, total_num)
        batch_img = all_imgs[start:end]
        batch_label = all_labels[start:end]

        # 调用你实现的FGSM方法（你的核心工作）
        adv_img = attacker.generate_fgsm(model, batch_img, batch_label)

        # 推理原始图片、对抗图片预测结果
        pred_origin = model(batch_img).argmax(dim=1)
        pred_adv = model(adv_img).argmax(dim=1)

        # 统计攻击成功样本（预测变化即攻击成功）
        success_mask = (pred_adv != pred_origin)
        total_success += int(success_mask.sum())

        # 保存单条样本信息
        for i in range(len(batch_img)):
            report_data.append({
                "origin_label": int(pred_origin[i]),
                "adv_label": int(pred_adv[i]),
                "attack_success": bool(success_mask[i])
            })

    # 6. 计算整体攻击成功率，输出结果
    success_rate = total_success / total_num
    print(f"FGSM攻击完成，总样本{total_num}，成功扰动{total_success}个，成功率：{success_rate:.2%}")

    # 7. 保存结果json文件，联调统一输出
    out_json = "fgsm_result.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "attack_type": "FGSM",
            "eps": 8/255,
            "success_rate": success_rate,
            "detail": report_data
        }, f, indent=2)
    print(f"测试报告已保存至 {out_json}")


if __name__ == "__main__":
    main()