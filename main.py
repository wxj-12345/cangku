"""
项目主运行流程
串联数据、模型、对抗攻击、结果统计，供团队联调使用
你的工作：导入队友写好的三个函数，依次调用，拼接完整流程
"""
import torch
import json
from torchvision.models import resnet50, ResNet50_Weights

# 1. 导入队友写的三个函数（三个模块都是队友开发）
from data_tool import load_cifar10
from predict_tool import model_predict
from eval_tool import compute_accuracy

# 导入你自己完成的攻击模块
from attack_generator import AttackGenerator


def main():
    import time
    # 记录程序整体开始时间
    total_start = time.time()

    # 设备初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("当前运行设备：", device)

    # 加载图像识别模型
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    print("ResNet50 模型加载完成")

    # 初始化你实现的攻击工具
    attacker = AttackGenerator(eps=8 / 255)
    print("攻击工具初始化完成，支持FGSM、PGD")

    # ======================
    # 第一处：调用队友函数 load_cifar10
    # ======================
    test_loader = load_cifar10(batch_size=16)

    total_flip = 0
    all_sample_num = 0
    report_data = []

    # 循环分批读取图片（删除了多余嵌套循环）
    for batch_img, batch_label in test_loader:
        batch_img = batch_img.to(device)
        batch_label = batch_label.to(device)
        all_sample_num += batch_img.shape[0]

        # 调用你写的攻击方法生成对抗样本
        adv_img = attacker.generate_fgsm(model, batch_img, batch_label)

        # ======================
        # 第二处：调用队友函数 model_predict
        # ======================
        pred_origin = model_predict(model, batch_img)
        pred_adv = model_predict(model, adv_img)

        # ======================
        # 第三处：调用队友函数 compute_accuracy
        # ======================
        batch_rate, batch_flip, batch_len = compute_accuracy(pred_origin, pred_adv)
        total_flip += batch_flip

        # 记录每一条样本结果
        for i in range(batch_len):
            report_data.append({
                "origin_label": int(pred_origin[i]),
                "adv_label": int(pred_adv[i]),
                "attack_success": bool(pred_adv[i] != pred_origin[i])
            })

    # 整体攻击成功率汇总
    total_success_rate = total_flip / all_sample_num
    print(f"FGSM攻击完成，总样本{all_sample_num}，成功扰动{total_flip}个，成功率：{total_success_rate:.2%}")

    # 保存实验结果json文件
    save_path = "fgsm_result.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({
            "attack_type": "FGSM",
            "eps": 8 / 255,
            "total_samples": all_sample_num,
            "success_count": total_flip,
            "success_rate": total_success_rate,
            "detail": report_data
        }, f, indent=2)
    print(f"测试报告已保存至 {save_path}")
    # 全部数据处理完成后，计算并打印总运行时长
    total_run_time = time.time() - total_start
    print("========================================")
    print(f"项目完整运行总时长：{total_run_time:.2f} 秒")
    print("========================================")


if __name__ == "__main__":
    main()