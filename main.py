import torch
import json
import time
from torchvision.models import resnet50, ResNet50_Weights
from data_tool import load_cifar10
from predict_tool import ProtectedModel, model_predict
from attack_generator import AttackGenerator
from eval_tool import compute_accuracy

def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化受保护的模型（配额 1000）
    raw_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    model = ProtectedModel(raw_model, max_queries=1000)
    
    # 2. 加载数据（采样率 0.1 满足萌的要求，1000次查询约能跑100张图）
    test_loader = load_cifar10(batch_size=16, sample_ratio=0.1)
    
    # 3. 初始化攻击器
    attacker = AttackGenerator(model, eps=8/255)
    
    total_flip, total_samples = 0, 0
    report_data = []

    print(f"开始评测... 目标：推理次数 <= 1000")

    try:
        for batch_img, batch_label in test_loader:
            batch_img, batch_label = batch_img.to(device), batch_label.to(device)
            
            # 每一个 Batch 运行前，预判是否会撞线（16张图 * 10次/张 = 160次消耗）
            if model.query_count + 160 > 1000:
                print(">>> 剩余配额不足以完成完整 Batch，提前停止以守住红线。")
                break

            # 生成对抗样本（触发查询）
            adv_img = attacker.generate_pgd(batch_img, batch_label)
            
            # 原始预测与对抗预测（触发查询）
            pred_origin = model_predict(model, batch_img)
            pred_adv = model_predict(model, adv_img)
            
            # 统计
            _, flip, b_len = compute_accuracy(pred_origin, pred_adv)
            total_flip += flip
            total_samples += b_len
            
            print(f"已处理样本: {total_samples} | 当前累计推理次数: {model.query_count}")

    except RuntimeError as e:
        if "QUERY_LIMIT_EXCEEDED" in str(e):
            print(">>> [警告] 推理次数触碰 1000 次红线，系统自动熔断停止！")
        else:
            raise e

    # 4. 结算报告
    total_time = time.time() - start_time
    success_rate = total_flip / total_samples if total_samples > 0 else 0
    
    print("\n" + "="*40)
    print("         XH202616 任务结算报告")
    print("="*40)
    print(f"【莹】推理总次数: {model.query_count}次 (限额1000) -> {'✅ 合规' if model.query_count <= 1000 else '❌ 超标'}")
    print(f"【佳】运行总时长: {total_time:.2f}秒 (限额300) -> {'✅ 合规' if total_time <= 300 else '❌ 超标'}")
    print(f"【综合】攻击成功率: {success_rate:.2%}")
    print("="*40)

    # 保存 JSON
    with open("final_report.json", "w") as f:
        json.dump({"queries": model.query_count, "success_rate": success_rate, "time": total_time}, f)

if __name__ == "__main__":
    main()
