"""
对抗攻击测试完整代码，修复梯度报错、JSON序列化报错、固定随机种子降低波动
周一：FGSM/PGD/BIM三类攻击；周二：5档扰动；周三：拐点识别+绘图
约束：推理≤1000次，时长≤300s
"""
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

# 固定随机种子，消除实验随机波动，保证结果稳定
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

from data_tool import load_cifar10
from predict_tool import model_predict, get_infer_count, reset_infer_count
from eval_tool import compute_accuracy, compute_migration_retention, calc_fluctuation
from attack_generator import AttackGenerator
from audit_tool import CompetitionAuditor


# 全局配置
MAX_INFER_LIMIT = 1000
MAX_TIME_LIMIT = 300
TEST_ROUND = 1
SAFE_THRESHOLD = 0.70
EPS_LIST = [0.01, 0.03, 0.05, 0.08, 0.10]
ATTACK_LIST = ["fgsm", "pgd", "bim"]
PER_EPS_REPEAT = 1  # 重复测试次数

# 二阶差分求拐点
def find_inflection_point(x_arr, y_arr):
    x = np.array(x_arr)
    y = np.array(y_arr)
    dy1 = np.diff(y) / np.diff(x)
    dy2 = np.diff(dy1)
    idx = np.argmax(np.abs(dy2)) + 1
    return x[idx], y[idx], idx

def run_single_test(attack, eps, imgs, labels, raw_model, target_model, device):
    attacker = AttackGenerator(eps=eps)
    # 干净样本推理：仅推理关闭梯度，提速
    with torch.no_grad():
        clean_pred = model_predict(raw_model, imgs)
    curr_infer = get_infer_count()
    print(f"\n推理批次：本批16张，累计推理总数：{curr_infer}")

    print(f"扰动强度eps={eps}，执行{attack.upper()}对抗样本生成")
    # 对抗样本生成必须开启梯度，不能加no_grad
    adv_imgs = getattr(attacker, f"generate_{attack}")(imgs, labels, raw_model)

    # 对抗推理关闭梯度
    with torch.no_grad():
        adv_pred_raw = model_predict(raw_model, adv_imgs)
        adv_pred_target = model_predict(target_model, adv_imgs)

    attack_rate, flip_num, total, clean_acc, adv_acc_raw = compute_accuracy(clean_pred, adv_pred_raw)
    _, _, _, _, adv_acc_target = compute_accuracy(clean_pred, adv_pred_target)
    migrate_rate, migrate_fail = compute_migration_retention(adv_acc_raw, adv_acc_target)
    over_safe = bool(adv_acc_raw >= SAFE_THRESHOLD)

    print("-------本轮测试指标汇总-------")
    print(f"扰动强度：{eps} | 攻击算法：{attack.upper()}")
    print(f"1.正常输入基准准确率：{clean_acc:.4f}")
    print(f"2.攻击下有效性能【源模型对抗准确率】：{adv_acc_raw:.4f}")
    print(f"3.目标模型迁移后对抗准确率：{adv_acc_target:.4f}")
    print(f"4.性能退化幅度(攻击成功率)：{attack_rate:.4f}")
    print(f"5.迁移性能保持率：{migrate_rate:.4f}（合格线≥0.9）")
    print(f"6.迁移失效标记：{migrate_fail}")
    print(f"7.强扰动安全阈值0.7，当前是否达标：{over_safe}")
    if not over_safe:
        print(f"   提示：源模型对抗准确率{adv_acc_raw:.4f} < {SAFE_THRESHOLD}，模型鲁棒性不足")
    print(f"8.预测翻转样本：{flip_num}/{total}")
    print(f"批次结束总推理次数：{get_infer_count()}")
    print("------------------------------")

    return {
        "eps": eps,
        "attack": attack,
        "clean_acc": clean_acc,
        "source_adv_acc": adv_acc_raw,
        "target_adv_acc": adv_acc_target,
        "attack_success": attack_rate,
        "migrate_keep": migrate_rate,
        "migrate_fail": migrate_fail,
        "safe_pass": over_safe,
        "flip_samples": f"{flip_num}/{total}"
    }

def main():
    total_start = time.time()
    device = torch.device("cpu")
    print("运行设备：CPU")
    print(f"资源约束：推理调用≤{MAX_INFER_LIMIT}次，总运行时长≤{MAX_TIME_LIMIT}秒")
    reset_infer_count()

    raw_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    target_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    test_loader = load_cifar10(batch_size=16)
    test_iter = iter(test_loader)

    global_summary = []
    eps_record = {eps: [] for eps in EPS_LIST}

    for round_idx in range(TEST_ROUND):
        run_time = time.time() - total_start
        if run_time >= MAX_TIME_LIMIT or get_infer_count() >= MAX_INFER_LIMIT:
            print("【触发约束】资源上限到达，提前终止实验，保存现有数据")
            try:
                with open("result_log.json", "w", encoding="utf-8") as f:
                    json.dump(global_summary, f, ensure_ascii=False, indent=2)
                print("临时数据保存成功：result_log.json")
            except Exception as e:
                print(f"保存JSON文件失败，错误信息：{e}")
            return
        print(f"\n===== 第{round_idx+1}轮完整测试 =====")
        for eps in EPS_LIST:
            for repeat in range(PER_EPS_REPEAT):
                print(f"\n---- 扰动强度eps={eps} 重复测试{repeat+1}/{PER_EPS_REPEAT} ----")
                try:
                    imgs, labels = next(test_iter)
                except StopIteration:
                    test_iter = iter(test_loader)
                    imgs, labels = next(test_iter)
                imgs, labels = imgs.to(device), labels.to(device)
                for attack_name in ATTACK_LIST:
                    res_data = run_single_test(attack_name, eps, imgs, labels, raw_model, target_model, device)
                    global_summary.append(res_data)
                    eps_record[eps].append(res_data["source_adv_acc"])
                    if time.time() - total_start >= MAX_TIME_LIMIT or get_infer_count() >= MAX_INFER_LIMIT:
                        print("【触发约束】资源超限，终止运行")
                        try:
                            with open("result_log.json", "w", encoding="utf-8") as f:
                                json.dump(global_summary, f, ensure_ascii=False, indent=2)
                            print("临时数据保存成功：result_log.json")
                        except Exception as e:
                            print(f"保存JSON文件失败，错误信息：{e}")
                        return

    # 后处理波动分析
    print("\n==========【后处理：扰动-准确率分析】==========")
    fluct_info = {}
    for eps, acc_list in eps_record.items():
        if len(acc_list) >= 2:
            fluct = calc_fluctuation(acc_list)
            fluct_info[eps] = {
                "acc_list": acc_list,
                "fluctuation": fluct,
                "is_ok": bool(fluct <= 0.05)  # 强制转为Python原生bool
            }
            print(f"eps={eps} | 波动幅度：{fluct:.4f} | 波动合规：{fluct <= 0.05}")

    # 拐点计算
    eps_x = EPS_LIST
    avg_acc_y = [np.mean(eps_record[e]) for e in EPS_LIST]
    inflect_x, inflect_y, _ = find_inflection_point(eps_x, avg_acc_y)
    print(f"\n扰动-准确率曲线拐点：eps={inflect_x:.4f}，对应对抗准确率={inflect_y:.4f}")

    # 绘图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9, 5))
    plt.plot(eps_x, avg_acc_y, marker="o", linewidth=2, label="平均对抗准确率")
    plt.scatter(inflect_x, inflect_y, c="red", s=120, label=f"拐点 eps={inflect_x:.3f}")
    plt.axhline(y=SAFE_THRESHOLD, c="orange", linestyle="--", label=f"安全阈值 {SAFE_THRESHOLD}")
    plt.xlabel("扰动强度 eps")
    plt.ylabel("源模型平均对抗准确率")
    plt.title("扰动强度-模型鲁棒准确率关系曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("eps_acc_curve.png", dpi=300)
    plt.close()
    print("曲线图已保存 eps_acc_curve.png")

    # 输出完整报告
    final_report = {
        "time_cost_s": round(time.time() - total_start, 2),
        "total_infer_count": get_infer_count(),
        "max_infer_limit": MAX_INFER_LIMIT,
        "max_time_limit": MAX_TIME_LIMIT,
        "safe_threshold": SAFE_THRESHOLD,
        "eps_list": EPS_LIST,
        "attack_types": ATTACK_LIST,
        "fluctuation_analysis": fluct_info,
        "curve_inflection": {"eps": inflect_x, "acc": inflect_y},
        "all_test_data": global_summary
    }
    try:
        with open("final_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)
        print("完整测试报告输出成功：final_report.json")
    except Exception as e:
        print(f"完整报告保存失败，错误：{e}")

    print("\n==========全部实验执行完成==========")
    print(f"总运行耗时：{time.time()-total_start:.2f}s")
    print(f"总推理次数：{get_infer_count()}")

    auditor = CompetitionAuditor()

    # 【关键修复】中文键名改为英文，和返回字典对应
    all_clean_acc = [
        item["clean_acc"]
        for item in global_summary
    ]

    all_attack_acc = [
        item["source_adv_acc"]
        for item in global_summary
    ]

    avg_clean_acc = np.mean(all_clean_acc)
    avg_attack_acc = np.mean(all_attack_acc)
    infer_count = get_infer_count()
    run_time = time.time() - total_start

    auditor.run_audit(
        clean_acc=avg_clean_acc,
        attack_acc=avg_attack_acc,
        query_count=infer_count,
        time_cost=run_time
    )

if __name__ == "__main__":
    main()