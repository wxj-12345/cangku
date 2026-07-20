"""
项目完整测试代码，完全匹配赛题全部指标约束+周开发任务
1. 周一任务：FGSM/PGD/BIM三种对抗攻击完整测试
2. 周二任务：5档扰动强度[0.01,0.03,0.05,0.08,0.10]，每组重复2次保证一致性
3. 周三任务：扰动-准确率曲线拟合、二阶差分识别性能拐点
4. 指标全覆盖：基准准确率、对抗准确率、攻击退化幅度、迁移保持率、波动幅度、迁移失效标记、强扰动安全阈值校验
5. 资源硬约束：单次流程总推理≤1000次，总运行时长≤300s
6. 内置漏洞审计工具，自动输出合规评估图表与JSON完整报告
"""
import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

from data_tool import load_cifar10
from predict_tool import model_predict, get_infer_count, reset_infer_count
from eval_tool import compute_accuracy, compute_migration_retention, calc_fluctuation
from attack_generator import AttackGenerator
from audit_tool import CompetitionAuditor

# 全局资源约束（严格匹配文档）
MAX_INFER_LIMIT = 1000    # 推理调用上限1000次
MAX_TIME_LIMIT = 300      # 总运行时长上限300秒(5分钟)
TEST_ROUND = 3
SAFE_THRESHOLD = 0.70     # 强扰动最低性能安全阈值
# 周二扩充扰动强度列表
EPS_LIST = [0.01, 0.03, 0.05, 0.08, 0.10]
# 周一完善三类对抗攻击
ATTACK_LIST = ["fgsm", "pgd", "bim"]
PER_EPS_REPEAT = 2        # 每组扰动重复2次，保证结果一致性

# 周三任务：二阶差分算法识别性能曲线拐点
def find_inflection_point(x_arr, y_arr):
    x = np.array(x_arr)
    y = np.array(y_arr)
    # 一阶差分：斜率变化
    dy1 = np.diff(y) / np.diff(x)
    # 二阶差分：斜率的变化率，拐点判定依据
    dy2 = np.diff(dy1)
    inflect_idx = np.argmax(np.abs(dy2)) + 1
    inflect_x = x[inflect_idx]
    inflect_y = y[inflect_idx]
    return inflect_x, inflect_y, inflect_idx

def run_single_test(attack, eps, imgs, labels, raw_model, target_model, device):
    attacker = AttackGenerator(eps=eps)
    # 1. 正常干净样本推理（基准性能指标）
    clean_pred = model_predict(raw_model, imgs)
    curr_infer = get_infer_count()
    print(f"\n推理批次：本批16张，累计推理总数：{curr_infer}")

    # 推理次数超限拦截
    if curr_infer >= MAX_INFER_LIMIT:
        print(f"【触发约束】推理次数达到1000上限，停止本轮测试")
        return None

    print(f"扰动强度eps={eps}，执行{attack.upper()}对抗样本生成")
    # 生成对应攻击的对抗样本
    if attack == "fgsm":
        adv_imgs = attacker.generate_fgsm(imgs, labels, raw_model, device)
    elif attack == "pgd":
        adv_imgs = attacker.generate_pgd(imgs, labels, raw_model, device)
    elif attack == "bim":
        adv_imgs = attacker.generate_bim(imgs, labels, raw_model, device)
    else:
        raise Exception("仅支持fgsm/pgd/bim三种对抗攻击")

    # 2. 源模型对抗样本推理，计算攻击退化幅度
    adv_pred_raw = model_predict(raw_model, adv_imgs)
    attack_rate, flip_num, total_num, clean_acc, adv_acc_raw = compute_accuracy(clean_pred, adv_pred_raw)

    # 3. 目标模型推理，计算迁移性能
    target_clean_pred = model_predict(target_model, imgs)
    adv_pred_target = model_predict(target_model, adv_imgs)
    _, _, _, _, target_adv_acc = compute_accuracy(clean_pred, adv_pred_target)

    # 迁移保持率计算，判定迁移失效（阈值90%）
    migrate_ret, migrate_fail = compute_migration_retention(adv_acc_raw, target_adv_acc)
    migrate_fail = bool(migrate_fail)

    # 强扰动性能下限校验
    over_safe = adv_acc_raw >= SAFE_THRESHOLD

    # 输出全部规范指标
    print("----------本轮测试指标汇总----------")
    print(f"扰动强度：{eps} | 攻击算法：{attack.upper()}")
    print(f"1.正常输入基准准确率：{clean_acc:.4f}")
    print(f"2.攻击下有效性能(源模型对抗准确率)：{adv_acc_raw:.4f}")
    print(f"3.目标模型迁移后对抗准确率：{target_adv_acc:.4f}")
    print(f"4.性能退化幅度(攻击成功率)：{attack_rate:.4f}")
    print(f"5.迁移性能保持率：{migrate_ret:.4f}（合格线≥0.9）")
    print(f"6.迁移失效标记：{migrate_fail}")
    print(f"7.强扰动安全阈值{SAFE_THRESHOLD}，当前是否达标：{over_safe}")
    print(f"8.预测翻转样本：{flip_num}/{total_num}")
    print(f"批次结束总推理次数：{get_infer_count()}")
    print("------------------------------------")

    return {
        "攻击方式": attack,
        "扰动强度": float(eps),
        "干净基准准确率": float(clean_acc),
        "源模型对抗准确率": float(adv_acc_raw),
        "目标模型对抗准确率": float(target_adv_acc),
        "攻击退化幅度": float(attack_rate),
        "迁移保持率": float(migrate_ret),
        "迁移失效": migrate_fail,
        "强扰动性能达标": over_safe,
        "翻转样本数": flip_num,
        "单批样本总量": total_num,
        "批次起始推理计数": curr_infer,
        "批次结束推理计数": get_infer_count()
    }


def main():
    total_start = time.time()
    device = torch.device("cpu")
    print("运行设备：CPU")
    print(f"资源约束：推理调用≤{MAX_INFER_LIMIT}次，总运行时长≤{MAX_TIME_LIMIT}秒")
    reset_infer_count()

    # 加载源模型、目标模型
    raw_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    target_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    test_loader = load_cifar10(batch_size=16)
    test_iter = iter(test_loader)
    global_summary = []
    # 存储每组扰动多轮结果，用于波动幅度计算
    eps_record = {eps: [] for eps in EPS_LIST}

    for round_idx in range(TEST_ROUND):
        print(f"\n===== 第{round_idx+1}轮完整测试 =====")
        for eps in EPS_LIST:
            for repeat_idx in range(PER_EPS_REPEAT):
                print(f"\n---- 扰动强度eps={eps} 重复测试{repeat_idx+1}/{PER_EPS_REPEAT} ----")
                for attack_name in ATTACK_LIST:
                    # 全局时间约束拦截
                    run_time = time.time() - total_start
                    if run_time >= MAX_TIME_LIMIT:
                        print(f"\n【触发约束】总运行时间超过{MAX_TIME_LIMIT}秒上限，终止全部实验")
                        print(f"当前累计推理次数：{get_infer_count()}")
                        # 安全保存现有结果
                        with open("result_log.json", "w", encoding="utf-8") as f:
                            json.dump(global_summary, f, ensure_ascii=False, indent=2)
                        return
                    # 读取一批测试数据
                    try:
                        imgs, labels = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        imgs, labels = next(test_iter)
                    imgs, labels = imgs.to(device), labels.to(device)
                    # 执行单组对抗测试
                    res = run_single_test(attack_name, eps, imgs, labels, raw_model, target_model, device)
                    if res is not None:
                        global_summary.append(res)
                        eps_record[eps].append(res["源模型对抗准确率"])
                    else:
                        # 推理次数超限，保存数据并退出
                        with open("result_log.json", "w", encoding="utf-8") as f:
                            json.dump(global_summary, f, ensure_ascii=False, indent=2)
                        print("【触发约束】推理次数超限，全部实验终止")
                        return

    # =====================周三任务后处理：波动分析、拐点识别、曲线绘图=====================
    print("\n==========【后处理：性能波动、拐点计算】==========")
    fluct_info = {}
    # 计算每组扰动多轮测试的波动幅度（要求≤0.05）
    for eps, acc_list in eps_record.items():
        if len(acc_list) >= 2:
            fluct = calc_fluctuation(acc_list)
            fluct_info[eps] = {
                "多轮准确率记录": acc_list,
                "性能波动幅度": fluct,
                "波动合规(≤0.05)": fluct <= 0.05
            }
    print("各扰动强度性能波动校验：")
    for k, v in fluct_info.items():
        print(f"eps={k} | 波动幅度：{v['性能波动幅度']:.4f} | 是否合规：{v['波动合规(≤0.05)']}")

    # 提取平均准确率，拟合扰动-性能曲线，识别拐点
    eps_x = list(EPS_LIST)
    avg_acc_y = []
    for eps in EPS_LIST:
        avg_acc = np.mean(eps_record[eps]) if len(eps_record[eps])>0 else 0
        avg_acc_y.append(avg_acc)
    inflect_x, inflect_y, inflect_idx = find_inflection_point(eps_x, avg_acc_y)
    print(f"\n扰动-准确率曲线拐点：扰动值={inflect_x}，对应对抗准确率={inflect_y:.4f}")

    # 绘制扰动强度-鲁棒性能曲线
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 5))
    plt.plot(eps_x, avg_acc_y, marker="o", linewidth=2, label="平均对抗准确率")
    plt.scatter(inflect_x, inflect_y, color="red", s=100, label=f"拐点 eps={inflect_x}")
    plt.axhline(y=SAFE_THRESHOLD, color="orange", linestyle="--", label=f"安全阈值{SAFE_THRESHOLD}")
    plt.xlabel("扰动强度 Epsilon")
    plt.ylabel("源模型对抗准确率")
    plt.title("扰动强度-模型鲁棒性能变化曲线")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("eps_acc_curve.png", dpi=300)
    plt.close()
    print("扰动性能曲线已保存为 eps_acc_curve.png")

    # 汇总全部测试数据输出完整报告
    final_output = {
        "全局资源约束信息": {
            "最大推理次数限制": MAX_INFER_LIMIT,
            "最大运行时长限制(s)": MAX_TIME_LIMIT,
            "最终实际推理次数": get_infer_count(),
            "总运行耗时(s)": round(time.time()-total_start, 2),
            "强扰动安全阈值": SAFE_THRESHOLD,
            "迁移保持率合格线": 0.9,
            "性能波动合规上限": 0.05
        },
        "单批次全部测试原始记录": global_summary,
        "各扰动强度波动合规分析": fluct_info,
        "性能曲线拐点信息": {
            "拐点扰动强度": inflect_x,
            "拐点准确率": inflect_y
        }
    }
    with open("full_test_result.json", "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    print("\n==========全部实验流程完成==========")
    print(f"总运行耗时：{time.time()-total_start:.2f}s")
    print(f"最终累计推理次数：{get_infer_count()}")
    print("完整测试报告保存：full_test_result.json")
    print("扰动性能曲线图片：eps_acc_curve.png")

    # 调用审计工具，生成漏洞评估与合规图表
    auditor = CompetitionAuditor()
    all_clean_acc = [item["干净基准准确率"] for item in global_summary]
    all_adv_acc = [item["源模型对抗准确率"] for item in global_summary]
    avg_clean = np.mean(all_clean_acc)
    avg_adv = np.mean(all_adv_acc)
    total_infer_num = get_infer_count()
    total_time_cost = round(time.time() - total_start, 2)

    auditor.run_audit(
        clean_acc=avg_clean,
        attack_acc=avg_adv,
        query_count=total_infer_num,
        time_cost=total_time_cost
    )
    print("\n漏洞审计报告、评估图表已生成至 ./results/plots 目录")

if __name__ == "__main__":
    main()