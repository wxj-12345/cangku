import numpy as np
import json
import os

def compute_migration_metrics(source_acc, target_acc, history_accs=None):
    retention = target_acc / source_acc if source_acc > 0 else 0.0
    stability = float(np.std(history_accs)) if history_accs and len(history_accs) > 1 else 0.0
    worst = float(min(history_accs)) if history_accs else float(target_acc)
    failed = retention < 0.9
    return {
        "migration_retention": float(retention),
        "migration_stability": float(stability),
        "worst_migration_acc": float(worst),
        "migration_failed": bool(failed),
    }


def batch_compute_attack_metrics(clean_acc, attack_results):
    results = {}
    all_accs = []
    for result in attack_results:
        attack_name = result['attack_name']
        attack_acc = result['attack_acc']
        all_accs.append(float(attack_acc))
        retention = attack_acc / clean_acc if clean_acc > 0 else 0.0
        results[attack_name] = {
            'attack_accuracy': float(attack_acc),
            'performance_retention': float(retention),
            'performance_degradation': float(1.0 - retention),
            'params': result.get('params', {})
        }
    results['summary'] = {
        'worst_case_accuracy': float(min(all_accs)),
        'avg_attack_accuracy': float(np.mean(all_accs)),
        'num_attacks': len(attack_results)
    }
    return results


def load_experiment_data():
    """读取主程序输出 final_report.json"""
    json_path = "final_report.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError("请先运行主训练测试代码，生成 final_report.json！")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def test_scenario_1(exp_data):
    print("\n【场景一】迁移过程中性能异常波动")
    print("=" * 60)
    print("📊 基于真实运行数据 (CIFAR-10)")
    print("-" * 60)

    # 取PGD重复测试序列（你波动出问题的攻击）
    eps_record = {}
    for item in exp_data["all_test_data"]:
        eps = item["eps"]
        atk = item["attack"]
        if eps not in eps_record:
            eps_record[eps] = {}
        if atk not in eps_record[eps]:
            eps_record[eps][atk] = []
        eps_record[eps][atk].append(item["source_adv_acc"])

    # 选用eps=0.03 PGD复测序列
    history_accs = eps_record[0.03]["pgd"]
    source_acc = exp_data["all_test_data"][0]["clean_acc"]

    print(f"  源域基准准确率: {source_acc:.2%}")
    print(f"  观测序列：eps=0.03 PGD重复测试结果")
    print()

    anomaly_count = 0
    for round_num, target_acc in enumerate(history_accs, 1):
        metrics = compute_migration_metrics(source_acc, target_acc, history_accs[:round_num])
        retention = metrics['migration_retention']
        failed = metrics['migration_failed']
        change = target_acc - history_accs[round_num-2] if round_num > 1 else 0

        if failed:
            anomaly_count += 1
            status = "❌ 迁移失效"
        else:
            status = "✅ 正常"

        print(f"  第{round_num}轮: 目标域={target_acc:.2%}, "
              f"保持率={retention:.2%}, 变化={change:+.2%} {status}")

    print("-" * 60)
    fluct_val = exp_data["fluctuation_analysis"]["0.03"]["fluctuation"]
    print(f"  📌 当前序列波动幅度: {fluct_val:.4f}（阈值0.05）")
    print(f"  📌 异常检测: {'❌ 检测到 ' + str(anomaly_count) + ' 轮迁移失效' if anomaly_count > 0 else '✅ 无异常'}")
    print(f"  📌 结论: {'✅ 迁移波动合规' if fluct_val <=0.05 else '⚠️ 波动超标，存在稳定性风险'}")


def test_scenario_2(exp_data):
    print("\n【场景二】攻击或异常扰动条件下性能显著下降")
    print("=" * 60)
    print("📊 基于真实运行数据")
    print("-" * 60)

    clean_acc = exp_data["all_test_data"][0]["clean_acc"]
    attack_results = []
    for item in exp_data["all_test_data"]:
        eps = item["eps"]
        atk = item["attack"]
        name = f"{atk.upper()}_eps_{str(eps).replace('.','')}"
        attack_results.append({
            'attack_name': name,
            'attack_acc': item["source_adv_acc"],
            'params': {'eps': eps, 'method': atk}
        })

    results = batch_compute_attack_metrics(clean_acc, attack_results)

    print(f"  干净样本基准准确率: {clean_acc:.2%}")
    print()

    for eps in [0.01, 0.03, 0.05, 0.08, 0.10]:
        print(f"  ─── 扰动强度 eps={eps} ───")
        eps_results = [r for r in attack_results if r['params']['eps'] == eps]
        for r in eps_results:
            method = r['params']['method']
            acc = r['attack_acc']
            retention = acc / clean_acc
            status = "✅ 达标" if acc >= exp_data["safe_threshold"] else "⚠️ 退化"
            print(f"    {method}: 准确率={acc:.2%}, 保持率={retention:.2%} {status}")
        print()

    all_accs = [r['attack_acc'] for r in attack_results]
    worst_acc = min(all_accs)
    worst_retention = worst_acc / clean_acc
    avg_acc = sum(all_accs) / len(all_accs)
    safe_thresh = exp_data["safe_threshold"]

    print("-" * 60)
    print(f"  📊 汇总:")
    print(f"    最差对抗准确率: {worst_acc:.2%}")
    print(f"    平均对抗准确率: {avg_acc:.2%}")
    print(f"    安全阈值：{safe_thresh}")
    print(f"    安全阈值判定: {'✅ 达标' if worst_acc >= safe_thresh else '❌ 不达标'}")


def test_scenario_3(exp_data):
    print("\n【场景三】长期运行或条件变化引发的可靠性问题")
    print("=" * 60)
    print("📊 资源消耗实测数据")
    print("-" * 60)

    total_infer = exp_data["total_infer_count"]
    total_time = exp_data["time_cost_s"]
    max_infer_limit = exp_data["max_infer_limit"]
    max_time_limit = exp_data["max_time_sec"]

    run_data = [
        {'round': 1, 'inference': total_infer, 'time': total_time, 'memory': "CPU运行"},
    ]

    print(f"  {'轮次':<6} {'推理次数':<10} {'执行时间(s)':<12} {'备注':<12} {'状态'}")
    print(f"  {'-'*52}")

    for data in run_data:
        round_num = data['round']
        inference = data['inference']
        elapsed = data['time']
        memory = data['memory']
        status = "✅ 正常" if inference <= max_infer_limit and elapsed <= max_time_limit else "❌ 资源超标"
        print(f"  {round_num:<6} {inference:<10} {elapsed:<12.2f} {memory:<12} {status}")

    print("-" * 60)
    print(f"  📊 汇总:")
    print(f"    总推理次数: {total_infer}次 (上限{max_infer_limit})")
    print(f"    总执行时间: {total_time:.2f}s (上限300)")
    print(f"    推理约束判定: {'✅ 满足' if total_infer <= max_infer_limit else '❌超限'}")
    print(f"    时长约束判定: {'✅ 满足' if total_time <= max_time_limit else '❌超限'}")


def show_transfer_score(exp_data):
    print("\n" + "=" * 60)
    print("📊 迁移保持率综合评估")
    print("=" * 60)
    mig_rates = [x["migrate_retention"] for x in exp_data["all_test_data"]]
    avg_transfer = np.mean(mig_rates)
    min_transfer = np.min(mig_rates)
    print(f"  平均迁移保持率: {avg_transfer:.4f}")
    print(f"  最低迁移保持率: {min_transfer:.4f}")
    print(f"  合格线≥0.9 判定：{'✅全部达标' if min_transfer >=0.9 else '❌存在不满足样本'}")


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 三个典型测试场景（加载主程序实验真实数据）")
    print("=" * 60)
    try:
        experiment_result = load_experiment_data()
    except Exception as e:
        print(f"数据读取失败：{e}")
        print("操作顺序：先运行你的主训练代码，生成 final_report.json，再运行本文件！")
        exit()

    test_scenario_1(experiment_result)
    print()
    test_scenario_2(experiment_result)
    print()
    test_scenario_3(experiment_result)
    show_transfer_score(experiment_result)

    print("\n" + "=" * 60)
    print("✅ 所有场景测试完成")
    print("=" * 60)