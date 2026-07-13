import numpy as np

# ==========================================
# 第一部分：compute_accuracy（修复兼容列表/数组）
# ==========================================
def compute_accuracy(origin_pred, adv_pred):
    # 统一转为numpy数组，兼容list、numpy数组输入
    origin_pred = np.array(origin_pred)
    adv_pred = np.array(adv_pred)
    # 对比标签是否发生变化
    diff = origin_pred != adv_pred
    flip_count = diff.sum()
    total = len(origin_pred)
    success_rate = flip_count / total
    # 返回三元组：攻击成功率、翻转样本数量、总样本数
    return success_rate, int(flip_count), int(total)


# ==========================================
# 第二部分：sample_data（已有）
# ==========================================
def sample_data(images, labels, ratio=0.1):
    """
    从完整数据集中随机抽取 ratio 比例的数据
    images: 所有图片数据，numpy数组，形状为 [N, 32, 32, 3]
    labels: 所有标签，numpy数组，形状为 [N]
    ratio: 抽取比例，默认0.1（即10%）
    返回: 抽取后的图片和标签
    """
    num_samples = len(images)
    sample_count = int(num_samples * ratio)
    indices = np.random.choice(num_samples, sample_count, replace=False)
    sampled_images = images[indices]
    sampled_labels = labels[indices]
    return sampled_images, sampled_labels


# ==========================================
# 第三部分：compute_all_metrics（已有，攻击鲁棒性指标）
# ==========================================
def compute_all_metrics(clean_acc, attack_acc, all_attack_accs=None):
    """
    计算所有评价指标（攻击鲁棒性）
    clean_acc: 干净样本上的准确率
    attack_acc: 攻击样本上的准确率
    all_attack_accs: 多个攻击强度下的准确率列表，用于算最差性能
    """
    retention = attack_acc / clean_acc if clean_acc > 0 else 0
    degradation = 1 - retention
    worst_case = min(all_attack_accs) if all_attack_accs else attack_acc

    return {
        "clean_accuracy": clean_acc,
        "attack_accuracy": attack_acc,
        "performance_retention": retention,
        "performance_degradation": degradation,
        "worst_case_accuracy": worst_case,
    }


# ==========================================
# 第四部分：compute_migration_metrics（新增！迁移学习指标）
# ==========================================
def compute_migration_metrics(source_acc, target_acc, history_accs=None):
    """
    计算迁移学习评估指标
    source_acc: 源域准确率（迁移前）
    target_acc: 目标域准确率（迁移后）
    history_accs: 多次迁移的历史准确率列表，用于计算稳定性

    返回: 迁移保持率、稳定性、最差性能、迁移失败标记
    """
    # 迁移后有效性能保持率（比赛要求≥90%）迁移后性能是迁移前的百分之多少
    retention = target_acc / source_acc if source_acc > 0 else 0

    # 迁移稳定性：多次迁移的标准差（比赛要求波动±5%以内）
    stability = np.std(history_accs) if history_accs and len(history_accs) > 1 else 0

    # 最差迁移性能
    worst = min(history_accs) if history_accs else target_acc

    # 迁移失败检测：保持率<90% 或 性能骤降
    failed = retention < 0.9

    return {
        "migration_retention": retention,           # 迁移保持率（目标≥0.9）
        "migration_stability": stability,           # 稳定性（标准差，目标±0.05）
        "worst_migration_acc": worst,               # 最差迁移性能
        "migration_failed": failed,                 # 是否迁移失败
    }


# ==========================================
# 第五部分：用假数据测试所有函数（修复打印逻辑）
# ==========================================
if __name__ == "__main__":
    print("=" * 40)
    print("测试 eval_tool.py 所有函数（假数据）")
    print("=" * 40)

    # 测试 compute_accuracy
    print("\n[测试1] compute_accuracy")
    fake_preds = [0, 1, 1, 0, 0]
    fake_labels = [0, 1, 0, 0, 0]
    success_rate, flip_count, total = compute_accuracy(fake_preds, fake_labels)
    print(f"  攻击成功率: {success_rate}，翻转样本数：{flip_count}，总样本：{total} (期望成功率: 0.2)")

    # 测试 sample_data
    print("\n[测试2] sample_data")
    fake_images = np.random.randn(100, 32, 32, 3)
    fake_labels = np.random.randint(0, 10, size=100)
    sampled_imgs, sampled_lbls = sample_data(fake_images, fake_labels, 0.1)
    print(f"  原始: 100张 → 采样后: {len(sampled_imgs)}张 (期望: 10张)")

    # 测试 compute_all_metrics
    print("\n[测试3] compute_all_metrics（攻击鲁棒性）")
    clean_acc = 0.92
    attack_acc = 0.78
    all_accs = [0.78, 0.75, 0.72]
    metrics = compute_all_metrics(clean_acc, attack_acc, all_accs)
    print(f"  性能保持率: {metrics['performance_retention']:.3f}")
    print(f"  退化幅度: {metrics['performance_degradation']:.3f}")
    print(f"  最差性能: {metrics['worst_case_accuracy']:.3f}")

    # ===== 新增测试：compute_migration_metrics =====
    print("\n[测试4] compute_migration_metrics（迁移学习）")
    source_acc = 0.95
    target_acc = 0.88
    history = [0.90, 0.88, 0.92, 0.87]  # 多次迁移历史
    mig = compute_migration_metrics(source_acc, target_acc, history)
    print(f"  迁移保持率: {mig['migration_retention']:.3f} (期望≥0.9，实际: {'达标' if mig['migration_retention'] >= 0.9 else '不达标'})")
    print(f"  迁移稳定性: {mig['migration_stability']:.3f} (标准差，期望≤0.05)")
    print(f"  最差迁移性能: {mig['worst_migration_acc']:.3f}")
    print(f"  迁移失败: {mig['migration_failed']}")

    # 测试迁移失败场景
    print("\n[测试5] 迁移失败场景")
    bad_target = 0.80
    mig_bad = compute_migration_metrics(source_acc, bad_target)
    print(f"  保持率: {mig_bad['migration_retention']:.3f} → 迁移失败: {mig_bad['migration_failed']}")

    print("\n" + "=" * 40)
    print("所有测试通过！✅")
    print("=" * 40)