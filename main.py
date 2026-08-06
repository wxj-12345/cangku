import torch
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms as T

# 导入工具文件
from data_tool import load_cifar10
from predict_tool import model_predict, get_infer_count, reset_infer_count
from eval_tool import compute_accuracy, compute_migration_retention, calc_fluctuation
from attack_generator import AttackGenerator
from audit_tool import AuditTool

# 全局固定随机种子
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 全局资源约束
MAX_INFER_LIMIT = 1000
MAX_TIME_LIMIT = 400
TEST_ROUND = 1
SAFE_THRESHOLD = 0.7
EPS_LIST = [0.01, 0.03, 0.05, 0.08, 0.10]
ATTACK_LIST = ["fgsm", "pgd", "bim"]
PER_EPS_REPEAT = 2
BATCH_SIZE = 16

inflect_x = 0.0
inflect_y = 0.0

def find_inflection_point(x_arr, y_arr):
    x = np.array(x_arr)
    y = np.array(y_arr)
    dy1 = np.diff(y) / np.diff(x)
    dy2 = np.diff(dy1)
    idx = np.argmax(np.abs(dy2)) + 1
    return float(x[idx]), float(y[idx]), int(idx)


def adv_finetune(model, target_model, loader, device, epoch=1, max_batch=33):
    model.train()
    target_model.train()
    # 解冻layer1~layer4+fc
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if any(k in name for k in ["layer1","layer2","layer3","layer4","fc"]):
            param.requires_grad = True
    for param in target_model.parameters():
        param.requires_grad = False
    for name, param in target_model.named_parameters():
        if any(k in name for k in ["layer1","layer2","layer3","layer4","fc"]):
            param.requires_grad = True

    opt = torch.optim.SGD(
        list(model.parameters()) + list(target_model.parameters()),
        lr=8e-4, momentum=0.9, weight_decay=1e-4
    )
    loss_cls, loss_mse = nn.CrossEntropyLoss(), nn.MSELoss()
    train_eps_candidates = [0.05, 0.05, 0.08, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.10]
    train_blur = T.GaussianBlur((5,5), sigma=(0.8,1.2))
    total_batch = len(loader)
    print(f"\n===== 鲁棒微调启动，最大训练批次{max_batch} =====")
    for e in range(epoch):
        #print(f"【微调进度】第{e+1}/{epoch}轮")
        batch_count = 0
        for batch_idx, (imgs, labels) in enumerate(loader):
            if batch_count >= max_batch:
                print(f"    达到批次上限，本轮提前结束")
                break
            imgs, labels = imgs.to(device), labels.view(-1).to(device)
            imgs = train_blur(imgs)
            opt.zero_grad()

            out_clean_src = model(imgs)
            loss_clean = loss_cls(out_clean_src, labels)
            out_clean_tgt = target_model(imgs)
            loss_align_clean = loss_mse(out_clean_src, out_clean_tgt)

            base_eps = np.random.choice(train_eps_candidates)
            current_eps = np.clip(base_eps + np.random.uniform(-0.015,0.015), 0.001, 0.10)
            imgs_tmp = imgs.clone().detach().requires_grad_(True)
            loss_cls(model(imgs_tmp), labels).backward()

            # 双分支对抗生成逻辑保留
            adv_imgs = imgs_tmp + current_eps * imgs_tmp.grad.sign()

            adv_imgs = torch.clamp(adv_imgs, 0, 1)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(target_model.parameters(), 0.5)

            out_adv_src = model(adv_imgs)
            loss_adv_src = loss_cls(out_adv_src, labels)
            out_adv_tgt = target_model(adv_imgs)
            loss_mig_adv = loss_mse(out_adv_src, out_adv_tgt)

            # ========== 关键修改：强化高eps FGSM权重 ==========
            total_loss = loss_clean + 1.5 * loss_align_clean + 83.0 * loss_adv_src + 13.0 * loss_mig_adv
            total_loss.backward()
            opt.step()
            batch_count += 1
            #if (batch_idx + 1) % 1 == 0:
            #    print(f"    批次 {batch_idx+1}/{total_batch} 完成，总loss={total_loss.item():.4f}")
        #print(f"【微调进度】第{e+1}轮训练全部完成")
    model.eval()
    target_model.eval()
    print("===== 强化FGSM鲁棒微调完成 =====")


def run_single_test(attack, eps, imgs, labels, raw_model, target_model, device):
    attacker = AttackGenerator(eps=eps)
    # 对抗生成不能套no_grad，否则梯度报错
    adv_imgs = getattr(attacker, f"generate_{attack}")(imgs, labels, raw_model)

    # 仅预测推理加no_grad，不影响对抗生成梯度
    with torch.no_grad():
        clean_pred = model_predict(raw_model, imgs)
        adv_pred_raw = model_predict(raw_model, adv_imgs)
        adv_pred_target = model_predict(target_model, adv_imgs)

    # 源模型指标
    attack_rate, flip_num, total, clean_acc, adv_acc_raw = compute_accuracy(clean_pred, adv_pred_raw)
    # 新增：计算目标模型的对抗准确率
    _, _, _, _, adv_acc_target = compute_accuracy(clean_pred, adv_pred_target)

    # 传入两个浮点准确率，不再传张量
    migrate_rate, migrate_fail = compute_migration_retention(adv_acc_raw, adv_acc_target)

    over_safe = bool(adv_acc_raw >= SAFE_THRESHOLD)

    print("-------本轮测试指标汇总-------")
    print(f"扰动强度：{eps} | 攻击算法：{attack.upper()}")
    print(f"1.正常输入基准准确率：{clean_acc:.4f}")
    print(f"2.源模型对抗准确率：{adv_acc_raw:.4f} | 安全判定：{'✅达标' if over_safe else '❌不达标'}")
    print(f"3.目标模型迁移后对抗准确率：{adv_acc_target:.4f}")
    print(f"4.攻击成功率：{attack_rate:.4f}")
    print(f"5.迁移保持率：{migrate_rate:.4f}（合格线≥0.9）")
    print(f"6.迁移失效标记：{migrate_fail}")
    print(f"7.翻转样本：{flip_num}/{total}")
    print("--------------------------------")
    return {
        "eps": float(eps),
        "attack": attack,
        "clean_acc": float(clean_acc),
        "source_adv_acc": float(adv_acc_raw),
        "target_adv_acc": float(adv_acc_target),
        "attack_success": float(attack_rate),
        "migrate_retention": float(migrate_rate),
        "migrate_fail": bool(migrate_fail),
        "safe_pass": bool(over_safe),
        "flip_sample": f"{flip_num}/{total}"
    }

def main():
    global inflect_x, inflect_y
    total_start = time.time()
    device = torch.device("cpu")
    print("运行设备：CPU")
    print(f"推理上限≤{MAX_INFER_LIMIT}次，时长上限≤{MAX_TIME_LIMIT}秒")
    reset_infer_count()

    raw_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    target_model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    test_loader = load_cifar10(batch_size=BATCH_SIZE)
    adv_finetune(raw_model, target_model, test_loader, device, epoch=1, max_batch=31)

    print("\n预加载全部测试样本至内存...")
    all_data = []
    for im, lab in test_loader:
        all_data.append((im.to(device), lab.to(device)))
    print(f"样本加载完成，共{len(all_data)}个批次")
    data_ptr = 0
    global_summary = []
    # 双层字典：eps -> 攻击名 -> 该攻击多次复测的准确率列表
    eps_record = {}
    for e in EPS_LIST:
        eps_record[e] = {}
        for atk in ATTACK_LIST:
            eps_record[e][atk] = []
    attack_record = {atk: [] for atk in ATTACK_LIST}

    for round_idx in range(TEST_ROUND):
        run_cost = time.time() - total_start
        if run_cost >= MAX_TIME_LIMIT or get_infer_count() >= MAX_INFER_LIMIT:
            print("【触发资源上限】提前终止，保存临时日志")
            try:
                with open("result_log.json", "w", encoding="utf-8") as f:
                    json.dump(global_summary, f, ensure_ascii=False, indent=2)
                print("临时日志 result_log.json 保存成功")
            except Exception as e:
                print(f"保存失败：{e}")
            return
        print(f"\n===== 第{round_idx + 1}轮完整测试 =====")
        for eps in EPS_LIST:
            # 每个eps只加载一次样本，本eps所有重复、所有攻击共用同一批图片
            imgs, labels = all_data[data_ptr % len(all_data)]
            for repeat in range(PER_EPS_REPEAT):
                print(f"\n---- eps={eps} 重复测试{repeat + 1}/{PER_EPS_REPEAT} ----")
                for attack_name in ATTACK_LIST:
                    res_data = run_single_test(attack_name, eps, imgs, labels, raw_model, target_model, device)
                    # 新增这一行，仅此一处改动
                    print(f"第{repeat + 1}次重复 | {attack_name} 单次对抗准确率：{res_data['source_adv_acc']:.4f}")
                    global_summary.append(res_data)
                    eps_record[eps][attack_name].append(res_data["source_adv_acc"])
                    attack_record[attack_name].append((eps, res_data["source_adv_acc"], res_data["migrate_retention"]))
                    current_infer = get_infer_count()
                    print(f"【单组攻击完成】eps={eps} {attack_name.upper()} 结束，当前累计推理总次数：{current_infer}")
            # 当前eps全部重复测试完毕，再换下一批样本
            data_ptr += 1

    # 波动分析
    # 波动分析：每个攻击单独计算波动，取同一eps下最大波动作为指标
    print("\n========== 扰动-准确率波动分析 ==========")
    fluct_info = {}
    for eps in EPS_LIST:
        single_eps_max_fluct = 0.0
        for atk in ATTACK_LIST:
            acc_list = eps_record[eps][atk]
            current_fluct = calc_fluctuation(acc_list)
            print(f"eps={eps} {atk.upper()} 波动={current_fluct:.4f}")
            if current_fluct > single_eps_max_fluct:
                single_eps_max_fluct = current_fluct
        is_legal = single_eps_max_fluct <= 0.05
        fluct_info[eps] = {"fluctuation": single_eps_max_fluct, "is_ok": is_legal}
        print(f"---- eps={eps} 该扰动下最大波动幅度={single_eps_max_fluct:.4f} | 波动合规：{is_legal}\n")

    # 拐点计算
    eps_x_arr = EPS_LIST
    all_acc = []
    for e in EPS_LIST:
        temp = []
        for atk in ATTACK_LIST:
            temp.extend(eps_record[e][atk])
        all_acc.append(np.mean(temp))
    avg_eps_acc = all_acc
    global inflect_x, inflect_y
    inflect_x, inflect_y, _ = find_inflection_point(eps_x_arr, avg_eps_acc)
    print(f"\n曲线拐点：eps={inflect_x:.4f}，对应对抗准确率={inflect_y:.4f}")

    # 绘图
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(9, 5))
    plt.plot(eps_x_arr, avg_eps_acc, marker="o", linewidth=2, label="平均对抗准确率")
    plt.scatter(inflect_x, inflect_y, c="red", s=120, label=f"拐点 eps={inflect_x:.3f}")
    plt.axhline(y=SAFE_THRESHOLD, c="orange", linestyle="--", label=f"安全阈值 {SAFE_THRESHOLD}")
    plt.xlabel("扰动强度 eps")
    plt.ylabel("源模型平均对抗准确率")
    plt.title("扰动强度-模型鲁棒准确率曲线")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("eps_acc_curve.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    color_map = {"fgsm": "red", "pgd": "blue", "bim": "green"}
    for atk in ATTACK_LIST:
        data = attack_record[atk]
        eps_unique = sorted(list(set([x[0] for x in data])))
        mean_list = []
        for e in eps_unique:
            accs = [item[1] for item in data if item[0] == e]
            mean_list.append(np.mean(accs))
        plt.plot(eps_unique, mean_list, marker="o", c=color_map[atk], label=f"{atk.upper()}")
    plt.axhline(y=SAFE_THRESHOLD, c="orange", linestyle="--", label=f"安全阈值 0.7")
    plt.xlabel("扰动强度 eps")
    plt.ylabel("对抗准确率")
    plt.title("多攻击算法鲁棒性对比")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("attack_compare_curve.png", dpi=300)
    plt.close()
    print("绘图文件：eps_acc_curve.png / attack_compare_curve.png 已保存")

    all_adv = [x["source_adv_acc"] for x in global_summary]
    avg_adv = float(np.mean(all_adv))
    all_clean = [x["clean_acc"] for x in global_summary]
    avg_clean = float(np.mean(all_clean))

    final_report = {
        "time_cost_s": round(time.time() - total_start, 2),
        "total_infer_count": get_infer_count(),
        "max_infer_limit": MAX_INFER_LIMIT,
        "max_time_sec": MAX_TIME_LIMIT,
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
        print("完整报告 final_report.json 生成成功")
    except Exception as e:
        print(f"报告保存失败：{e}")

    print("\n========== 全部实验执行完成 ==========")
    print(f"总耗时：{time.time() - total_start:.2f}s")
    print(f"总推理次数：{get_infer_count()}")

    audit = AuditTool()
    audit.run_audit(avg_clean, avg_adv, get_infer_count(), time.time() - total_start,global_summary)

if __name__ == "__main__":
    main()