import numpy as np
import torch

def compute_accuracy(origin_pred, adv_pred):
    origin_pred = np.array(origin_pred)
    adv_pred = np.array(adv_pred)
    diff = origin_pred != adv_pred
    flip_count = diff.sum()
    total = len(origin_pred)
    attack_success_rate = flip_count / total
    clean_acc = 1.0
    adv_acc = 1 - attack_success_rate
    return attack_success_rate, int(flip_count), int(total), clean_acc, adv_acc

def compute_migration_retention(source_adv_acc, target_adv_acc):
    source_adv_acc = float(source_adv_acc)
    target_adv_acc = float(target_adv_acc)
    if source_adv_acc <= 0.0001:
        retention = 0.0
        migrate_fail = True
    else:
        retention = target_adv_acc / source_adv_acc
        migrate_fail = retention < 0.9
    # 转为Python原生bool，彻底解决JSON序列化报错
    return round(retention, 4), bool(migrate_fail)

def calc_fluctuation(acc_list):
    if len(acc_list) < 2:
        return 0.0
    arr = np.array(acc_list)
    fluct = abs(np.max(arr) - np.min(arr))
    return round(fluct, 4)