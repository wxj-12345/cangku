from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def load_cifar10(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    full_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    total_all = len(full_dataset)

    # 0.0625采样，625张样本，40个批次，平衡鲁棒与推理上限
    np.random.seed(42)
    sample_count = int(total_all * 0.0625)
    indices = np.random.permutation(total_all)[:sample_count]
    sub_dataset = Subset(full_dataset, indices)

    print(f"完整数据集总量：{total_all}，本次测试选用样本：{sample_count}，数据占比：{sample_count / total_all:.2%}")
    loader = DataLoader(sub_dataset, batch_size=16, shuffle=False, drop_last=False)
    return loader