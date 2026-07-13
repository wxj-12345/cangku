from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def load_cifar10(batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    # 完整原始数据集
    full_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    total_all = len(full_dataset)

    # 赛题约束：使用数据不超过原始10%，同时匹配推理上限1000
    sample_count = 480
    print(f"完整数据集总量：{total_all}，本次测试选用样本：{sample_count}，数据占比：{sample_count / total_all:.2%}")

    # 随机无放回抽取，均衡样本分布，减少评估偏差
    select_idx = np.random.choice(total_all, sample_count, replace=False)
    small_dataset = Subset(full_dataset, select_idx)
    loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=False)
    return loader