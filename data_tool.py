import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def load_cifar10(batch_size=16, sample_ratio=1.0):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    # 原路径不变，读取已下载数据集，不会重新下载
    test_dataset = datasets.CIFAR10(root="D:/PythonProject4/data", train=False, download=True, transform=transform)
    total_len = len(test_dataset)
    sample_len = int(total_len * sample_ratio)
    indices = np.random.choice(total_len, sample_len, replace=False)
    subset = Subset(test_dataset, indices)
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return test_loader