import torch
import torch.nn as nn
from torchvision import transforms

class ProtectedModel(nn.Module):
    def __init__(self, base_model, max_queries=1000):
        super().__init__()
        self.model = base_model
        self.query_count = 0
        self.max_queries = max_queries
        # CIFAR-10 标准归一化（这是提升识别准确率和攻击成功率的关键）
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 硬熔断检查：如果本次推理会导致总数超过1000，则报错
        if self.query_count + batch_size > self.max_queries:
            raise RuntimeError("QUERY_LIMIT_EXCEEDED")
        
        self.query_count += batch_size
        
        # 对输入进行归一化后传给原始模型
        x_norm = self.normalize(x)
        return self.model(x_norm)

def model_predict(model, images):
    """
    供 main.py 调用的预测函数
    """
    model.eval()
    with torch.no_grad():
        # 这里会触发 ProtectedModel 的 forward，从而计数
        outputs = model(images)
        return torch.argmax(outputs, dim=1)
