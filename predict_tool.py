import random
import numpy as np
import torch


def model_predict(model, images):
     """
     模型预测函数，main.py 直接导入调用
     :param model: 加载好的 resnet50 模型
     :param images: 图片张量 [batch,3,H,W]
     :return: 每张图片预测类别索引 tensor
     """
     # 推理模式，关闭梯度节省资源
     model.eval()
     with torch.no_grad():
         outputs = model(images)
         # 取概率最大的类别
         pred_labels = torch.argmax(outputs, dim=1)
     return pred_labels
 # 配套模型加载包装类（替代你之前不匹配的ModelWrapper）


class ModelWrapper:
    """
    AI模型统一接口
    当前为Stub版本（占位实现）
    后续替换为真实PyTorch模型
    """

    def __init__(self):
        # 后续加载的模型
        self.model = None

        # 推理次数统计
        self.predict_counter = 0

        # 最大推理次数限制
        self.max_predict = 1000

    def load_model(self, model_path=None):
        """
        后续加载PyTorch模型
        当前为空实现   1
        """
        self.model = None

    def reset_counter(self):
        """
        重置推理计数
        """
        self.predict_counter = 0

    def get_features(self, vm_state):
        """
        根据VM状态提取特征

        当前返回随机特征
        后续替换为真实特征工程   2
        """

        return np.random.rand(64)

    def predict(self, features):
        """
        模型推理

        当前返回随机动作
        后续替换为真实模型输出   3
        """

        if self.predict_counter >= self.max_predict:
            raise RuntimeError("Inference limit exceeded (>1000).")

        self.predict_counter += 1

        action = random.randint(0, 9)

        confidence = random.random()

        return {
            "action": action,
            "confidence": confidence
        }