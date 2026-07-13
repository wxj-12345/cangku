import random
import numpy as np
import torch

# 全局推理计数器
INFER_COUNT = 0

def get_infer_count():
    global INFER_COUNT
    return INFER_COUNT

def reset_infer_count():
    global INFER_COUNT
    INFER_COUNT = 0

def model_predict(model, images):
     """
     模型预测函数，每调用一次自动累加当前批次图片数量作为推理次数
     """
     global INFER_COUNT
     batch_size = images.shape[0]
     INFER_COUNT += batch_size

     model.eval()
     with torch.no_grad():
         outputs = model(images)
         pred_labels = torch.argmax(outputs, dim=1)
     return pred_labels

# 你原始的占位ModelWrapper，完整保留不修改
class ModelWrapper:
    def __init__(self):
        self.model = None
        self.predict_counter = 0
        self.max_predict = 1000

    def load_model(self, model_path=None):
        self.model = None

    def reset_counter(self):
        self.predict_counter = 0

    def get_features(self, vm_state):
        return np.random.rand(64)

    def predict(self, features):
        if self.predict_counter >= self.max_predict:
            raise RuntimeError("Inference limit exceeded (>1000).")
        self.predict_counter += 1
        action = random.randint(0, 9)
        confidence = random.random()
        return {
            "action": action,
            "confidence": confidence
        }