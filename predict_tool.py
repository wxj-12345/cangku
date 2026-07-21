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

# 配套模型加载包装类
class ModelWrapper:
    """
    AI模型统一接口占位类
    """
    def __init__(self):
        self.model = None
        self.predict_counter = 0