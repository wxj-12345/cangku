# config.py
import torch

# 比赛约束
LIMIT_QUERIES = 1000    # 莹的任务
LIMIT_TIME = 300        # 佳的任务 (5分钟)
LIMIT_DATA_RATIO = 0.1  # 萌的任务 (10%)

# 实验参数
EPSILON = 8/255
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
