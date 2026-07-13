import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, model_wrapper, eps=8/255):
        self.eps = eps
        # 初始化绑定模型，不要在循环里反复初始化 fb.PyTorchModel，否则运行极慢
        self.fmodel = fb.PyTorchModel(model_wrapper, bounds=(0, 1))
        
        # 将 steps 设为 7。每张图生成消耗 8 次查询。
        self.pgd_attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=7, random_start=True)

    def generate_pgd(self, images, labels):
        # 内部调用会消耗 model_wrapper 的 query_count
        _, adv_img, _ = self.pgd_attack(self.fmodel, images, labels, epsilons=self.eps)
        return torch.clamp(adv_img, 0.0, 1.0)
