"""
对抗攻击生成模块
负责生成FGSM、PGD对抗样本，封装攻击参数与执行逻辑
"""
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import foolbox as fb
from foolbox.attacks import LinfProjectedGradientDescentAttack, FGSM


class AttackGenerator:
    def __init__(self, eps=8/255):
        # 通用扰动参数
        self.eps = eps
        self.device = torch.device("cpu")  # 无显卡仅CPU运行
        # 初始化两种攻击
        self.pgd_attack = LinfProjectedGradientDescentAttack(steps=10, random_start=True)
        self.fgsm_attack = FGSM()  # FGSM单步攻击

    def generate_adv_sample(self, model, images, labels):
        """原有PGD生成对抗样本（保留，兼容之前代码）"""
        fmodel = fb.PyTorchModel(model, bounds=(0.0, 1.0), device=self.device)
        adv, _, _ = self.pgd_attack(fmodel, images, labels, epsilons=self.eps)
        return adv

    def generate_fgsm(self, model, images, labels):
        """
        表格要求新增：FGSM对抗样本生成函数
        :param model: resnet50模型
        :param images: 输入图片张量 [batch,3,H,W]
        :param labels: 图片真实标签
        :return: fgsm对抗样本图片
        """
        # Foolbox包装模型
        fmodel = fb.PyTorchModel(model, bounds=(0.0, 1.0), device=self.device)
        # 执行FGSM攻击
        adv_fgsm, _, _ = self.fgsm_attack(fmodel, images, labels, epsilons=self.eps)
        return adv_fgsm


# 测试入口，单独运行验证FGSM是否可用
if __name__ == "__main__":
    # 实例化攻击工具
    attacker = AttackGenerator(eps=8/255)
    # 加载resnet50模型
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT).to(attacker.device).eval()
    # 生成测试随机图片
    test_img = torch.rand(4, 3, 64, 64).to(attacker.device)
    test_label = torch.randint(0, 1000, (4,)).to(attacker.device)
    # 调用表格要求的generate_fgsm方法
    fgsm_adv_img = attacker.generate_fgsm(model, test_img, test_label)
    print("FGSM对抗样本生成完成，shape：", fgsm_adv_img.shape)