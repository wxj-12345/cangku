import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=0.05):
        self.eps = eps
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
        self.origin_train_state = False

    def _wrap_model(self, model):
        pre = dict(mean=self.mean, std=self.std)
        self.origin_train_state = model.training
        return fb.PyTorchModel(model, bounds=(0, 1), preprocessing=pre)

    def _restore_model(self, model):
        if self.origin_train_state:
            model.train()

    def generate_fgsm(self, imgs, labels, model):
        m = self._wrap_model(model)
        fgsm = fb.attacks.FGSM()
        _, img_adv, _ = fgsm(m, imgs, labels, epsilons=self.eps)
        img_adv = torch.clamp(img_adv, 0.0, 1.0)
        self._restore_model(model)
        # 生成对抗样本后切断梯度，避免残留梯度干扰后续循环波动
        return img_adv.detach()

    def generate_pgd(self, imgs, labels, model):
        m = self._wrap_model(model)
        # random_start=False 关闭随机初始化！关键！
        pgd = fb.attacks.PGD(steps=1, random_start=False)
        _, img_adv, _ = pgd(m, imgs, labels, epsilons=self.eps)
        self._restore_model(model)
        return img_adv.detach()

    def generate_bim(self, imgs, labels, model):
        m = self._wrap_model(model)
        bim = fb.attacks.LinfBasicIterativeAttack(steps=1)
        _, img_adv, _ = bim(m, imgs, labels, epsilons=self.eps)
        self._restore_model(model)
        return img_adv.detach()