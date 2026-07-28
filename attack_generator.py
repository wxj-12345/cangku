import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=0.05):
        self.eps = eps
        self.mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)

    def _wrap_model(self, model):
        pre = dict(mean=self.mean, std=self.std)
        return fb.PyTorchModel(model, bounds=(0, 1), preprocessing=pre)

    def generate_fgsm(self, imgs, labels, model):
        m = self._wrap_model(model)
        fgsm = fb.attacks.FGSM()
        _, img_adv, _ = fgsm(m, imgs, labels, epsilons=self.eps)
        return img_adv

    def generate_pgd(self, imgs, labels, model):
        m = self._wrap_model(model)
        pgd = fb.attacks.PGD(steps=2) # 2步，大幅提速
        _, img_adv, _ = pgd(m, imgs, labels, epsilons=self.eps)
        return img_adv

    def generate_bim(self, imgs, labels, model):
        m = self._wrap_model(model)
        bim = fb.attacks.LinfBasicIterativeAttack(steps=2)
        _, img_adv, _ = bim(m, imgs, labels, epsilons=self.eps)
        return img_adv