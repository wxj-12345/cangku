import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=0.05):
        self.eps = eps

    def _wrap_model(self, model):
        preprocess = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        fb_model = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=preprocess)
        return fb_model

    def generate_fgsm(self, imgs, labels, model):
        fb_m = self._wrap_model(model)
        fgsm = fb.attacks.FGSM()
        raw, clip_img, success = fgsm(fb_m, imgs, labels, epsilons=self.eps)
        return clip_img

    def generate_pgd(self, imgs, labels, model):
        fb_m = self._wrap_model(model)
        # 迭代2步，攻击力度极低
        pgd = fb.attacks.PGD(steps=2)
        raw, clip_img, success = pgd(fb_m, imgs, labels, epsilons=self.eps)
        return clip_img

    def generate_bim(self, imgs, labels, model):
        fb_m = self._wrap_model(model)
        bim = fb.attacks.LinfBasicIterativeAttack(steps=2)
        raw, clip_img, success = bim(fb_m, imgs, labels, epsilons=self.eps)
        return clip_img