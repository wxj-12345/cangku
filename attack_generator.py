import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=8/255):
        self.eps = eps
        self.fgsm_attack = fb.attacks.FGSM()
        self.pgd_attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=10, random_start=True)

    def return_origin(self, img):
        return img.clone()

    def generate_fgsm(self, model, images, labels):
        fb_model = fb.models.PyTorchModel(model, bounds=(0, 1))
        _, adv_img, _ = self.fgsm_attack(fb_model, images, labels, epsilons=self.eps)
        return adv_img

    def generate_pgd(self, model, images, labels):
        fb_model = fb.models.PyTorchModel(model, bounds=(0, 1))
        _, adv_img, _ = self.pgd_attack(fb_model, images, labels, epsilons=self.eps)
        return adv_img