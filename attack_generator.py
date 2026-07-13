import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=12/255):
        self.eps = eps
        self.pgd_steps = 8
        self.fgsm_attack = fb.attacks.FGSM()
        self.pgd_attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=self.pgd_steps, random_start=True)

    def generate_fgsm(self, model, images, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)
        fixed_model = model.eval()
        fb_model = fb.PyTorchModel(fixed_model, bounds=(0, 1))
        adv_img, _, _ = self.fgsm_attack(fb_model, images, labels, epsilons=self.eps)
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        return adv_img

    def generate_pgd(self, model, images, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)
        fixed_model = model.eval()
        fb_model = fb.PyTorchModel(fixed_model, bounds=(0, 1))
        adv_img, _, _ = self.pgd_attack(fb_model, images, labels, epsilons=self.eps)
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        return adv_img