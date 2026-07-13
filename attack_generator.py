import foolbox as fb
import torch

class AttackGenerator:
    def __init__(self, eps=8/255):
        self.eps = eps
        self.fgsm_attack = fb.attacks.FGSM()
        self.pgd_attack = fb.attacks.LinfProjectedGradientDescentAttack(steps=10, random_start=True)

    def generate_fgsm(self, model, images, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)

        fb_model = fb.PyTorchModel(model, bounds=(0, 1))

        adv_img, _ , _= self.fgsm_attack(fb_model, images, labels, epsilons=self.eps)
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        return adv_img


        # 2. 生成之后，再限制像素范围 (Clamp)
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        # --- 修改结束 ---

        return adv_img

    def generate_pgd(self, model, images, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = images.to(device)
        labels = labels.to(device)

        fb_model = fb.PyTorchModel(model, bounds=(0, 1))
        adv_img = self.pgd_attack(fb_model, images, labels, epsilons=self.eps)
        adv_img = torch.clamp(adv_img, 0.0, 1.0)
        return adv_img