import torch
import torchvision
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class FGSM:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def fgsm_attack(image, epsilon):
        image_grad = image.grad
        sign_data_grad = image_grad.sign()
        perturbed_image = image + epsilon * sign_data_grad
        return torch.clamp(perturbed_image, 0, 1)

    def evaluate(self, images, labels, epsilons):
        """
        Performs FGSM on given batch of images.

        Parameters:
        - images (torch.Tensor): A batch of images (Tensor of shape [batch_size, channels, height, width]).
        - labels (torch.Tensor): True labels of the images (Tensor of shape [batch_size]).
        - epsilons(List or float): Small value(s) that adjust the perturbation amount.
        """
        epsilons = list(epsilons)
        robustness = {}
        pred_labels = {}
        

        for epsilon in epsilons:
            correct = 0
            last_preds=[]
            perturbed_images = [] 
            for image, target in zip(images, labels):
                image = image.to(self.device)
                target = target.to(self.device)
                image.requires_grad = True

                output = self.model(image.unsqueeze(0))
                first_pred = output.max(1, keepdim=True)[1]

                loss = F.nll_loss(output, target.unsqueeze(0))
                self.model.zero_grad()
                loss.backward()

                perturbed_image = self.fgsm_attack(image, epsilon)
                perturbed_images.append(perturbed_image.cpu())
                output = self.model(perturbed_image.unsqueeze(0))
                last_pred = output.max(1, keepdim=True)[1]
                last_preds.append(last_pred)
                
                if first_pred == last_pred and target.item() == last_pred:
                    correct += 1

            pred_labels[epsilon] = last_preds
            robustness[epsilon] = correct / len(images)

        perturbed_images = torch.stack(perturbed_images)
        return robustness, pred_labels, perturbed_images 