import torch
import torchvision
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


class FGSM:
    def __init__(self):
        pass

    @staticmethod
    def fgsm_attack(image, epsilon):
        image_grad = image.grad
        image_grad_sign = image_grad.sign()
        perturbed_image = image + epsilon * image_grad_sign
        return torch.clamp(perturbed_image, 0, 1)
        