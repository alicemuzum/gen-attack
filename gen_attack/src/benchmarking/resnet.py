import torch
import torchvision
import foolbox as fb
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys
sys.path.insert(1, '/home/acuzum/samproject/gen-attack/gen_attack/src/adversarial_attacks/')
from fgsm import FGSM
from visualize import Visualize as V

class RESNET():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT').to(self.device)
        self.model = self.model.eval()

        preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
        bounds = (0, 1)

        # Get dataset
        fmodel = fb.PyTorchModel(self.model, bounds=bounds, preprocessing=preprocessing)
        self.images, self.labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)
        
    

    def attack(self, attack_type, epsilons):

        epsilons = list(epsilons)

        if attack_type == "fgsm":
            attacker = FGSM()
            accuracy, pred_labels, perturbed_images = self.evaluate(attacker.fgsm_attack, epsilons)
            with open('imagenet_class_names.txt') as f:
                classes = f.read()
            classes = eval(classes)
            V.visualize_resnet(perturbed_images, self.labels, list(pred_labels.values())[-1], accuracy, classes)

        #if attack_type == "pga":
            #attacker = PGA()
            #self.evaluate(attacker.pga_attack)        
        

    def evaluate(self, attack_fn, epsilons):
          
        robustness = {}
        pred_labels = {}
        
        for epsilon in epsilons:

            correct = 0
            last_preds=[]
            perturbed_images = [] 
            for image, target in zip(self.images, self.labels):
                
                image = image.to(self.device)
                target = target.to(self.device)
                image.requires_grad = True

                output = self.forward(image, target, loss=True)
                pred = output.max(1, keepdim=True)[1]

                perturbed_image = attack_fn(image, epsilon)
                perturbed_images.append(perturbed_image.cpu())

                output_ = self.forward(perturbed_image)
                pred_ = output_.max(1, keepdim=True)[1]
                last_preds.append(pred_)
                
                if pred == pred_ and target.item() == pred_:
                    correct += 1

            pred_labels[epsilon] = last_preds
            robustness[epsilon] = correct / len(self.images)

        perturbed_images = torch.stack(perturbed_images)
        return robustness, pred_labels, perturbed_images 


    def forward(self, image, target=None, loss=False):
        
        output = self.model(image.unsqueeze(0))

        if loss:
            loss = F.nll_loss(output, target.unsqueeze(0))
            self.model.zero_grad()
            loss.backward()

        return output
        

if "__main__" == __name__:
    epsilons = np.linspace(0.0, 0.005, num=20)#userdan epsilon değerleri alacak sekilde yap
    resnet = RESNET()
    resnet.attack("fgsm", epsilons)