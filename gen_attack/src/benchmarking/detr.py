import torch
import numpy as np
from transformers import DetrForObjectDetection , DetrImageProcessor
from PIL import Image
from transformers import logging
logging.set_verbosity_error()

import sys
sys.path.insert(1, '/home/acuzum/samproject/gen-attack/gen_attack/src/benchmarking/')
sys.path.insert(1, '/home/acuzum/samproject/gen-attack/gen_attack/src/adversarial_attacks/')
from visualize import Visualize
from fgsm import FGSM

class DETR():


    def __init__(self) -> None:

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(self.device)
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.image = Image.open("/home/acuzum/samproject/gen-attack/gen_attack/src/image-generation/examp.png")
        self.annotations = [
            {
            "image_id" : 0,
            "annotations" : [{"id": 0, 
                            "image_id" : 0,
                            "category_id" : 71,
                            "segmentation" : None,
                            "area" : None, 
                            "bbox": [64,97,445,242],
                            "iscrowd" : 0},
                            ]
            }
        ]


    def attack(self, attack_type:str):

        # Inputsu direct .to(device) yapamadıgım için tek tek attım gpuya
        inputs = self.processor.preprocess(self.image,self.annotations, return_tensors="pt")
        labels = [{k: v.to(self.device) for k, v in inputs["labels"][0].items()}]
        pixel_mask = inputs["pixel_mask"].to(self.device)
        pixel_values = inputs["pixel_values"].to(self.device)
        pixel_values.requires_grad = True

        outputs = self._forward(loss=True, pixel_values = pixel_values, pixel_mask = pixel_mask, labels = labels)
        self._evaluate(outputs)

        epsilon = 0.03
        if attack_type == "fgsm":
            fgsm = FGSM()
            perturbed_image = fgsm.fgsm_attack(pixel_values, epsilon)

        inputs_ = self.processor.preprocess(perturbed_image, return_tensors="pt", do_rescale=False).to(self.device)
        outputs_ = self._forward(**inputs_)
        self._evaluate(outputs_)

        
    def _forward(self, loss=False, **inputs):

        if "labels" not in inputs.keys():
            outputs = self.model(inputs["pixel_values"], inputs["pixel_mask"]) 
        else:
            outputs = self.model(inputs["pixel_values"], inputs["pixel_mask"], labels=inputs["labels"])

        if loss:
            self.model.zero_grad()
            outputs.loss.backward()

        return outputs

    
    def _evaluate(self, outputs):
        
        target_sizes = torch.tensor([self.image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):

            box = [round(i, 2) for i in box.tolist()]
            print(
                f"Detected {self.model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )


if __name__ == "__main__":
    detr = DETR()
    detr.attack('fgsm')
