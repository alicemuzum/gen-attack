import matplotlib.pyplot as plt
import torch
import math
from PIL import Image 
import os
PATH = "/home/acuzum/samproject/gen-attack/gen_attack/src/benchmarking/results"

class Visualize():
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def _build_figure(num_images):
        
        # Calculate the grid size
        grid_size = math.ceil(num_images ** 0.5)
        
        # Set up the figure
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        fig.subplots_adjust(hspace=0.5)

        return fig, axes, grid_size
    
    @staticmethod
    def visualize_resnet(images, true_labels, predicted_labels, accuracy, class_names=None):
        """
        Visualizes a given set of images with their true and predicted labels for resnet.

        Parameters:
        - images (torch.Tensor): A batch of images (Tensor of shape [batch_size, channels, height, width]).
        - true_labels (torch.Tensor): True labels of the images (Tensor of shape [batch_size]).
        - predicted_labels (torch.Tensor): Predicted labels of the images (Tensor of shape [batch_size]).
        - class_names (list, optional): List of class names corresponding to label indices. Default is None.
        """
        num_images = len(images)
        fig, axes, grid_size = Visualize._build_figure(num_images)
        
        for i in range(num_images):
            # Get the image, true label, and predicted label
            img = images[i].permute(1, 2, 0).cpu().detach().numpy()  # Convert from CHW to HWC
            true_label = true_labels[i].item()
            predicted_label = predicted_labels[i].item()

            # If class names are provided, use them
            if class_names:
                true_label_name = class_names[true_label].split(",")[0]
                predicted_label_name = class_names[predicted_label].split(",")[0]
            else:
                true_label_name = str(true_label)
                predicted_label_name = str(predicted_label)

            # Get the correct subplot
            ax = axes[i // grid_size, i % grid_size]
            ax.imshow(img)
            ax.set_title(f'True: {true_label_name}\nPred: {predicted_label_name}')
            ax.axis('off')

        plt.savefig(os.path.join(PATH, "resnet_results_images.jpg"))

        plt.figure(2)
        plt.plot(accuracy.keys(),accuracy.values())
        plt.savefig(os.path.join(PATH, "resnet_results_accuracy.jpg"))

    @staticmethod
    def visualize_detr(images, scores, pred_labels, boxes):
        """
        Visualizes images for detr model.

        Parameters
        images (PIL Image): image to be displayed
        scores (Float): Probability socre of predicted class
        pred_labels (int): Integer representing class 
        boxes (list of floats): Upper left corner coordinations and width and height of the bounding box. -> x,y,width,height
        """

        if not isinstance(images, Image.Image):
            raise ValueError("Image should be a PIL Image.")


        plt.figure(figsize=(10,10))
        plt.imshow(images)
        ax = plt.gca()
        
        
        for p, label, (xmin, ymin, xmax, ymax) in zip(scores, pred_labels, boxes):
            text = label + " " + str(round(p.item(),2))
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, linewidth=3, color = (1, 0, 0)))

            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
            
        plt.axis('off')
        plt.savefig(os.path.join(PATH, "detr_results.jpg"))


