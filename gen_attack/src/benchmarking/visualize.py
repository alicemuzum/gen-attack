import matplotlib.pyplot as plt
import torch
import math

def visualize_images(images, true_labels, predicted_labels, class_names=None):
    """
    Visualizes a given set of images with their true and predicted labels.

    Parameters:
    - images (torch.Tensor): A batch of images (Tensor of shape [batch_size, channels, height, width]).
    - true_labels (torch.Tensor): True labels of the images (Tensor of shape [batch_size]).
    - predicted_labels (torch.Tensor): Predicted labels of the images (Tensor of shape [batch_size]).
    - class_names (list, optional): List of class names corresponding to label indices. Default is None.
    """
    # Ensure the number of images doesn't exceed the batch size
    num_images = len(images)
    
    # Calculate the grid size
    grid_size = math.ceil(num_images ** 0.5)
    
    # Set up the figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.5)

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

    plt.show()


