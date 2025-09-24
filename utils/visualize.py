import numpy as np 

from matplotlib import pyplot as plt

import torch

from utils import label2color, id2label

id_to_color_map = np.array([label2color[id2label[i]] for i in range(len(id2label))])  # (40, 3) including background

def denormalize_image(image, mean = np.array([0.485, 0.456, 0.406]), std = np.array([0.229, 0.224, 0.225])):
    """
    Denormalizes an image that was previously normalized using the given mean and standard deviation.
    Args:
        image (numpy.ndarray): The normalized image to be denormalized. Expected shape is (C, H, W).
        mean (numpy.ndarray, optional): The mean used for normalization. Default is np.array([0.485, 0.456, 0.406]) as used in ImageNet.
        std (numpy.ndarray, optional): The standard deviation used for normalization. Default is np.array([0.229, 0.224, 0.225]) as used in ImageNet.
    Returns:
        numpy.ndarray: The denormalized image with pixel values in the range [0, 255] and dtype uint8.
    """

    unnormalized_image = (image * std[:, None, None]) + mean[:, None, None]
    unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
    return unnormalized_image

def color_label(label):
    label = label.squeeze()
    label_colors = np.array([[id_to_color_map[pixel] for pixel in row] for row in np.array(label)])
    return label_colors

def color_correctness(label, pred):
    pred = pred.squeeze()
    label = label.squeeze()
    semseg = np.zeros((pred.shape[0], pred.shape[1], 3))
    semseg[pred==label] = np.array([0, 0, 1])
    semseg[pred!=label] = np.array([1, 0, 0])
    semseg[label==0] = 1
    return semseg

def show_samples(dataset, denormalize = True, n: int = 2):
    """
    Display n sample images and their corresponding segmentation maps from a dataset.
    Parameters:
    -----------
    dataset : Dataset
        The dataset from which to retrieve the samples. The dataset should support
        indexing and return either a tuple (image, label) or an object with attributes
        `transformed_image` and `transformed_segmentation_map`.
    denormalize : bool, optional
        If True, the images will be denormalize before displaying. Default is True.
    n : int, optional
        The number of samples to display. Default is 3.
    """

    if n > len(dataset):
        raise ValueError("n is larger than the dataset size")

    fig, ax = plt.subplots(n, 2, figsize=(10, 3 * n))

    for i in range(n):
        image, label = dataset.__getitem__(i)
        label_colors = color_label(label)
        if(denormalize and np.min(image)<0):
            image = denormalize_image(image)
        image = image.transpose(1, 2, 0)

        ax[i, 0].imshow(image)
        ax[i, 0].set_title("Image")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(image)
        ax[i, 1].imshow(label_colors, alpha=0.4)
        ax[i, 1].set_title("Segmentation Map")
        ax[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

# <=============================== Visualization functions for training and testing ===============================>
def training_curve(train_losses, val_losses, train_mious=None, val_mious=None, 
                  train_accuracies=None, val_accuracies=None, learning_rates=None, save_path=None):
    """
    Plot training and validation curves for loss, mIoU, accuracy, and learning rate.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_mious (list): List of training mIoU per epoch.
        val_mious (list): List of validation mIoU per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        learning_rates (list): List of learning rates per epoch.
        save_path (str): Path to save the plot.
    """
    # Find the minimum length among all metrics to ensure consistency
    min_length = len(train_losses)
    if val_losses:
        min_length = min(min_length, len(val_losses))
    if train_mious:
        min_length = min(min_length, len(train_mious))
    if val_mious:
        min_length = min(min_length, len(val_mious))
    if train_accuracies:
        min_length = min(min_length, len(train_accuracies))
    if val_accuracies:
        min_length = min(min_length, len(val_accuracies))
    if learning_rates:
        min_length = min(min_length, len(learning_rates))
    
    # Truncate all lists to the minimum length
    train_losses = train_losses[:min_length]
    val_losses = val_losses[:min_length]
    if train_mious:
        train_mious = train_mious[:min_length]
    if val_mious:
        val_mious = val_mious[:min_length]
    if train_accuracies:
        train_accuracies = train_accuracies[:min_length]
    if val_accuracies:
        val_accuracies = val_accuracies[:min_length]
    if learning_rates:
        learning_rates = learning_rates[:min_length]
    
    epochs = range(1, min_length + 1)
    
    # Determine number of subplots
    num_metrics = 1  # Loss is always present
    if train_mious is not None:
        num_metrics += 1
    if train_accuracies is not None:
        num_metrics += 1
    if learning_rates is not None:
        num_metrics += 1
    
    fig, axes = plt.subplots(1, num_metrics, figsize=(6 * num_metrics, 5))
    if num_metrics == 1:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    plot_idx = 1
    
    # mIoU plot
    if train_mious is not None and val_mious is not None:
        axes[plot_idx].plot(epochs, train_mious, 'b-', label='Training mIoU', linewidth=2)
        axes[plot_idx].plot(epochs, val_mious, 'r-', label='Validation mIoU', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('mIoU')
        axes[plot_idx].set_title('Training and Validation mIoU')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Accuracy plot
    if train_accuracies is not None and val_accuracies is not None:
        axes[plot_idx].plot(epochs, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        axes[plot_idx].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Accuracy')
        axes[plot_idx].set_title('Training and Validation Accuracy')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Learning rate plot
    if learning_rates is not None:
        axes[plot_idx].plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        axes[plot_idx].set_yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def create_traditional_plots(metrics, log_dir):
    """Create traditional training curves using existing visualization function and save metrics."""
    # Create training curves
    training_curve(
        metrics['train_losses'],
        metrics['val_losses'],
        metrics['train_mious'],
        metrics['val_mious'],
        metrics['train_accuracies'],
        metrics['val_accuracies'],
        metrics['learning_rates'],
        save_path=f"{log_dir}/training_curves.png"
    )
    
    # Save metrics for compatibility with existing README update system
    torch.save(metrics, f"{log_dir}/training_metrics.pth")
    print(f"Training metrics saved to {log_dir}/training_metrics.pth")

def visualize_test_predictions(model, test_dataloader, device, save_path=None, num_samples=5):
    """
    Visualize test predictions with ground truth comparison and comprehensive class index.
    
    Args:
        model: Trained model for inference
        test_dataloader: Test dataset dataloader
        device: Device to run inference on
        save_path: Path to save the visualization
        num_samples: Number of test samples to visualize
    """
    model.eval()
    
    # Get test samples
    images_list = []
    masks_list = []
    predictions_list = []
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_dataloader):
            if len(images_list) >= num_samples:
                break
                
            images = images.to(device)
            masks = masks.to(device)

            if model.processor is not None:
                # Preprocess images if a processor is defined
                inputs = model.processor(images=images, return_tensors="pt").to(device)
            else:
                inputs = images

            # Get predictions
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            # Move to CPU and convert to numpy
            images_cpu = images.cpu()
            masks_cpu = masks.cpu()
            predictions_cpu = predictions.cpu()
            
            for i in range(min(images.shape[0], num_samples - len(images_list))):
                # Denormalize image
                img = images_cpu[i].numpy()
                img_denorm = denormalize_image(img)
                img_denorm = img_denorm.transpose(1, 2, 0)
                
                # Get mask and prediction
                mask = masks_cpu[i].numpy()
                pred = predictions_cpu[i].numpy()
                
                images_list.append(img_denorm)
                masks_list.append(mask)
                predictions_list.append(pred)
    
    # Create visualization
    fig = plt.figure(figsize=(20, 6 * num_samples))
    
    # Main visualization area (left side)
    main_width = 0.75
    
    for i in range(num_samples):
        # Image
        ax1 = plt.subplot2grid((num_samples, 4), (i, 0))
        ax1.imshow(images_list[i])
        ax1.set_title(f"Sample {i+1}: Image")
        ax1.axis('off')
        
        # Ground Truth
        ax2 = plt.subplot2grid((num_samples, 4), (i, 1))
        gt_colors = color_label(masks_list[i])
        ax2.imshow(images_list[i])
        ax2.imshow(gt_colors, alpha=0.6)
        ax2.set_title("Ground Truth")
        ax2.axis('off')
        
        # Prediction
        ax3 = plt.subplot2grid((num_samples, 4), (i, 2))
        pred_colors = color_label(predictions_list[i])
        ax3.imshow(images_list[i])
        ax3.imshow(pred_colors, alpha=0.6)
        ax3.set_title("Prediction")
        ax3.axis('off')
        
        # Correctness
        ax4 = plt.subplot2grid((num_samples, 4), (i, 3))
        correctness = color_correctness(masks_list[i], predictions_list[i])
        ax4.imshow(images_list[i])
        ax4.imshow(correctness, alpha=0.6)
        ax4.set_title("Correctness (Blue=Correct, Red=Wrong)")
        ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Test predictions visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

def create_class_colormap_figure(save_path=None):
    """
    Create a comprehensive colormap showing all 39 classes (excluding background).
    
    Args:
        save_path: Path to save the colormap figure
    """
    # Get all classes except background (class 0)
    classes = [(i, id2label[i]) for i in range(1, 40)]
    
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(12, 20))
    
    # Create a grid to show colors
    rows = len(classes)
    
    for idx, (class_id, class_name) in enumerate(classes):
        # Get color for this class
        color = np.array(id_to_color_map[class_id]) / 255.0
        
        # Create a color patch
        y_pos = rows - idx - 1  # Reverse order to start from top
        ax.add_patch(plt.Rectangle((0, y_pos), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
        
        # Add text label
        ax.text(1.1, y_pos + 0.5, f"{class_id}: {class_name}", 
                va='center', ha='left', fontsize=10, weight='bold')
    
    ax.set_xlim(0, 4)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Class Colormap (39 Classes)', fontsize=16, weight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class colormap saved to: {save_path}")
        plt.close()
    else:
        plt.show()
