import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

def visualize_sample(image, mask, class_names):
    """
    Visualize a single sample of image and its corresponding mask.

    Args:
        image (PIL Image or np.array): The input image.
        mask (np.array): The segmentation mask.
        class_names (dict): Mapping from class IDs to class names.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    sns.heatmap(mask, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=False, yticklabels=False)
    plt.title("Segmentation Mask")
    
    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                          markerfacecolor='gray', markersize=10) for i in class_names]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def visualize_batch(images, masks, class_names, batch_size=4):
    """
    Visualize a batch of images and their corresponding masks.

    Args:
        images (list of PIL Images or np.arrays): The input images.
        masks (list of np.arrays): The segmentation masks.
        class_names (dict): Mapping from class IDs to class names.
        batch_size (int): Number of samples to visualize from the batch.
    """
    plt.figure(figsize=(15, 5 * batch_size))

    for i in range(batch_size):
        plt.subplot(batch_size, 2, 2*i + 1)
        plt.imshow(images[i])
        plt.title(f"Input Image {i+1}")
        plt.axis('off')

        plt.subplot(batch_size, 2, 2*i + 2)
        sns.heatmap(masks[i], annot=True, fmt="d", cmap="viridis", cbar=False,
                    xticklabels=False, yticklabels=False)
        plt.title(f"Segmentation Mask {i+1}")

    # Create a legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                          markerfacecolor='gray', markersize=10) for i in class_names]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

def training_curve(train_losses, val_losses, train_mious=None, val_mious=None, 
                  train_accuracies=None, val_accuracies=None, save_path=None):
    """
    Plot training and validation curves for loss, mIoU, and accuracy.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_mious (list): List of training mIoU per epoch.
        val_mious (list): List of validation mIoU per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
        save_path (str): Path to save the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Determine number of subplots
    num_metrics = 1  # Loss is always present
    if train_mious is not None:
        num_metrics += 1
    if train_accuracies is not None:
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
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()

def visualize_predictions_with_gt(model, dataloader, device, num_samples=8, save_path=None, id2label=None):
    """
    Visualize model predictions compared with ground truth masks.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        id2label: Dictionary mapping class IDs to labels
    """
    model.eval()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    sample_count = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            if sample_count >= num_samples:
                break
                
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                if sample_count >= num_samples:
                    break
                
                # Get individual image, mask, and prediction
                image = images[i].cpu()
                gt_mask = torch.argmax(masks[i], dim=0).cpu().numpy()
                pred_mask = predictions[i].cpu().numpy()
                
                # Denormalize image for visualization
                image = image * torch.tensor([0.200, 0.194, 0.198]).view(3, 1, 1) + torch.tensor([0.306, 0.508, 0.491]).view(3, 1, 1)
                image = torch.clamp(image, 0, 1)
                image = image.permute(1, 2, 0).numpy()
                
                # Plot original image
                axes[sample_count, 0].imshow(image)
                axes[sample_count, 0].set_title(f'Original Image {sample_count + 1}')
                axes[sample_count, 0].axis('off')
                
                # Plot ground truth mask
                im1 = axes[sample_count, 1].imshow(gt_mask, cmap='tab20', vmin=0, vmax=39)
                axes[sample_count, 1].set_title(f'Ground Truth Mask {sample_count + 1}')
                axes[sample_count, 1].axis('off')
                
                # Plot predicted mask
                im2 = axes[sample_count, 2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=39)
                axes[sample_count, 2].set_title(f'Predicted Mask {sample_count + 1}')
                axes[sample_count, 2].axis('off')
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
                       fraction=0.05, pad=0.1, aspect=50)
    cbar.set_label('Class ID')
    
    # Add accuracy information
    with torch.no_grad():
        total_correct = 0
        total_pixels = 0
        
        for images, masks in dataloader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            gt_classes = torch.argmax(masks, dim=1)
            
            correct = (predictions.cpu() == gt_classes).sum().item()
            total_correct += correct
            total_pixels += gt_classes.numel()
        
        overall_accuracy = total_correct / total_pixels
        fig.suptitle(f'Predictions vs Ground Truth (Overall Accuracy: {overall_accuracy:.4f})', 
                    fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison saved to: {save_path}")
    
    plt.show()