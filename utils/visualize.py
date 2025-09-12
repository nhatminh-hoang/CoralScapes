import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F

def visualize_sample(image, mask, class_names, label2color=None):
    """
    Visualize a single sample of image and its corresponding mask.

    Args:
        image (PIL Image or np.array): The input image.
        mask (np.array): The segmentation mask.
        class_names (dict): Mapping from class IDs to class names.
        label2color (dict): Mapping from class names to RGB colors.
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    
    # Create custom colormap if label2color is provided
    if label2color and class_names:
        colors = []
        # Add background color (black for class 0)
        colors.append([0, 0, 0])
        # Add colors for each class
        for class_id in sorted(class_names.keys()):
            if class_id == 0:
                continue  # Skip background, already added
            label_name = class_names[class_id]
            if label_name in label2color:
                # Normalize RGB values to [0,1]
                rgb = [c/255.0 for c in label2color[label_name]]
                colors.append(rgb)
            else:
                # Default color if not found
                colors.append([0.5, 0.5, 0.5])
        
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(colors)
        
        plt.imshow(mask, cmap=custom_cmap, vmin=0, vmax=len(class_names))
    else:
        sns.heatmap(mask, annot=True, fmt="d", cmap="viridis", cbar=False,
                    xticklabels=False, yticklabels=False)
    
    plt.title("Segmentation Mask")
    
    # Create a legend with actual colors if available
    if label2color and class_names:
        handles = []
        for class_id in sorted(class_names.keys()):
            label_name = class_names[class_id]
            if class_id == 0:
                color = [0, 0, 0]  # Black for background
            elif label_name in label2color:
                color = [c/255.0 for c in label2color[label_name]]
            else:
                color = [0.5, 0.5, 0.5]  # Gray for unknown
            
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    label=f"{class_id}: {label_name[:15]}",
                                    markerfacecolor=color, markersize=10))
        
        plt.legend(handles=handles[:10], bbox_to_anchor=(1.05, 1))  # Limit to 10 items
    else:
        # Fallback to generic legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                              markerfacecolor='gray', markersize=10) for i in class_names]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()

def visualize_batch(images, masks, class_names, batch_size=4, label2color=None):
    """
    Visualize a batch of images and their corresponding masks.

    Args:
        images (list of PIL Images or np.arrays): The input images.
        masks (list of np.arrays): The segmentation masks.
        class_names (dict): Mapping from class IDs to class names.
        batch_size (int): Number of samples to visualize from the batch.
        label2color (dict): Mapping from class names to RGB colors.
    """
    plt.figure(figsize=(15, 5 * batch_size))

    # Create custom colormap if label2color is provided
    if label2color and class_names:
        colors = []
        # Add background color (black for class 0)
        colors.append([0, 0, 0])
        # Add colors for each class
        for class_id in sorted(class_names.keys()):
            if class_id == 0:
                continue  # Skip background, already added
            label_name = class_names[class_id]
            if label_name in label2color:
                # Normalize RGB values to [0,1]
                rgb = [c/255.0 for c in label2color[label_name]]
                colors.append(rgb)
            else:
                # Default color if not found
                colors.append([0.5, 0.5, 0.5])
        
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(colors)
    else:
        custom_cmap = "viridis"

    for i in range(batch_size):
        plt.subplot(batch_size, 2, 2*i + 1)
        plt.imshow(images[i])
        plt.title(f"Input Image {i+1}")
        plt.axis('off')

        plt.subplot(batch_size, 2, 2*i + 2)
        if label2color and class_names:
            plt.imshow(masks[i], cmap=custom_cmap, vmin=0, vmax=len(class_names))
        else:
            sns.heatmap(masks[i], annot=True, fmt="d", cmap="viridis", cbar=False,
                        xticklabels=False, yticklabels=False)
        plt.title(f"Segmentation Mask {i+1}")

    # Create a legend with actual colors if available
    if label2color and class_names:
        handles = []
        for class_id in sorted(class_names.keys()):
            label_name = class_names[class_id]
            if class_id == 0:
                color = [0, 0, 0]  # Black for background
            elif label_name in label2color:
                color = [c/255.0 for c in label2color[label_name]]
            else:
                color = [0.5, 0.5, 0.5]  # Gray for unknown
            
            handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                    label=f"{class_id}: {label_name[:15]}",
                                    markerfacecolor=color, markersize=10))
        
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1))  # Limit to 10 items
    else:
        # Fallback to generic legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                              markerfacecolor='gray', markersize=10) for i in class_names]
        plt.legend(handles=handles, bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.show()

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

def visualize_predictions_with_gt(model, dataloader, device, num_samples=8, save_path=None, id2label=None, label2color=None):
    """
    Visualize model predictions compared with ground truth masks.
    
    Args:
        model: Trained model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        num_samples: Number of samples to visualize
        save_path: Path to save the visualization
        id2label: Dictionary mapping class IDs to labels
        label2color: Dictionary mapping labels to RGB colors
    """
    model.eval()
    
    # Create custom colormap from label2color if provided
    if label2color and id2label:
        colors = []
        # Add background color (black for class 0)
        colors.append([0, 0, 0])
        # Add colors for each class
        for class_id in range(1, len(id2label) + 1):
            if class_id in id2label:
                label_name = id2label[class_id]
                if label_name in label2color:
                    # Normalize RGB values to [0,1]
                    rgb = [c/255.0 for c in label2color[label_name]]
                    colors.append(rgb)
                else:
                    # Default color if not found
                    colors.append([0.5, 0.5, 0.5])
            else:
                colors.append([0.5, 0.5, 0.5])
        
        from matplotlib.colors import ListedColormap
        custom_cmap = ListedColormap(colors)
    else:
        custom_cmap = 'tab20'
    
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
                
                # Plot ground truth mask with custom colors
                im1 = axes[sample_count, 1].imshow(gt_mask, cmap=custom_cmap, vmin=0, vmax=len(id2label) if id2label else 39)
                axes[sample_count, 1].set_title(f'Ground Truth Mask {sample_count + 1}')
                axes[sample_count, 1].axis('off')
                
                # Plot predicted mask with custom colors
                im2 = axes[sample_count, 2].imshow(pred_mask, cmap=custom_cmap, vmin=0, vmax=len(id2label) if id2label else 39)
                axes[sample_count, 2].set_title(f'Predicted Mask {sample_count + 1}')
                axes[sample_count, 2].axis('off')
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Add colorbar with class labels if available
    # if id2label:
    #     cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
    #                        fraction=0.05, pad=0.1, aspect=50)
    #     cbar.set_label('Coral Species')
        
    #     # Set colorbar ticks to show class names
    #     tick_positions = list(range(0, len(id2label) + 1, 1))
    #     cbar.set_ticks(tick_positions)
    #     tick_labels = []
    #     for pos in tick_positions:
    #         if pos == 0:
    #             tick_labels.append('Background')
    #         elif pos in id2label:
    #             label = id2label[pos]
    #             # Truncate long labels
    #             tick_labels.append(label + '...' if len(label) > 15 else label)
    #         else:
    #             tick_labels.append(f'Class {pos}')
    #     cbar.set_ticklabels(tick_labels, rotation=45, ha='right')
    # else:
    #     cbar = fig.colorbar(im1, ax=axes, orientation='horizontal', 
    #                        fraction=0.05, pad=0.1, aspect=50)
    #     cbar.set_label('Class ID')
    
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