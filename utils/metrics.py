import torch
import numpy as np

def calculate_miou(predictions, targets, num_classes):
    """
    Calculate mean Intersection over Union (mIoU) for semantic segmentation.
    
    Args:
        predictions: torch.Tensor of shape (N, num_classes, H, W) - model outputs
        targets: torch.Tensor of shape (N, num_classes, H, W) - ground truth one-hot encoded
        num_classes: int - number of classes
    
    Returns:
        miou: float - mean IoU across all classes
        class_ious: list - IoU for each class
    """
    # Convert predictions to class predictions
    pred_classes = torch.argmax(predictions, dim=1).to(predictions.device)  # (N, H, W)
    target_classes = torch.argmax(targets, dim=1).to(predictions.device)   # (N, H, W)

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred_classes == cls)
        target_cls = (target_classes == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = torch.tensor(1.0) if intersection == 0 else torch.tensor(0.0)
        else:
            iou = intersection / union
        
        ious.append(iou.item())
    
    return np.mean(ious), ious

def calculate_pixel_accuracy(predictions, targets):
    """
    Calculate pixel-wise accuracy for semantic segmentation.
    
    Args:
        predictions: torch.Tensor of shape (N, num_classes, H, W) - model outputs
        targets: torch.Tensor of shape (N, num_classes, H, W) - ground truth one-hot encoded
    
    Returns:
        accuracy: float - pixel-wise accuracy
    """
    pred_classes = torch.argmax(predictions, dim=1).to(predictions.device)  # (N, H, W)
    target_classes = torch.argmax(targets, dim=1).to(predictions.device)   # (N, H, W)

    correct = (pred_classes == target_classes).sum().float()
    total = target_classes.numel()
    
    return (correct / total).item()