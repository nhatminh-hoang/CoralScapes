import torch

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