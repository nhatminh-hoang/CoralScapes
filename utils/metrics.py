import torch

def calculate_pixel_accuracy(predictions, targets):
    """
    Calculate pixel-wise accuracy for semantic segmentation.
    
    Args:
        predictions: torch.Tensor of shape (N, num_classes, H, W) - model outputs
        targets: torch.Tensor of shape (N, H, W) - ground truth class indices
    
    Returns:
        accuracy: float - pixel-wise accuracy
    """
    pred_classes = torch.argmax(predictions, dim=1).to(predictions.device)  # (N, H, W)

    correct = (pred_classes == targets).sum().float()
    total = targets.numel()

    return (correct / total).item()