import os

import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import UNet
from utils import ds, num_classes, id2label
from utils import CoralSegmentationDataset, train_transform, val_transform
from utils import training_curve, visualize_predictions_with_gt, calculate_miou, calculate_pixel_accuracy

def forward_pass(model, images, masks, criterion, device):
    images = images.to(device)
    masks = masks.to(device)

    outputs = model(images)
    try:
        # For CrossEntropyLoss, we need class indices, not one-hot encoded masks
        mask_indices = torch.argmax(masks, dim=1)  # Convert from one-hot to class indices
        loss = criterion(outputs, mask_indices)
    except Exception as e:
        print("Error computing loss:", e)
        print("Outputs shape:", outputs.shape)
        print("Masks shape:", masks.shape)
        import sys; sys.exit(1)

    return loss, outputs

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=45, device='cuda', model_name="UNet"):
    model.to(device)
    best_val_loss = float('inf')
    best_miou = 0.0
    
    # Initialize tracking lists
    train_losses = []
    val_losses = []
    train_mious = []
    val_mious = []
    train_accuracies = []
    val_accuracies = []
    
    # Create logs directory
    log_dir = f"logs/{model_name}_epochs{num_epochs}_lr{optimizer.param_groups[0]['lr']}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training {model_name} for {num_epochs} epochs...")
    print(f"Logs will be saved to: {log_dir}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_miou = 0.0
        running_accuracy = 0.0
        num_batches = 0
        
        for images, masks in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            optimizer.zero_grad()
            loss, outputs = forward_pass(model, images, masks, criterion, device)

            loss.backward()
            optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                miou, _ = calculate_miou(outputs, masks, num_classes)
                accuracy = calculate_pixel_accuracy(outputs, masks)
            
            running_loss += loss.item()
            running_miou += miou
            running_accuracy += accuracy
            num_batches += 1

        # Calculate epoch averages
        epoch_loss = running_loss / num_batches
        epoch_miou = running_miou / num_batches
        epoch_accuracy = running_accuracy / num_batches
        
        train_losses.append(epoch_loss)
        train_mious.append(epoch_miou)
        train_accuracies.append(epoch_accuracy)
        print(f"  Training - Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_miou = 0.0
        val_running_accuracy = 0.0
        val_num_batches = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                loss, outputs = forward_pass(model, images, masks, criterion, device)
                
                # Calculate metrics
                miou, _ = calculate_miou(outputs, masks, num_classes)
                accuracy = calculate_pixel_accuracy(outputs, masks)

                val_running_loss += loss.item()
                val_running_miou += miou
                val_running_accuracy += accuracy
                val_num_batches += 1

        # Calculate validation epoch averages
        val_epoch_loss = val_running_loss / val_num_batches
        val_epoch_miou = val_running_miou / val_num_batches
        val_epoch_accuracy = val_running_accuracy / val_num_batches
        
        val_losses.append(val_epoch_loss)
        val_mious.append(val_epoch_miou)
        val_accuracies.append(val_epoch_accuracy)
        
        print(f"  Validation - Loss: {val_epoch_loss:.4f}, mIoU: {val_epoch_miou:.4f}, Accuracy: {val_epoch_accuracy:.4f}")

        # Save the best model based on mIoU
        if val_epoch_miou > best_miou:
            best_miou = val_epoch_miou
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), f'{log_dir}/best_model.pth')
            print(f"  ✓ Best model saved (mIoU: {best_miou:.4f})")
        
        print("-" * 60)

    print("Training complete!")
    print(f"Best Validation mIoU: {best_miou:.4f}")
    
    # Create and save training curves
    training_curve(train_losses, val_losses, train_mious, val_mious, 
                  train_accuracies, val_accuracies, save_path=f"{log_dir}/training_curves.png")
    
    # Save training metrics to file
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_mious': train_mious,
        'val_mious': val_mious,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_miou': best_miou,
        'best_val_loss': best_val_loss
    }
    
    torch.save(metrics, f"{log_dir}/training_metrics.pth")
    print(f"Training metrics saved to {log_dir}/training_metrics.pth")

    return model, metrics, log_dir

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    num_epochs = 45
    learning_rate = 1e-4
    batch_size = 64
    features = 64  # Initial number of features in UNet

    # Instantiate datasets and dataloaders
    train_dataset = CoralSegmentationDataset(ds["train"], transform=train_transform)
    val_dataset = CoralSegmentationDataset(ds["validation"], transform=val_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Instantiate model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=num_classes, init_features=64).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Model name for logging
    model_name = f"UNet_features{features}_batch{batch_size}"

    # Train the model
    trained_model, metrics, log_dir = train_model(
        model, train_dataloader, val_dataloader, criterion, optimizer,
        num_epochs=num_epochs, device=device, model_name=model_name
    )
    
    print(f"\nTraining completed! Results saved in: {log_dir}")
    
    # Generate prediction visualizations with ground truth comparison
    print("Generating prediction visualizations...")
    visualize_predictions_with_gt(
        model=trained_model,
        dataloader=val_dataloader,
        device=device,
        num_samples=8,
        save_path=f"{log_dir}/predictions_vs_gt.png",
        id2label=id2label
    )
    
    # Automatically update README with best results
    print("\nUpdating README.md with latest results...")
    try:
        from update_readme import find_best_experiment, copy_images_to_readme_assets, update_readme
        
        # Find the best experiment (including this one)
        best_log_dir, best_metrics = find_best_experiment()
        
        if best_log_dir:
            # Copy images to assets directory
            image_paths = copy_images_to_readme_assets(best_log_dir)
            
            # Update README
            update_readme(best_log_dir, best_metrics, image_paths)
            print("✅ README.md automatically updated with best results!")
        else:
            print("⚠️  Could not update README - no valid experiments found")
            
    except ImportError:
        print("⚠️  update_readme.py not found - skipping automatic README update")
    except Exception as e:
        print(f"⚠️  Error updating README: {e}")