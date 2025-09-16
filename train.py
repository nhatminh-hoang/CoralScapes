import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.segmentation import MeanIoU

from model import UNet
from utils import get_free_vram
from utils import ds, num_classes, id2label, label2color
from utils import CoralSegmentationDataset, train_transform, val_transform, augment_transform
from utils import training_curve, visualize_predictions_with_gt, calculate_pixel_accuracy

class CoralSegmentationLightningModule(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=40, init_features=64, learning_rate=1e-4, model_name="UNet", batch_size=64):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=init_features)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.miou_metric = MeanIoU(num_classes=num_classes)
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Metrics tracking for direct saving
        self.train_losses = []
        self.val_losses = []
        self.train_mious = []
        self.val_mious = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.learning_rates = []  # Track learning rates
        
        # Step outputs for epoch aggregation
        self.train_step_outputs = []
        self.val_step_outputs = []
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_miou",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch

        augmented = augment_transform({"image": images, "mask": masks})
        images = augmented["image"]
        masks = augmented["mask"]

        outputs = self.forward(images)

        # Handle mask format - masks should be class indices for CrossEntropyLoss
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if present
        masks = masks.long()
        
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            miou = self.miou_metric(predictions, masks)
            accuracy = calculate_pixel_accuracy(outputs, masks)
        
        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_miou', miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('learning_rate', current_lr, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Store outputs for epoch end
        self.train_step_outputs.append({
            'loss': loss.detach().cpu(),
            'miou': miou.cpu(),
            'accuracy': accuracy,
            'lr': current_lr
        })
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        
        # Handle mask format - masks should be class indices for CrossEntropyLoss
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if present
        masks = masks.long()
        
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs, dim=1)
            miou = self.miou_metric(predictions, masks)
            accuracy = calculate_pixel_accuracy(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_miou', miou, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Store outputs for epoch end
        self.val_step_outputs.append({
            'loss': loss.detach().cpu(),
            'miou': miou.cpu(),
            'accuracy': accuracy
        })
        
        return loss
    
    def on_train_epoch_end(self):
        if not self.train_step_outputs:
            return
        
        # Calculate average metrics for the epoch
        epoch_loss = torch.stack([x['loss'] for x in self.train_step_outputs]).mean()
        epoch_miou = sum([x['miou'] for x in self.train_step_outputs]) / len(self.train_step_outputs)
        epoch_accuracy = sum([x['accuracy'] for x in self.train_step_outputs]) / len(self.train_step_outputs)
        
        # Get learning rate (same for all steps in epoch)
        epoch_lr = self.train_step_outputs[-1]['lr']
        
        # Store metrics for visualization
        self.train_losses.append(epoch_loss.cpu().item())
        self.train_mious.append(epoch_miou.cpu())
        self.train_accuracies.append(epoch_accuracy)
        self.learning_rates.append(epoch_lr)
        
        # Clear for next epoch
        self.train_step_outputs.clear()
    
    def on_validation_epoch_end(self):
        # Calculate epoch averages
        if self.val_step_outputs:
            avg_loss = torch.stack([x['loss'] for x in self.val_step_outputs]).mean().item()
            avg_miou = sum([x['miou'] for x in self.val_step_outputs]) / len(self.val_step_outputs)
            avg_accuracy = sum([x['accuracy'] for x in self.val_step_outputs]) / len(self.val_step_outputs)
            
            # Store metrics
            self.val_losses.append(avg_loss)
            self.val_mious.append(avg_miou)
            self.val_accuracies.append(avg_accuracy)
            
            print(f"Validation - Loss: {avg_loss:.4f}, mIoU: {avg_miou:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        # Clear outputs
        self.val_step_outputs.clear()
    
    def get_metrics_dict(self):
        """Get metrics dictionary for direct saving."""
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_mious': self.train_mious,
            'val_mious': self.val_mious,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
        }
    
    def predict_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.forward(images)
        predictions = torch.argmax(outputs, dim=1)
        
        # Handle mask format consistently
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if present
        
        return {'predictions': predictions, 'ground_truth': masks, 'images': images}

    def train_dataloader(self):
        """Create train dataloader."""
        train_dataset = CoralSegmentationDataset(ds["train"], transform=train_transform)
        return DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        val_dataset = CoralSegmentationDataset(ds["validation"], transform=val_transform)
        return DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            persistent_workers=True
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        test_dataset = CoralSegmentationDataset(ds["test"], transform=val_transform)
        return DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2,
            persistent_workers=True
        )

def train_lightning_model(
    in_channels=3,
    out_channels=40,
    init_features=64,
    learning_rate=1e-4,
    batch_size=64,
    num_epochs=200,
    model_name="UNet"
):
    """Train the model using PyTorch Lightning."""
    torch.set_float32_matmul_precision('high')

    # Create log directory
    log_dir = f"logs/{model_name}_features{init_features}_batch{batch_size}_epochs{num_epochs}_lr{learning_rate}"
    os.makedirs(log_dir, exist_ok=True)

    while get_free_vram()[0] < 10000 and torch.cuda.is_available():
        print("Waiting for GPU memory...")
        torch.cuda.empty_cache()
        import time
        time.sleep(300)
    
    # Initialize the Lightning module
    model = CoralSegmentationLightningModule(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        learning_rate=learning_rate,
        model_name=model_name,
        batch_size=batch_size
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename='best_model',
        monitor='val_miou',
        mode='max',
        save_top_k=1,
        verbose=True,
        save_last=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_miou',
        mode='max',
        patience=num_epochs // 2,
        verbose=True
    )
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name="lightning_logs",
        version=0
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        num_nodes=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_model_summary=False,
    )
    
    print(f"Training {model_name} for {num_epochs} epochs...")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Using device: {trainer.device_ids if trainer.device_ids else 'CPU'}")
    
    # Train the model
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        print("Training interrupted by user. Proceeding to load the best model and save metrics...")
    
    # Load the best model
    best_model = CoralSegmentationLightningModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        learning_rate=learning_rate,
        model_name=model_name,
        batch_size=batch_size
    )
    
    # Get metrics directly from the model
    metrics = model.get_metrics_dict()
    
    # Create traditional training curves and save metrics
    if metrics and metrics['train_losses']:
        create_traditional_plots(metrics, log_dir)
    else:
        print("No metrics available for plotting")
    
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best Validation mIoU: {metrics.get('best_miou', 0.0):.4f}")
    
    return best_model, metrics, log_dir

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

if __name__ == "__main__":
    

    # Hyperparameters
    num_epochs = 1000
    learning_rate = 1e-3
    batch_size = 32
    features = 64  # Initial number of features in UNet

    # Model name for logging
    model_name = f"UNet"

    # Train the model using Lightning
    trained_model, metrics, log_dir = train_lightning_model(
        in_channels=3,
        out_channels=num_classes,
        init_features=features,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        model_name=model_name
    )
    
    print(f"\n✅ Training completed! Results saved in: {log_dir}")
    
    # Generate prediction visualizations with ground truth comparison
    print("📊 Generating prediction visualizations...")
    
    # Set model to evaluation mode
    trained_model.eval()
    
    # Create test dataloader for visualization
    test_dataset = CoralSegmentationDataset(ds["test"], transform=val_transform, split="test")
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        persistent_workers=True
    )
    
    visualize_predictions_with_gt(
        model=trained_model.model,  # Extract the UNet model from Lightning module
        dataloader=test_dataloader,
        device=next(trained_model.parameters()).device,
        num_samples=3,
        save_path=f"{log_dir}/predictions_vs_gt.png",
        id2label=id2label,
        label2color=label2color
    )
    
    # Automatically update README with best results
    print("\n📝 Updating README.md with latest results...")
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
    
    print("\n🎉 Lightning training complete!")
    print(f"📁 Check TensorBoard logs: tensorboard --logdir {log_dir}/lightning_logs")
    print("=" * 60)