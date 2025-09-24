import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.segmentation import MeanIoU

from model import get_model_and_processor
from utils import get_loss_fn, calculate_pixel_accuracy
from utils import get_free_vram
from utils import ds, id2label, label2color
from utils import CoralSegmentationDataset, create_augmentation_transforms
from utils import create_traditional_plots
from utils import visualize_test_predictions, create_class_colormap_figure
from utils import load_config, Config
from utils import DEFAULT_ARGS
from utils.dataloader import make_transform
from utils import ds

# Custom visualization callback
class VisualizationCallback(pl.Callback):
    def __init__(self, log_dir, config):
        self.log_dir = log_dir
        self.config = config
        self.best_metric = -float('inf') if DEFAULT_ARGS['checkpoint_mode'] == 'max' else float('inf')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # Get current metric
        current_metric = trainer.logged_metrics.get(DEFAULT_ARGS['checkpoint_monitor'], None)
        if current_metric is None:
            return
            
        # Check if this is a new best
        is_better = (DEFAULT_ARGS['checkpoint_mode'] == 'max' and current_metric > self.best_metric) or \
                    (DEFAULT_ARGS['checkpoint_mode'] == 'min' and current_metric < self.best_metric)
        
        if is_better:
            self.best_metric = current_metric
            print(f"🎯 New best {DEFAULT_ARGS['checkpoint_monitor']}: {current_metric:.4f}")
            
            # Generate visualizations
            self._generate_visualizations(pl_module)
    
    def _generate_visualizations(self, pl_module):
        """Generate test predictions visualization."""

        transform = make_transform(resize_size=self.config.dataset.input_size, config=self.config)
        test_dataset = CoralSegmentationDataset(ds["test"], transform=transform, split="test", config=self.config)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=DEFAULT_ARGS['num_workers'],
            persistent_workers=DEFAULT_ARGS['persistent_workers'],
            pin_memory=DEFAULT_ARGS['pin_memory']
        )
        
        # Generate visualization
        visualize_test_predictions(
            model=pl_module,
            test_dataloader=test_dataloader,
            device=pl_module.device,
            save_path=f"{self.log_dir}/test_predictions_best.png",
            num_samples=5
        )
        
        # Generate class colormap (only once)
        colormap_path = f"{self.log_dir}/class_colormap.png"
        if not os.path.exists(colormap_path):
            create_class_colormap_figure(save_path=colormap_path)
        
        print("✅ Visualizations updated!")

class CoralSegmentationLightningModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        
        # Set up model parameters from config
        if config.model.name == "UNet":
            model_config = config.model.unet
            model_config.out_channels = config.dataset.num_classes
        elif config.model.name == "DINOv3":
            model_config = config.model.dinov3
            model_config.num_labels = config.dataset.num_classes

        # Model
        self.model, self.processor = get_model_and_processor(
            config.model.name, 
            model_config, 
            pretrained_model_name=getattr(model_config, 'pretrained_model_name', None),
            num_labels=config.dataset.num_classes
        )

        # Loss function
        self.criterion = get_loss_fn(config.training.loss.name, config.training.loss.params)

        # Metrics
        self.miou_metric = MeanIoU(num_classes=config.dataset.num_classes)
        
        # Hyperparameters
        self.learning_rate = config.training.learning_rate
        self.model_name = config.model.name
        self.batch_size = config.training.batch_size
        
        # Create augmentation transforms from config
        self.augment_img, self.augment_mask = create_augmentation_transforms(config)
        
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
        """Configure optimizers and schedulers."""
        # Optimizer
        if self.config.training.optimizer.name == "Adam":
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay
            )
        elif self.config.training.optimizer.name == "AdamW":
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.config.training.optimizer.lr,
                weight_decay=self.config.training.optimizer.weight_decay
            )
        elif self.config.training.optimizer.name == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.config.training.optimizer.lr,
                momentum=getattr(self.config.training.optimizer, 'momentum', 0.9),
                weight_decay=self.config.training.optimizer.weight_decay
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.config.training.optimizer.lr)
        
        # Scheduler
        if self.config.training.scheduler.name == "CosineAnnealingWarmRestarts":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=self.config.training.scheduler.T_0,
                T_mult=self.config.training.scheduler.T_mult,
                eta_min=self.config.training.scheduler.eta_min
            )
        elif self.config.training.scheduler.name == "StepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=getattr(self.config.training.scheduler, 'step_size', 30),
                gamma=getattr(self.config.training.scheduler, 'gamma', 0.1)
            )
        elif self.config.training.scheduler.name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, 
                gamma=getattr(self.config.training.scheduler, 'gamma', 0.95)
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": DEFAULT_ARGS['checkpoint_monitor'],
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        images, masks = batch

        images = self.augment_img(images)
        masks = self.augment_mask(masks)

        if self.processor is not None:
            images = self.processor(images=images, return_tensors="pt")

        outputs = self.forward(images)

        # Handle mask format - masks should be class indices for CrossEntropyLoss
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if present
        masks = masks.long().contiguous()
        
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

        if self.processor is not None:
            images = self.processor(images=images, return_tensors="pt")
                
        outputs = self.forward(images)
        
        # Handle mask format - masks should be class indices for CrossEntropyLoss
        if len(masks.shape) == 4:
            masks = masks.squeeze(1)  # Remove channel dimension if present
        masks = masks.long().contiguous()
        
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
        
        # Print epoch summary
        current_epoch = self.current_epoch + 1
        print(f"Epoch {current_epoch:3d} - Training - Loss: {epoch_loss:.4f}, mIoU: {epoch_miou:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {epoch_lr:.2e}")
        
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
        transform = make_transform(resize_size=self.config.dataset.input_size, config=self.config)
        train_dataset = CoralSegmentationDataset(ds["train"], transform=transform, pre_compute=True, split="train", config=self.config)
        return DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=DEFAULT_ARGS['num_workers'],
            persistent_workers=DEFAULT_ARGS['persistent_workers'],
            pin_memory=DEFAULT_ARGS['pin_memory']
        )
    
    def val_dataloader(self):
        """Create validation dataloader."""
        transform = make_transform(resize_size=self.config.dataset.input_size, config=self.config)
        val_dataset = CoralSegmentationDataset(ds["validation"], transform=transform, pre_compute=True, split="validation", config=self.config)
        return DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=DEFAULT_ARGS['num_workers'],
            persistent_workers=DEFAULT_ARGS['persistent_workers'],
            pin_memory=DEFAULT_ARGS['pin_memory']
        )
    
    def test_dataloader(self):
        """Create test dataloader."""
        transform = make_transform(resize_size=self.config.dataset.input_size, config=self.config)
        test_dataset = CoralSegmentationDataset(ds["test"], transform=transform, pre_compute=True, split="test", config=self.config)
        return DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=DEFAULT_ARGS['num_workers'],
            persistent_workers=DEFAULT_ARGS['persistent_workers'],
            pin_memory=DEFAULT_ARGS['pin_memory']
        )

def train_lightning_model(config_path: str = "config/base_config.yaml"):
    """Train the model using PyTorch Lightning with config file."""
    
    # Load configuration
    config = load_config(config_path)
    
    torch.set_float32_matmul_precision('high')

    # Create log directory using hardcoded defaults
    log_dir = f"{DEFAULT_ARGS['save_dir']}/{config.model.name}/{config.logging['experiment_name']}"
    os.makedirs(log_dir, exist_ok=True)

    # GPU memory check using hardcoded threshold
    if torch.cuda.is_available() and DEFAULT_ARGS['gpu_memory_threshold'] > 0:
        while get_free_vram()[0] < DEFAULT_ARGS['gpu_memory_threshold']:
            print("Waiting for GPU memory...")
            torch.cuda.empty_cache()
            import time
            time.sleep(300)
    
    # Initialize the Lightning module
    model = CoralSegmentationLightningModule(config=config)

    # Callbacks with hardcoded defaults
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=DEFAULT_ARGS['checkpoint_filename'],
        monitor=DEFAULT_ARGS['checkpoint_monitor'],
        mode=DEFAULT_ARGS['checkpoint_mode'],
        save_top_k=DEFAULT_ARGS['checkpoint_save_top_k'],
        verbose=True,
        save_last=DEFAULT_ARGS['checkpoint_save_last']
    )
    
    callbacks = [checkpoint_callback]
    
    # Early stopping with hardcoded defaults
    if DEFAULT_ARGS['early_stopping_enable']:
        early_stopping = EarlyStopping(
            monitor=DEFAULT_ARGS['early_stopping_monitor'],
            mode=DEFAULT_ARGS['early_stopping_mode'],
            patience=DEFAULT_ARGS['early_stopping_patience'],
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Logger with hardcoded defaults
    if DEFAULT_ARGS['tensorboard_enable']:
        logger = TensorBoardLogger(
            save_dir=log_dir,
            name="lightning_logs"
        )
    else:
        logger = None

    # Add visualization callback
    visualization_callback = VisualizationCallback(logger.log_dir, config)
    callbacks.append(visualization_callback)
    
    # Trainer with hardcoded defaults
    trainer = pl.Trainer(
        max_epochs=config.training.num_epochs,
        callbacks=callbacks,
        logger=logger,
        accelerator=DEFAULT_ARGS['accelerator'],
        devices=DEFAULT_ARGS['devices'],
        precision=DEFAULT_ARGS['precision'] if torch.cuda.is_available() else 32,
        log_every_n_steps=DEFAULT_ARGS['log_every_n_steps'],
        check_val_every_n_epoch=DEFAULT_ARGS['check_val_every_n_epoch'],
        enable_progress_bar=DEFAULT_ARGS['enable_progress_bar'],
        enable_model_summary=DEFAULT_ARGS['enable_model_summary'],
        enable_checkpointing=DEFAULT_ARGS['enable_checkpointing']
    )
    
    print(f"Training {config.model.name} for {config.training.num_epochs} epochs...")
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
        config=config
    )
    
    # Get metrics directly from the model
    metrics = model.get_metrics_dict()
    
    # Create traditional training curves and save metrics
    if DEFAULT_ARGS['save_training_curves'] and metrics and metrics['train_losses']:
        create_traditional_plots(metrics, logger.log_dir)
    else:
        print("No metrics available for plotting")
    
    print("Training complete!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best Validation mIoU: {metrics.get('best_miou', 0.0):.4f}")
    
    return best_model, metrics, log_dir, config

if __name__ == "__main__":
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train CoralScapes Segmentation Model')
    parser.add_argument('--config', type=str, default='config/base_config.yaml',
                       help='Path to config file (default: config/base_config.yaml)')
    args = parser.parse_args()

    # Train the model using Lightning with config
    trained_model, metrics, log_dir, config = train_lightning_model(config_path=args.config)
    
    print(f"\n✅ Training completed! Results saved in: {log_dir}")
    
    # Generate final visualization summary
    print("📊 Generating final test visualization summary...")
    
    # Set model to evaluation mode
    trained_model.eval()

    # Create test dataloader for final visualization
    transform = make_transform(resize_size=config.dataset.input_size, config=config)
    test_dataset = CoralSegmentationDataset(ds["test"], transform=transform, split="test", config=config)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=DEFAULT_ARGS['num_workers'],
        persistent_workers=DEFAULT_ARGS['persistent_workers'],
        pin_memory=DEFAULT_ARGS['pin_memory']
    )
    
    # Generate final visualization
    visualize_test_predictions(
        model=trained_model.model,
        test_dataloader=test_dataloader,
        device=next(trained_model.parameters()).device,
        save_path=f"{log_dir}/final_test_predictions.png",
        num_samples=5
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
    print(f"📁 Check TensorBoard logs: tensorboard --logdir {log_dir}/lightning_logs")    # Automatically update README with best results