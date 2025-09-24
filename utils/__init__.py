from utils.tool import *
from utils.setup import ds, id2label, label2color
from utils.metrics import *
from utils.dataloader import CoralSegmentationDataset, create_augmentation_transforms
from utils.visualize import training_curve, create_traditional_plots, visualize_test_predictions, create_class_colormap_figure
from utils.loss import CombinedCrossEntropyDiceLoss, get_loss_fn
from utils.config_loader import load_config, Config

# Hardcoded default arguments
DEFAULT_ARGS = {
    'num_workers': 2,
    'persistent_workers': True,
    'pin_memory': True,
    'prefetch_factor': 2,
    'accelerator': 'auto',
    'devices': 'auto',
    'precision': 'bf16-mixed',
    'gpu_memory_threshold': 10000,
    'project_name': 'CoralScapes',
    'log_every_n_steps': 10,
    'save_dir': 'logs',
    'tensorboard_enable': True,
    'checkpoint_monitor': 'val_miou',
    'checkpoint_mode': 'max',
    'checkpoint_save_top_k': 1,
    'checkpoint_save_last': True,
    'checkpoint_filename': 'best_model',
    'early_stopping_enable': True,
    'early_stopping_monitor': 'val_miou',
    'early_stopping_mode': 'max',
    'early_stopping_patience': 50,
    'check_val_every_n_epoch': 1,
    'val_check_interval': 1.0,
    'enable_checkpointing': True,
    'enable_model_summary': True,
    'enable_progress_bar': True,
    'visualization_enable': True,
    'num_samples': 3,
    'save_predictions': True,
    'save_training_curves': True,
    'random_seed': 42,
    'deterministic': False,
    'benchmark': True
}