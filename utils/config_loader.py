import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DatasetConfig:
    name: str = "CoralScapes"
    input_size: List[int] = field(default_factory=lambda: [128, 256])
    num_classes: int = 40
    normalization: Dict[str, List[float]] = field(default_factory=lambda: {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    })

@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 40
    init_features: int = 64

@dataclass
class DINOv3Config:
    pretrained_model_name: str = "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    input_size: int = 256
    hidden_size: int = 64
    tokenW: int = 49
    tokenH: int = 16

@dataclass
class ModelConfig:
    name: str = "UNet"
    unet: UNetConfig = field(default_factory=UNetConfig)
    dinov3: DINOv3Config = field(default_factory=DINOv3Config)

@dataclass
class OptimizerConfig:
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

@dataclass
class SchedulerConfig:
    name: str = "CosineAnnealingWarmRestarts"
    T_0: int = 100
    T_mult: int = 2
    eta_min: float = 1e-6

@dataclass
class LossConfig:
    name: str = "cross_entropy+dice"
    params: Dict[str, float] = field(default_factory=lambda: {
        "dice_weight": 0.5,
        "focal_alpha": 1.0,
        "focal_gamma": 2.0
    })

@dataclass
class TrainingConfig:
    num_epochs: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)

@dataclass
class AugmentationConfig:
    train: Dict[str, Any] = field(default_factory=lambda: {
        "enable": True,
        "random_crop": {"enable": True, "size": [128, 128]},
        "random_horizontal_flip": {"enable": True, "probability": 0.5},
        "random_rotation": {"enable": True, "degrees": 45},
        "random_affine": {"enable": True, "probability": 0.5, "degrees": 20, "translate": [0.05, 0.05]}
    })

@dataclass
class Config:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    
    # Hardcoded defaults for system settings
    dataloader: Dict[str, Any] = field(default_factory=lambda: {
        "num_workers": 2,
        "persistent_workers": True,
        "pin_memory": True,
        "prefetch_factor": 2
    })
    hardware: Dict[str, Any] = field(default_factory=lambda: {
        "device": "auto",
        "precision": "bf16-mixed",
        "gpu_memory_threshold": 10000
    })
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "project_name": "CoralScapes",
        "experiment_name": "baseline",
        "log_every_n_steps": 10,
        "save_dir": "logs",
        "tensorboard": {"enable": True, "log_images": True, "log_histograms": False},
        "checkpoint": {"monitor": "val_miou", "mode": "max", "save_top_k": 1, "save_last": True, "filename": "best_model"},
        "early_stopping": {"enable": True, "monitor": "val_miou", "mode": "max", "patience": 100}
    })
    validation: Dict[str, Any] = field(default_factory=lambda: {
        "check_val_every_n_epoch": 1,
        "val_check_interval": 1.0
    })
    visualization: Dict[str, Any] = field(default_factory=lambda: {
        "enable": True,
        "num_samples": 3,
        "save_predictions": True,
        "save_training_curves": True
    })
    paths: Dict[str, str] = field(default_factory=lambda: {
        "dataset_dir": "dataset",
        "model_dir": "saved_model",
        "log_dir": "logs",
        "output_dir": "outputs"
    })
    environment: Dict[str, Any] = field(default_factory=lambda: {
        "random_seed": 42,
        "deterministic": False,
        "benchmark": True
    })

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file with hardcoded defaults."""
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    return create_config_from_dict(yaml_data)

def create_config_from_dict(yaml_data: Dict[str, Any]) -> Config:
    """Create Config object from dictionary with simplified structure."""
    config = Config()
    
    # Dataset config
    if 'dataset' in yaml_data:
        dataset_data = yaml_data['dataset']
        config.dataset = DatasetConfig(
            name=dataset_data.get('name', config.dataset.name),
            input_size=dataset_data.get('input_size', config.dataset.input_size),
            num_classes=dataset_data.get('num_classes', config.dataset.num_classes),
            normalization=dataset_data.get('normalization', config.dataset.normalization)
        )
    
    # Model config
    if 'model' in yaml_data:
        model_data = yaml_data['model']
        config.model = ModelConfig(
            name=model_data.get('name', config.model.name)
        )
        
        if 'unet' in model_data:
            unet_data = model_data['unet']
            config.model.unet = UNetConfig(
                in_channels=unet_data.get('in_channels', config.model.unet.in_channels),
                out_channels=unet_data.get('out_channels', config.model.unet.out_channels),
                init_features=unet_data.get('init_features', config.model.unet.init_features)
            )
        
        if 'dinov3' in model_data:
            dinov3_data = model_data['dinov3']
            config.model.dinov3 = DINOv3Config(
                pretrained_model_name=dinov3_data.get('pretrained_model_name', config.model.dinov3.pretrained_model_name),
                input_size=dinov3_data.get('input_size', config.dataset.input_size),
                hidden_size=int(dinov3_data.get('hidden_size', config.model.dinov3.hidden_size)),
                tokenW=dinov3_data.get('tokenW', config.model.dinov3.tokenW),
                tokenH=dinov3_data.get('tokenH', config.model.dinov3.tokenH)
            )
    
    # Training config
    if 'training' in yaml_data:
        training_data = yaml_data['training']
        config.training = TrainingConfig(
            num_epochs=training_data.get('num_epochs', config.training.num_epochs),
            batch_size=training_data.get('batch_size', config.training.batch_size),
            learning_rate=training_data.get('learning_rate', config.training.learning_rate)
        )
        
        if 'optimizer' in training_data:
            opt_data = training_data['optimizer']
            config.training.optimizer = OptimizerConfig(
                name=opt_data.get('name', config.training.optimizer.name),
                lr=float(opt_data.get('lr', config.training.learning_rate)),
                weight_decay=float(opt_data.get('weight_decay', config.training.optimizer.weight_decay)),
                betas=opt_data.get('betas', config.training.optimizer.betas)
            )
        
        if 'scheduler' in training_data:
            sched_data = training_data['scheduler']
            config.training.scheduler = SchedulerConfig(
                name=sched_data.get('name', config.training.scheduler.name),
                T_0=sched_data.get('T_0', config.training.scheduler.T_0),
                T_mult=sched_data.get('T_mult', config.training.scheduler.T_mult),
                eta_min=float(sched_data.get('eta_min', config.training.scheduler.eta_min))
            )
        
        if 'loss' in training_data:
            loss_data = training_data['loss']
            config.training.loss = LossConfig(
                name=loss_data.get('name', config.training.loss.name),
                params=loss_data.get('params', config.training.loss.params)
            )
    
    # Augmentation config (simplified)
    if 'augmentation' in yaml_data:
        config.augmentation = AugmentationConfig(
            train=yaml_data['augmentation'].get('train', config.augmentation.train)
        )
    
    # Update experiment name based on model
    config.logging["experiment_name"] = yaml_data.get("project_name", f"{config.model.name}_baseline")
    
    # Update early stopping patience based on epochs
    config.logging["early_stopping"]["patience"] = max(config.training.num_epochs // 20, 50)
    
    return config

def save_config(config: Config, config_path: str):
    """Save configuration to YAML file (simplified version)."""
    config_dict = {
        'dataset': {
            'name': config.dataset.name,
            'input_size': config.dataset.input_size,
            'num_classes': config.dataset.num_classes,
            'normalization': config.dataset.normalization
        },
        'model': {
            'name': config.model.name,
            'unet': {
                'in_channels': config.model.unet.in_channels,
                'out_channels': config.model.unet.out_channels,
                'init_features': config.model.unet.init_features
            },
            'dinov3': {
                'pretrained_model_name': config.model.dinov3.pretrained_model_name,
                'hidden_size': config.model.dinov3.hidden_size,
                'tokenW': config.model.dinov3.tokenW,
                'tokenH': config.model.dinov3.tokenH
            }
        },
        'training': {
            'num_epochs': config.training.num_epochs,
            'batch_size': config.training.batch_size,
            'learning_rate': config.training.learning_rate,
            'optimizer': {
                'name': config.training.optimizer.name,
                'weight_decay': config.training.optimizer.weight_decay,
                'betas': config.training.optimizer.betas
            },
            'scheduler': {
                'name': config.training.scheduler.name,
                'T_0': config.training.scheduler.T_0,
                'T_mult': config.training.scheduler.T_mult,
                'eta_min': config.training.scheduler.eta_min
            },
            'loss': {
                'name': config.training.loss.name,
                'params': config.training.loss.params
            }
        },
        'augmentation': {
            'train': config.augmentation.train
        }
    }
    
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False, indent=2)