# Configuration System for CoralScapes

This project uses a YAML-based configuration system to manage all training parameters, model settings, and data processing options.

## Configuration Files

### 📁 Available Configs

- **`config/base_config.yaml`** - Default UNet configuration
- **`config/dinov3_config.yaml`** - Optimized DINOv3 configuration

### 🔧 Configuration Structure

The configuration is organized into the following sections:

#### 📊 Dataset Configuration
```yaml
dataset:
  name: "CoralScapes"
  input_size: [128, 256]  # [height, width]
  num_classes: 40
  resize_method: "bilinear"
  normalization:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

#### 🧠 Model Configuration
```yaml
model:
  name: "UNet"  # Options: UNet, DINOv3
  unet:
    in_channels: 3
    out_channels: 40
    init_features: 64
  dinov3:
    pretrained_model_name: "facebook/dinov3-convnext-base-pretrain-lvd1689m"
    hidden_size: 384
```

#### 📈 Training Configuration
```yaml
training:
  num_epochs: 2000
  batch_size: 32
  learning_rate: 1e-3
  optimizer:
    name: "AdamW"
    weight_decay: 1e-4
  scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 100
```

#### 🎲 Data Augmentation
```yaml
augmentation:
  train:
    enable: true
    random_horizontal_flip:
      enable: true
      probability: 0.5
    random_rotation:
      enable: true
      degrees: 45
```

## 🚀 Usage

### Basic Training
```bash
# Train with default UNet configuration
python train.py --config config/base_config.yaml

# Train with DINOv3 configuration
python train.py --config config/dinov3_config.yaml
```

### Creating Custom Configurations

1. **Copy an existing config:**
   ```bash
   cp config/base_config.yaml config/my_config.yaml
   ```

2. **Modify parameters as needed:**
   ```yaml
   training:
     batch_size: 16
     learning_rate: 5e-4
   model:
     name: "DINOv3"
   ```

3. **Train with your config:**
   ```bash
   python train.py --config config/my_config.yaml
   ```

### 🧪 Testing Configuration

Run the test script to validate your configuration:
```bash
python test_config.py
```

## 📝 Configuration Parameters

### Model Options
- **UNet**: Traditional U-Net architecture
  - `init_features`: Number of initial features (32, 64, 128)
  - `bilinear`: Use bilinear upsampling vs transposed convolution

- **DINOv3**: Vision Transformer with DINOv3 backbone
  - `pretrained_model_name`: HuggingFace model identifier
  - `hidden_size`: Hidden dimension size
  - `tokenW`, `tokenH`: Token dimensions for patch embedding

### Optimizer Options
- **AdamW**: Adam with weight decay (recommended)
- **Adam**: Standard Adam optimizer
- **SGD**: Stochastic Gradient Descent with momentum

### Scheduler Options
- **CosineAnnealingWarmRestarts**: Cosine annealing with warm restarts
- **StepLR**: Step-based learning rate decay
- **ExponentialLR**: Exponential learning rate decay

### Loss Functions
- **cross_entropy**: Standard cross-entropy loss
- **dice**: Dice coefficient loss
- **cross_entropy+dice**: Combined loss (recommended)
- **focal**: Focal loss for imbalanced classes

## 🎯 Best Practices

### UNet Configuration
```yaml
model:
  name: "UNet"
  unet:
    init_features: 64  # Good balance of performance and memory
training:
  batch_size: 32
  learning_rate: 1e-3
  optimizer:
    name: "AdamW"
```

### DINOv3 Configuration
```yaml
model:
  name: "DINOv3"
training:
  batch_size: 8      # Smaller batch size due to memory requirements
  learning_rate: 5e-5  # Lower LR for pretrained models
dataset:
  input_size: [224, 224]  # Square images work better with ViT
```

### High-Resolution Training
```yaml
dataset:
  input_size: [256, 512]  # Higher resolution
training:
  batch_size: 16  # Reduce batch size for memory
hardware:
  precision: "bf16-mixed"  # Use mixed precision
```

## 🔍 Configuration Validation

The system automatically validates configurations and provides helpful error messages:

- ✅ **Valid parameters**: All parameters are checked against expected types
- ✅ **Model compatibility**: Ensures model-specific parameters are valid
- ✅ **Hardware constraints**: Warns about memory requirements
- ✅ **Path validation**: Checks that required directories exist

## 📊 Monitoring and Logging

Configure comprehensive logging and monitoring:

```yaml
logging:
  tensorboard:
    enable: true
    log_images: true
  checkpoint:
    monitor: "val_miou"
    mode: "max"
    save_top_k: 3
  early_stopping:
    enable: true
    patience: 100
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or `input_size`
2. **Slow Training**: Increase `num_workers` or enable `pin_memory`
3. **Poor Convergence**: Adjust `learning_rate` or try different `scheduler`
4. **Config Errors**: Run `python test_config.py` to validate

### Memory Optimization
```yaml
training:
  batch_size: 8  # Reduce batch size
hardware:
  precision: "bf16-mixed"  # Use mixed precision
dataloader:
  num_workers: 2  # Reduce if CPU limited
  pin_memory: false  # Disable if memory constrained
```

## 📚 Examples

See the `config/` directory for complete working examples:
- `base_config.yaml` - Standard UNet setup
- `dinov3_config.yaml` - DINOv3 with optimized parameters