# 🐠 CoralScapes Semantic Segmentation

A deep learning project for coral reef semantic segmentation using U-Net architecture on the EPFL-ECEO CoralScapes dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Metrics](#metrics)
- [Contributing](#contributing)

## 🔍 Overview

This project implements semantic segmentation for coral reef images using a U-Net convolutional neural network. The model is trained to classify 39 different coral species and marine life categories, providing detailed pixel-level annotations of underwater coral reef scenes.

### Key Features
- **U-Net Architecture**: Deep convolutional network optimized for semantic segmentation
- **Multi-class Segmentation**: 39 coral species + background (40 classes total)
- **Comprehensive Metrics**: mIoU, pixel accuracy, and loss tracking
- **Automated Logging**: Organized experiment tracking and visualization
- **Data Validation**: Built-in mask validation and preprocessing verification

## 📊 Dataset

**EPFL-ECEO CoralScapes Dataset**
- **Source**: Hugging Face Hub (`EPFL-ECEO/coralscapes`)
- **Training samples**: 1,517 images
- **Validation samples**: 166 images  
- **Test samples**: 392 images
- **Image size**: Resized to 128×256 pixels
- **Classes**: 39 coral species + background

### Class Distribution
The dataset includes diverse coral species such as:
- Seagrass, Brain coral, Fire coral
- Sea fans, Sponges, Algae
- Various hard and soft corals
- Marine life and substrate types

## 🏗️ Model Architecture

**U-Net with Skip Connections**
- **Input**: 3-channel RGB images (128×256)
- **Output**: 40-channel probability maps
- **Features**: 64 initial features (configurable)
- **Architecture**: 
  - Encoder: 4 downsampling blocks with MaxPooling
  - Bottleneck: Feature extraction at lowest resolution
  - Decoder: 4 upsampling blocks with skip connections
  - Final layer: 1×1 convolution for classification

### Model Configuration
```python
model = UNet(
    in_channels=3,
    out_channels=40,  # 39 classes + background
    init_features=64
)
```

## 📈 Training Results

<!-- TRAINING_RESULTS_START -->
*Training results will be automatically updated here after each experiment*

### Latest Training Run
- **Model**: UNet_features64_batch64
- **Epochs**: 45
- **Learning Rate**: 0.0001
- **Optimizer**: AdamW
- **Best Validation mIoU**: *To be updated*
- **Best Validation Accuracy**: *To be updated*

### Training Curves
*Training curves will be automatically embedded here*

### Prediction Examples
*Prediction vs ground truth comparisons will be automatically embedded here*
<!-- TRAINING_RESULTS_END -->

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Setup Environment
```bash
# Clone the repository
git clone <repository-url>
cd CoralScapes

# Install dependencies
pip install -r requirements.txt

# Verify installation
python check_masks.py
```

## 🚀 Usage

### Training
```bash
# Run training with default parameters
python train.py

# Results will be saved to logs/UNet_features64_batch64_epochs45_lr0.0001/
```

### Data Validation
```bash
# Check mask preprocessing
python check_masks.py
```

### Custom Training
Modify hyperparameters in `train.py`:
```python
num_epochs = 45
learning_rate = 1e-4
batch_size = 64
features = 64
```

### Inference
```python
from model import UNet
import torch

# Load trained model
model = UNet(in_channels=3, out_channels=40, init_features=64)
model.load_state_dict(torch.load('logs/.../best_model.pth'))
model.eval()

# Run inference
with torch.no_grad():
    predictions = model(input_tensor)
```

## 📁 Project Structure

```
CoralScapes/
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── train.py                  # Main training script
├── check_masks.py            # Data validation utility
├── model/
│   ├── __init__.py
│   └── UNet.py               # U-Net model implementation
├── utils/
│   ├── __init__.py
│   ├── setup.py              # Dataset loading
│   ├── dataloader.py         # Data preprocessing
│   ├── metrics.py            # Evaluation metrics
│   └── visualize.py          # Visualization functions
├── dataset/                  # Downloaded dataset (auto-created)
├── logs/                     # Training logs and results
│   └── UNet_features64_batch64_epochs45_lr0.0001/
│       ├── best_model.pth    # Best model weights
│       ├── training_curves.png # Loss/mIoU/accuracy plots
│       ├── predictions_vs_gt.png # Prediction examples
│       └── training_metrics.pth # Detailed metrics
└── saved_model/              # Legacy model storage
```

## 📊 Metrics

### Primary Metrics
- **mIoU (mean Intersection over Union)**: Primary segmentation metric
- **Pixel Accuracy**: Overall classification accuracy
- **Cross-Entropy Loss**: Training objective function

### Evaluation Details
- **mIoU calculation**: Averaged across all 40 classes
- **Best model selection**: Based on validation mIoU
- **Real-time tracking**: Metrics computed every epoch
- **Visualization**: Automatic generation of training curves

### Per-Class Analysis
The model tracks IoU for each of the 39 coral species individually, allowing for detailed analysis of performance across different coral types.

## 🔍 Features

### Automated Experiment Tracking
- **Structured Logging**: Each experiment gets a unique directory
- **Metric Persistence**: All metrics saved for later analysis
- **Visualization**: Automatic generation of plots and comparisons
- **Model Checkpointing**: Best models saved based on validation mIoU

### Data Pipeline
- **Automatic Download**: Dataset downloaded via Hugging Face Hub
- **Preprocessing**: Resize, normalization, and augmentation
- **Validation**: Built-in checks for data integrity
- **Efficient Loading**: Multi-worker data loading with caching

### Monitoring and Visualization
- **Training Curves**: Loss, mIoU, and accuracy over time
- **Prediction Comparison**: Side-by-side ground truth vs predictions
- **Class Distribution**: Analysis of dataset class balance
- **Performance Metrics**: Comprehensive evaluation suite

## 🛠️ Advanced Usage

### Custom Model Configuration
```python
# Smaller model for faster training
model = UNet(in_channels=3, out_channels=40, init_features=32)

# Larger model for better performance
model = UNet(in_channels=3, out_channels=40, init_features=128)
```

### Data Augmentation
Modify transforms in `utils/dataloader.py`:
```python
train_transform = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),     # Add vertical flip
    transforms.ColorJitter(0.1, 0.1),   # Add color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.306, 0.508, 0.491], 
                        std=[0.200, 0.194, 0.198])
])
```

### Hyperparameter Tuning
Key parameters to experiment with:
- `learning_rate`: 1e-3 to 1e-5
- `batch_size`: 16, 32, 64, 128
- `init_features`: 32, 64, 128
- `num_epochs`: 25, 45, 100

## 📈 Performance Optimization

### Training Tips
1. **Batch Size**: Use largest batch size that fits in GPU memory
2. **Learning Rate**: Start with 1e-4, adjust based on loss curves
3. **Early Stopping**: Monitor validation mIoU for convergence
4. **Data Augmentation**: Helps with limited training data

### Hardware Requirements
- **Minimum**: 4GB GPU memory, 8GB RAM
- **Recommended**: 8GB+ GPU memory, 16GB+ RAM
- **Optimal**: RTX 3080/4080 or better

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings for new functions
- Include tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **EPFL-ECEO** for providing the CoralScapes dataset
- **Hugging Face** for dataset hosting and tools
- **PyTorch** team for the deep learning framework
- **U-Net paper**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation"

## 📞 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

*Last updated: September 9, 2025*
