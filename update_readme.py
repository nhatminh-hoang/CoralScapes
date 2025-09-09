#!/usr/bin/env python3
"""
Script to automatically update README.md with the best training results.
This script finds the best performing model and updates the README with
training curves, metrics, and prediction visualizations.
"""

import os
import re
import glob
import torch
import shutil
from pathlib import Path
from datetime import datetime

def find_best_experiment(logs_dir="logs"):
    """
    Find the experiment with the best validation mIoU.
    
    Args:
        logs_dir (str): Directory containing experiment logs
        
    Returns:
        tuple: (best_log_dir, best_metrics) or (None, None) if no experiments found
    """
    if not os.path.exists(logs_dir):
        print(f"Logs directory '{logs_dir}' not found.")
        return None, None
    
    best_miou = 0.0
    best_log_dir = None
    best_metrics = None
    
    # Search for all experiment directories
    experiment_dirs = glob.glob(os.path.join(logs_dir, "*"))
    
    if not experiment_dirs:
        print("No experiment directories found in logs/")
        return None, None
    
    print(f"Found {len(experiment_dirs)} experiment(s):")
    
    for exp_dir in experiment_dirs:
        if not os.path.isdir(exp_dir):
            continue
            
        metrics_file = os.path.join(exp_dir, "training_metrics.pth")
        if not os.path.exists(metrics_file):
            continue
            
        try:
            metrics = torch.load(metrics_file, map_location='cpu')
            miou = metrics.get('best_miou', 0.0)
            
            exp_name = os.path.basename(exp_dir)
            print(f"  {exp_name}: mIoU = {miou:.4f}")
            
            if miou > best_miou:
                best_miou = miou
                best_log_dir = exp_dir
                best_metrics = metrics
                
        except Exception as e:
            print(f"  Error loading metrics from {exp_dir}: {e}")
            continue
    
    if best_log_dir:
        print(f"\nBest experiment: {os.path.basename(best_log_dir)} (mIoU: {best_miou:.4f})")
        return best_log_dir, best_metrics
    else:
        print("No valid experiments found with metrics.")
        return None, None

def copy_images_to_readme_assets(log_dir, assets_dir="readme_assets"):
    """
    Copy training curves and prediction images to a readme assets directory.
    
    Args:
        log_dir (str): Path to the best experiment log directory
        assets_dir (str): Directory to store README assets
        
    Returns:
        dict: Paths to copied images
    """
    os.makedirs(assets_dir, exist_ok=True)
    
    copied_files = {}
    
    # Copy training curves
    curves_src = os.path.join(log_dir, "training_curves.png")
    if os.path.exists(curves_src):
        curves_dst = os.path.join(assets_dir, "training_curves.png")
        shutil.copy2(curves_src, curves_dst)
        copied_files['training_curves'] = curves_dst
        print(f"Copied training curves to: {curves_dst}")
    
    # Copy prediction comparisons
    pred_src = os.path.join(log_dir, "predictions_vs_gt.png")
    if os.path.exists(pred_src):
        pred_dst = os.path.join(assets_dir, "predictions_vs_gt.png")
        shutil.copy2(pred_src, pred_dst)
        copied_files['predictions'] = pred_dst
        print(f"Copied predictions to: {pred_dst}")
    
    return copied_files

def format_metrics_table(metrics):
    """
    Format metrics into a markdown table.
    
    Args:
        metrics (dict): Training metrics dictionary
        
    Returns:
        str: Formatted markdown table
    """
    best_miou = metrics.get('best_miou', 0.0)
    best_loss = metrics.get('best_val_loss', 0.0)
    
    # Get final epoch metrics
    final_train_acc = metrics.get('train_accuracies', [0])[-1] if metrics.get('train_accuracies') else 0
    final_val_acc = metrics.get('val_accuracies', [0])[-1] if metrics.get('val_accuracies') else 0
    
    table = f"""
| Metric | Value |
|--------|-------|
| **Best Validation mIoU** | {best_miou:.4f} |
| **Best Validation Loss** | {best_loss:.4f} |
| **Final Training Accuracy** | {final_train_acc:.4f} |
| **Final Validation Accuracy** | {final_val_acc:.4f} |
| **Total Epochs** | {len(metrics.get('train_losses', []))} |
"""
    return table

def extract_experiment_info(log_dir_name):
    """
    Extract experiment configuration from directory name.
    
    Args:
        log_dir_name (str): Name of the log directory
        
    Returns:
        dict: Extracted configuration parameters
    """
    # Pattern: UNet_features64_batch64_epochs45_lr0.0001
    pattern = r'UNet_features(\d+)_batch(\d+)_epochs(\d+)_lr([\d.]+)'
    match = re.match(pattern, log_dir_name)
    
    if match:
        return {
            'model': 'UNet',
            'features': int(match.group(1)),
            'batch_size': int(match.group(2)),
            'epochs': int(match.group(3)),
            'learning_rate': float(match.group(4))
        }
    else:
        return {
            'model': 'UNet',
            'features': 'Unknown',
            'batch_size': 'Unknown',
            'epochs': 'Unknown',
            'learning_rate': 'Unknown'
        }

def update_readme(log_dir, metrics, image_paths, readme_path="README.md"):
    """
    Update README.md with the best experiment results.
    
    Args:
        log_dir (str): Path to the best experiment log directory
        metrics (dict): Training metrics
        image_paths (dict): Paths to copied images
        readme_path (str): Path to README.md file
    """
    if not os.path.exists(readme_path):
        print(f"README.md not found at {readme_path}")
        return
    
    # Read current README
    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract experiment info
    exp_name = os.path.basename(log_dir)
    exp_info = extract_experiment_info(exp_name)
    
    # Create the updated training results section
    results_section = f"""<!-- TRAINING_RESULTS_START -->
## 🏆 Latest Best Performance

**Experiment**: `{exp_name}`

### Configuration
- **Model**: {exp_info['model']} with {exp_info['features']} initial features
- **Batch Size**: {exp_info['batch_size']}
- **Epochs**: {exp_info['epochs']}
- **Learning Rate**: {exp_info['learning_rate']}
- **Optimizer**: AdamW
- **Training Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

### Performance Metrics
{format_metrics_table(metrics)}

### Training Progress
"""
    
    # Add training curves if available
    if 'training_curves' in image_paths:
        results_section += f"""
![Training Curves]({image_paths['training_curves']})

*Training curves showing loss, mIoU, and accuracy progression over epochs*
"""
    
    # Add predictions if available
    if 'predictions' in image_paths:
        results_section += f"""
### Model Predictions vs Ground Truth

![Predictions vs Ground Truth]({image_paths['predictions']})

*Examples of model predictions compared to ground truth annotations*
"""
    
    results_section += """
### Key Insights
- Model convergence and stability analysis
- Performance across different coral species
- Areas for potential improvement

*Results automatically updated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*"
    
    results_section += "\n<!-- TRAINING_RESULTS_END -->"
    
    # Replace the results section in README
    pattern = r'<!-- TRAINING_RESULTS_START -->.*?<!-- TRAINING_RESULTS_END -->'
    
    if re.search(pattern, content, re.DOTALL):
        # Replace existing section
        new_content = re.sub(pattern, results_section, content, flags=re.DOTALL)
    else:
        # Insert before the Installation section
        installation_pattern = r'## 🔧 Installation'
        if re.search(installation_pattern, content):
            new_content = re.sub(
                installation_pattern, 
                results_section + '\n\n## 🔧 Installation', 
                content
            )
        else:
            # Append at the end if no installation section found
            new_content = content + '\n\n' + results_section
    
    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✅ README.md updated successfully!")
    print(f"📊 Best mIoU: {metrics.get('best_miou', 0.0):.4f}")
    print(f"📁 Experiment: {exp_name}")

def main():
    """Main function to update README with best results."""
    print("=" * 60)
    print("🚀 AUTOMATICALLY UPDATING README WITH BEST RESULTS")
    print("=" * 60)
    
    # Find best experiment
    best_log_dir, best_metrics = find_best_experiment()
    
    if not best_log_dir:
        print("❌ No experiments found. Train a model first!")
        return
    
    # Copy images to assets directory
    print("\n📷 Copying visualization assets...")
    image_paths = copy_images_to_readme_assets(best_log_dir)
    
    # Update README
    print("\n📝 Updating README.md...")
    update_readme(best_log_dir, best_metrics, image_paths)
    
    print("\n" + "=" * 60)
    print("✅ README UPDATE COMPLETE!")
    print("The README.md now includes:")
    print("  • Best model performance metrics")
    print("  • Training configuration details")
    print("  • Training curves visualization")
    print("  • Prediction examples")
    print("=" * 60)

if __name__ == "__main__":
    main()
