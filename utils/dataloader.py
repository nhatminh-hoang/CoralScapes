from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
import numpy as np
import os
from utils import ds, num_classes, get_free_vram


SIZE = (256, 512)  # Resize images to 128x256

def one_hot_encode_batch(masks: torch.Tensor, num_classes):
    """
    Batch one-hot encoding for masks
    Args:
        masks: (B, H, W) with values in {0, 1, ..., num_classes-1}
    Returns:
        one_hot: (B, num_classes, H, W) one-hot encoded
    """
    batch_size, height, width = masks.shape
    masks_flat = masks.view(batch_size, -1).long()
    one_hot_flat = torch.zeros(batch_size, height * width, num_classes, device=masks.device)
    one_hot_flat.scatter_(2, masks_flat.unsqueeze(2), 1)
    one_hot = one_hot_flat.view(batch_size, height, width, num_classes).permute(0, 3, 1, 2)
    return one_hot.float()

def one_hot_encode(mask: torch.Tensor, num_classes):
    """Single mask one-hot encoding (fallback)"""
    mask_flat = mask.view(-1).long()
    one_hot_flat = torch.zeros(mask_flat.size(0), num_classes, device=mask.device)
    one_hot_flat.scatter_(1, mask_flat.unsqueeze(1), 1)
    one_hot = one_hot_flat.view(mask.shape[0], mask.shape[1], num_classes).permute(2, 0, 1)
    return one_hot.float()

class CoralSegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None, device=None, batch_size=2, split_name="unknown"):
        self.dataset = dataset
        self.transform = transform
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessing_batch_size = batch_size
        self.split_name = split_name
        
        # Get target size from transform
        target_h, target_w = SIZE
        if self.transform:
            for t in self.transform.transforms:
                if isinstance(t, transforms.Resize):
                    if hasattr(t, 'size'):
                        target_h, target_w = t.size if isinstance(t.size, (list, tuple)) else (t.size, t.size)
        
        # Define cache directory and file path
        self.cache_file = f"dataset/{split_name}_preprocess_{target_h}_{target_w}.pt"
        print(f"Cache file: {self.cache_file}")

        if self.device == 'cuda':
            while get_free_vram()[0] < 10000 and torch.cuda.is_available():
                print("Waiting for GPU memory...")
                torch.cuda.empty_cache()
                import time
                time.sleep(300)
        
        # Check if preprocessed data exists
        if os.path.exists(self.cache_file):
            print(f"Loading preprocessed data from cache: {self.cache_file}")
            self._load_from_cache()
        else:
            print(f"Preprocessing data and saving to cache: {self.cache_file}")
            self._preprocess_and_cache()

    def _load_from_cache(self):
        """Load preprocessed data from cache file"""
        try:
            cached_data = torch.load(self.cache_file, map_location='cpu')
            self.images = cached_data['images']
            self.masks = cached_data['masks']
            print(f"Successfully loaded {len(self.images)} samples from cache")
        except Exception as e:
            print(f"Error loading cache: {e}")
            print("Falling back to preprocessing...")
            self._preprocess_and_cache()

    def _preprocess_and_cache(self):
        """Preprocess data and save to cache"""
        print(f"Preprocessing on device: {self.device}")
        print(f"Using batch size: {self.preprocessing_batch_size} for preprocessing")

        # Batch preprocessing for faster GPU utilization
        self.images = []
        self.masks = []
        
        # Process dataset in batches
        dataset_list = list(self.dataset)
        total_batches = (len(dataset_list) + self.preprocessing_batch_size - 1) // self.preprocessing_batch_size
        
        for batch_idx in tqdm(range(total_batches), desc=f"Preprocessing {self.split_name} batches"):
            start_idx = batch_idx * self.preprocessing_batch_size
            end_idx = min(start_idx + self.preprocessing_batch_size, len(dataset_list))
            batch_items = dataset_list[start_idx:end_idx]
            
            # Process batch
            batch_images, batch_masks = self._process_batch(batch_items)
            
            # Store processed batch
            for i in range(len(batch_images)):
                self.images.append(batch_images[i].cpu())
                self.masks.append(batch_masks[i].cpu())
            
            # Clean up GPU memory
            del batch_images, batch_masks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del self.dataset  # Free memory
        print(f"Preprocessing complete. Total samples: {len(self.images)}")
        
        # Save to cache
        self._save_to_cache()

    def _save_to_cache(self):
        """Save preprocessed data to cache file"""
        try:

            # Prepare data for saving
            cache_data = {
                'images': self.images,
                'masks': self.masks,
                'split_name': self.split_name,
                'total_samples': len(self.images)
            }
            
            # Save to cache file
            torch.save(cache_data, self.cache_file)
            print(f"Preprocessed data saved to cache: {self.cache_file}")
            
            # Print cache file size
            file_size_mb = os.path.getsize(self.cache_file) / (1024 * 1024)
            print(f"Cache file size: {file_size_mb:.2f} MB")
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            print("Continuing without cache...")

    def _process_batch(self, batch_items):
        """Process a batch of items on GPU"""
        batch_images = []
        batch_masks = []
        
        # Convert batch to tensors
        for item in batch_items:
            image = torch.from_numpy(np.array(item["image"]))
            mask = np.array(item["label"], dtype=np.long)
            mask = torch.from_numpy(mask)
            
            batch_images.append(image)
            batch_masks.append(mask)
        
        if not self.transform:
            return batch_images, batch_masks
        
        # Stack into batch tensors and move to GPU
        batch_images = torch.stack(batch_images).to(self.device)  # (B, H, W, C)
        batch_masks = torch.stack(batch_masks).to(self.device)    # (B, H, W)
        
        # Convert images: (B, H, W, C) -> (B, C, H, W) and normalize to [0,1]
        batch_images = batch_images.permute(0, 3, 1, 2).float() / 255.0
        assert batch_images.shape[1] == 3, "Images must have 3 channels (RGB)"
        
        # One-hot encode masks in batch
        batch_masks = one_hot_encode_batch(batch_masks, num_classes)  # (B, num_classes, H, W)
        
        # Apply transforms to entire batch
        for t in self.transform.transforms:
            if isinstance(t, transforms.Resize):
                batch_images = t(batch_images)
                batch_masks = t(batch_masks)
            elif isinstance(t, transforms.Normalize):
                batch_images = t(batch_images)
        
        return batch_images, batch_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Apply remaining transforms (data augmentation) during training
        if self.transform:
            for t in self.transform.transforms:
                if not isinstance(t, (transforms.Resize, transforms.ToTensor, transforms.Normalize)):
                    # Apply augmentation transforms
                    image = t(image)
                    mask = t(mask)

        return image, mask

# Updated transforms - remove ToTensor since we handle tensor conversion manually
train_transform = transforms.Compose([
    transforms.Resize(SIZE),  # Resize to a common size
    transforms.RandomCrop(SIZE),  # Random crop to target size
    transforms.RandomHorizontalFlip(),  # Data augmentation (applied during __getitem__)
    transforms.RandomRotation(degrees=15),  # Data augmentation (applied during __getitem__)
])

val_transform = transforms.Compose([
    transforms.Resize(SIZE),  # Resize to a common size
])

# Example usage
if __name__ == "__main__":
    # Choose optimal batch size based on GPU memory
    preprocessing_batch_size = 64 if torch.cuda.is_available() else 16
    
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Instantiate datasets with batch preprocessing and caching
    train_dataset = CoralSegmentationDataset(
        ds["train"], 
        transform=train_transform,
        batch_size=preprocessing_batch_size,
        split_name="train"
    )
    
    val_dataset = CoralSegmentationDataset(
        ds["validation"], 
        transform=val_transform,
        batch_size=preprocessing_batch_size,
        split_name="validation"
    )
    
    test_dataset = CoralSegmentationDataset(
        ds["test"], 
        transform=val_transform,
        batch_size=preprocessing_batch_size,
        split_name="test"
    )

    # Define dataloaders
    batch_size = 8  # Training batch size
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=4
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=4
    )

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Train dataloader batches:", len(train_dataloader))
    print("Validation dataloader batches:", len(val_dataloader))
    print("Test dataloader batches:", len(test_dataloader))
    
    # Test a batch
    for images, masks in train_dataloader:
        print("Image batch shape:", images.size())
        print("Mask batch shape:", masks.size())
        print("Image range:", images.min().item(), "to", images.max().item())
        print("Mask shape check:", masks.shape)
        break