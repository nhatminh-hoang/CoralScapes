import os

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors
import numpy as np

try:
    from utils import ds
except:
    from setup import ds

SIZE = (128, 256)  # Resize images to 128x256

class CoralSegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None, pre_compute=True, split="Unknown"):
        self.dataset = dataset
        self.pre_compute = pre_compute

        if self.pre_compute:
            self.transform = transform
            self.__pre_compute(split)

    def __pre_compute(self, split):
        name = f"dataset/{split}_preprocess_{SIZE[0]}_{SIZE[1]}.pt"

        if os.path.exists(name):
            print(f"Loading pre-processed dataset from {name}...")
            self.dataset = torch.load(name, weights_only=False)
            return
        
        # Pre_compute resize and to tensor for all images and masks
        print(f"Pre-processing dataset and saving to {name}...")
        imgs, msks = [], []
        for i in tqdm(range(len(self.dataset))):
            item = self.dataset[i]

            image = item["image"].convert("RGB")
            mask = tv_tensors.Mask(item["label"])

            sample = {"image": image, "mask": mask}

            if self.transform:
                sample = self.transform(sample)

                image = sample["image"]
                mask = sample["mask"]

                imgs.append(image)
                msks.append(mask)

                assert image.shape[1:] == SIZE, f"Image size mismatch: {image.shape[1:]} vs {SIZE}"
                assert mask.shape[1:] == SIZE, f"Mask size mismatch: {mask.shape[1:]} vs {SIZE}"

                del sample, item, image, mask

        imgs = torch.stack(imgs)                  # (N,C,H,W) fp16
        msks = torch.stack(msks).to(torch.int64)  # (N,H,W)   int64
        
        self.dataset = {"image": imgs, "label": msks}
        # Save into files for future use
        torch.save(self.dataset, f"dataset/{split}_preprocess_{SIZE[0]}_{SIZE[1]}.pt")

    def __len__(self):
        return len(self.dataset["image"])

    def __getitem__(self, idx):
        image = self.dataset["image"][idx]
        mask  = self.dataset["label"][idx]

        return image, mask

# Define transformations
# Use the calculated mean and std from previous steps

train_transform = transforms.Compose([
    transforms.Resize(SIZE),
    transforms.ToTensor(),  # Convert PIL image to tensor
])

augment_transform = transforms.Compose([
    transforms.RandomCrop(SIZE),
    transforms.RandomHorizontalFlip(),  # Data augmentation (applied during __getitem__)
    transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.05, 0.05),
                                  interpolation=InterpolationMode.BILINEAR)], p=0.5),
])

val_transform = transforms.Compose([
    transforms.Resize(SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
    transforms.ToTensor(),  # Convert PIL image to tensor
])

def create_data_loaders(batch_size=64, num_workers=2):
    """Create train and validation data loaders."""
    train_dataset = CoralSegmentationDataset(ds["train"], transform=train_transform, pre_compute=True, split="train")
    val_dataset = CoralSegmentationDataset(ds["validation"], transform=val_transform, pre_compute=True, split="validation")
    test_dataset = CoralSegmentationDataset(ds["test"], transform=val_transform, pre_compute=True, split="test")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_dataloader, val_dataloader, test_dataloader

# Example usage
if __name__ == "__main__":
    import time

    batch_size = 64
    num_workers = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataloader, val_dataloader, test_dataloader = create_data_loaders(batch_size, num_workers)

    print("Train dataset size:", len(train_dataloader.dataset))
    print("Validation dataset size:", len(val_dataloader.dataset))
    print("Test dataset size:", len(test_dataloader.dataset))
    print("Train dataloader batches:", len(train_dataloader))
    print("Validation dataloader batches:", len(val_dataloader))
    print("Test dataloader batches:", len(test_dataloader))
    
    start_time = time.time()
    for epoch in range(1):
        print(f"Epoch {epoch+1}")
        for batch_idx, (images, masks) in enumerate(tqdm(train_dataloader)):
            images = images.to(device)
            masks = masks.to(device)

            # Augmentation
            augmented = augment_transform({"image": images, "mask": masks})
            images = augmented["image"]
            masks = augmented["mask"].float()

        print(f"image shape: {images.shape}")
        print(f"image min: {images.min().item():.4f}, image max: {images.max().item():.4f}, image mean: {images.mean().item():.4f}, image std: {images.std().item():.4f}")
        print(f"mask shape: {masks.squeeze(1).shape}")
        print(f"mask min: {masks.min().item():.4f}, mask max: {masks.max().item():.4f}, mask mean: {masks.mean().item():.4f}, mask std: {masks.std().item():.4f}")

    end_time = time.time()
    print(f"Time taken for one epoch: {end_time - start_time:.2f} seconds")