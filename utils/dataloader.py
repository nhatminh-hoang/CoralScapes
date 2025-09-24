import os

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.transforms import InterpolationMode
from torchvision import tv_tensors

try:
    from utils import ds
except:
    from setup import ds

# Default size - will be overridden by config
SIZE = (128, 256)  # Resize images to 128x256
processor = None  # to be set from setup.py if available

class CoralSegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None, pre_compute=True, split="Unknown", config=None):
        self.dataset = dataset
        self.pre_compute = pre_compute
        self.config = config
        
        # Update SIZE from config if provided
        global SIZE
        if config and hasattr(config, 'dataset') and hasattr(config.dataset, 'input_size'):
            SIZE = tuple(config.dataset.input_size)

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

def make_transform(resize_size, config=None):
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    to_tensor = transforms.ToImage()
    resize = transforms.Resize((resize_size[0], resize_size[1]), antialias=True)
    to_float = transforms.ToDtype(torch.float32, scale=True)
    
    # Use normalization from config if provided
    if config and hasattr(config, 'dataset') and hasattr(config.dataset, 'normalization'):
        normalize = transforms.Normalize(
            mean=config.dataset.normalization['mean'],
            std=config.dataset.normalization['std'],
        )
    else:
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    return transforms.Compose([to_tensor, resize, to_float, normalize])

def create_augmentation_transforms(config=None):
    """Create augmentation transforms based on config."""
    global SIZE
    
    if config and hasattr(config, 'augmentation') and config.augmentation.train.get('enable', True):
        aug_config = config.augmentation.train
        
        augment_img_transforms = []
        augment_mask_transforms = []
        
        # Random crop
        if aug_config.get('random_crop', {}).get('enable', True):
            crop_size = aug_config.get('random_crop', {}).get('size', [SIZE[0], SIZE[0]])
            augment_img_transforms.append(transforms.RandomCrop(crop_size))
            augment_mask_transforms.append(transforms.RandomCrop(crop_size))
        
        # Random horizontal flip
        if aug_config.get('random_horizontal_flip', {}).get('enable', True):
            prob = aug_config.get('random_horizontal_flip', {}).get('probability', 0.5)
            augment_img_transforms.append(transforms.RandomHorizontalFlip(p=prob))
            augment_mask_transforms.append(transforms.RandomHorizontalFlip(p=prob))
        
        # Random affine
        if aug_config.get('random_affine', {}).get('enable', True):
            affine_config = aug_config.get('random_affine', {})
            prob = affine_config.get('probability', 0.5)
            degrees = affine_config.get('degrees', 20)
            translate = affine_config.get('translate', [0.05, 0.05])
            
            augment_img_transforms.append(
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=degrees, 
                        translate=translate,
                        interpolation=InterpolationMode.BILINEAR
                    )
                ], p=prob)
            )
            augment_mask_transforms.append(
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=degrees, 
                        translate=translate,
                        interpolation=InterpolationMode.NEAREST
                    )
                ], p=prob)
            )
        
        augment_img = transforms.Compose(augment_img_transforms)
        augment_mask = transforms.Compose(augment_mask_transforms)
    else:
        # Default augmentation
        augment_img = transforms.Compose([
            transforms.RandomCrop((SIZE[0], SIZE[0])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.05, 0.05),
                                          interpolation=InterpolationMode.BILINEAR)], p=0.5),
        ])

        augment_mask = transforms.Compose([
            transforms.RandomCrop((SIZE[0], SIZE[0])),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.05, 0.05),
                                          interpolation=InterpolationMode.NEAREST)], p=0.5),
        ])
    
    return augment_img, augment_mask

transform = make_transform(resize_size=SIZE)
augment_img, augment_mask = create_augmentation_transforms()

def create_data_loaders(batch_size=64, num_workers=2, config=None):
    """Create train and validation data loaders."""
    # Hardcoded defaults for dataloader settings
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_PIN_MEMORY = True
    DEFAULT_PERSISTENT_WORKERS = True
    
    # Use config values if provided, otherwise use defaults
    if config:
        batch_size = config.training.batch_size if hasattr(config, 'training') else batch_size
        num_workers = DEFAULT_NUM_WORKERS
        pin_memory = DEFAULT_PIN_MEMORY
        persistent_workers = DEFAULT_PERSISTENT_WORKERS
        
        # Update global augmentation transforms
        global augment_img, augment_mask
        augment_img, augment_mask = create_augmentation_transforms(config)
        
        # Update transform with config
        global transform
        transform = make_transform(resize_size=SIZE, config=config)
    else:
        pin_memory = True
        persistent_workers = True
    
    train_dataset = CoralSegmentationDataset(ds["train"], transform=transform, pre_compute=True, split="train", config=config)
    val_dataset = CoralSegmentationDataset(ds["validation"], transform=transform, pre_compute=True, split="validation", config=config)
    test_dataset = CoralSegmentationDataset(ds["test"], transform=transform, pre_compute=True, split="test", config=config)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        pin_memory=pin_memory
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
            images = augment_img(images)
            masks = augment_mask(masks)

            if processor is not None:
                inputs = processor(images=images, return_tensors="pt").to(device)

        print(f"image shape: {images.shape}")
        print(f"image min: {images.min().item():.4f}, image max: {images.max().item():.4f}, image mean: {images.mean().item():.4f}, image std: {images.std().item():.4f}")
        print(f"mask shape: {masks.squeeze(1).shape}")
        print(f"mask min: {masks.min().item():.4f}, mask max: {masks.max().item():.4f}, mask mean: {masks.mean().item():.4f}, mask std: {masks.std().item():.4f}")

    end_time = time.time()
    print(f"Time taken for one epoch: {end_time - start_time:.2f} seconds")