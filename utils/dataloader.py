import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from utils.setup import ds, num_classes

SIZE = (128, 256)  # Resize images to 128x256

def one_hot_encode(mask, num_classes):
    # mask: (H, W) with values in {0, 1, ..., num_classes-1}
    # returns: (num_classes, H, W) one-hot encoded
    one_hot = np.zeros((num_classes, mask.shape[0], mask.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        one_hot[c] = (mask == c).astype(np.float32)
    return one_hot

class CoralSegmentationDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        mask = np.array(item["label"], dtype=np.int64)
        mas_one_hot = one_hot_encode(mask, num_classes=num_classes)  # 39 classes + background
        mask = torch.from_numpy(mas_one_hot)

        if self.transform:
            image = self.transform(image)
            # Apply the same spatial transformations to the mask
            # Note: Transformations like Normalize should only be applied to the image
            # and not the mask. We handle this by applying transformations sequentially.
            for t in self.transform.transforms:
                if isinstance(t, (transforms.Resize, transforms.RandomCrop, transforms.RandomHorizontalFlip, transforms.RandomVerticalFlip, transforms.Pad)):
                    mask = t(mask)
                    # pass
                elif isinstance(t, transforms.ToTensor):
                     # ToTensor scales image pixels to [0, 1], but we don't want this for masks
                     # We already converted mask to tensor, so skip this for mask.
                     pass
                elif isinstance(t, transforms.Normalize):
                    # Normalize only applies to the image
                    pass
                else:
                    # Apply other transforms if any (handle with care for masks)
                    pass


        return image, mask

# Define transformations
# Use the calculated mean and std from previous steps
# mean: [0.30615412 0.50791315 0.49072459]
# std: [0.19963073 0.19414033 0.19804019]

train_transform = transforms.Compose([
    transforms.Resize(SIZE), # Resize to a common size
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.306, 0.508, 0.491], std=[0.200, 0.194, 0.198]) # Normalize image
])

val_transform = transforms.Compose([
    transforms.Resize(SIZE), # Resize to a common size
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.306, 0.508, 0.491], std=[0.200, 0.194, 0.198]) # Normalize image
])

# Example usage
if __name__ == "__main__":
    # Instantiate datasets
    train_dataset = CoralSegmentationDataset(ds["train"], transform=train_transform)
    val_dataset = CoralSegmentationDataset(ds["validation"], transform=val_transform)
    test_dataset = CoralSegmentationDataset(ds["test"], transform=val_transform)

    # Define dataloaders
    batch_size = 2 # Example batch size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Train dataloader batches:", len(train_dataloader))
    print("Validation dataloader batches:", len(val_dataloader))
    print("Test dataloader batches:", len(test_dataloader))
    
    for images, masks in train_dataloader:
        print("Image batch shape:", images.size())
        print("Mask batch shape:", masks.size())
        break