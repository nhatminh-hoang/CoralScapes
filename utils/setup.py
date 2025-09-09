import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

import torchvision.models as models

DATASET_ID = "EPFL-ECEO/coralscapes"  # schema: image (PIL), label (PIL)
dataset_dir = "./dataset"
ds = load_dataset(DATASET_ID, cache_dir=dataset_dir)  # splits: train/validation/test
print({k: len(v) for k,v in ds.items()})
# -> {'train': 1517, 'validation': 166, 'test': 392}

# Load id2label + label2color from the repo
id2label_fp   = hf_hub_download(repo_id=DATASET_ID, filename="id2label.json", repo_type="dataset")
label2color_fp= hf_hub_download(repo_id=DATASET_ID, filename="label2color.json", repo_type="dataset")
id2label   = {int(k):v for k,v in json.load(open(id2label_fp)).items()}  # {1:'seagrass',...,39:'dead clam'}
label2color= json.load(open(label2color_fp))                             # {'seagrass':[R,G,B],...}

print("id2label:", id2label)

num_classes = len(id2label) + 1  # 40 (including background)
print(num_classes, "classes")

resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

mobilenet_v2 = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet_v3_small = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
mobilenet_v3_large = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)

efficientnet_b0 = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
efficientnet_b1 = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
efficientnet_b2 = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
efficientnet_b3 = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)

resnext50_32x4d = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)