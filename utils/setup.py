import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel

from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

DATASET_ID = "EPFL-ECEO/coralscapes"  # schema: image (PIL), label (PIL)
dataset_dir = "./dataset"
ds = load_dataset(DATASET_ID, cache_dir=dataset_dir)  # splits: train/validation/test
# -> {'train': 1517, 'validation': 166, 'test': 392}

# Load id2label + label2color from the repo
id2label_fp   = hf_hub_download(repo_id=DATASET_ID, filename="id2label.json", repo_type="dataset")
label2color_fp= hf_hub_download(repo_id=DATASET_ID, filename="label2color.json", repo_type="dataset")
id2label   = {int(k):v for k,v in json.load(open(id2label_fp)).items()}  # {1:'seagrass',...,39:'dead clam'}
id2label[0] = 'background'  # add background
label2color= json.load(open(label2color_fp))                             # {'seagrass':[R,G,B],...}
label2color['background'] = [0,0,0]  # add background

num_classes = len(id2label) + 1  # 40 (including background)

pretrained_model_names = [
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "facebook/dinov3-convnext-large-pretrain-lvd1689m",
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ]

for pretrained_model_name in pretrained_model_names:
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token=HF_TOKEN)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto",
        token=HF_TOKEN
    )