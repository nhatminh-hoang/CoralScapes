import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel

from dotenv import load_dotenv
import os

try:
    from model.linear_classifier import LinearClassifier
except:
    from .linear_classifier import LinearClassifier

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

class Dinov3ForSemanticSegmentation(nn.Module):
    def __init__(self, config, pretrained_model_name, num_labels):
        super(Dinov3ForSemanticSegmentation, self).__init__()
        self.config = config
        self.num_labels = int(num_labels)
        self.input_size = config.input_size
        self.pretrained_model_name = config.pretrained_model_name

        self.dinov3 = AutoModel.from_pretrained(
            pretrained_model_name,
            token=HF_TOKEN
        ).train()

        self.classifier = LinearClassifier(in_channels=config.hidden_size, tokenW=config.tokenW, tokenH=config.tokenH, num_labels=num_labels)

    def forward(self, inputs):
        # Forward pass through the DINOv3 model
        outputs = self.dinov3(**inputs)
        
        # Extract the last hidden state
        hidden_states = outputs.last_hidden_state[:, 1:, :]  # (batch_size, seq_len-1, hidden_size)
        
        # Classifier to get logits
        logits = self.classifier(hidden_states)  # (batch_size, num_labels, height, width)

        if self.training:
            size = self.input_size[0] - 6
            logits = torch.nn.functional.interpolate(logits, size=(size, size), mode="bilinear", align_corners=False)

        else:
            logits = torch.nn.functional.interpolate(logits, size=self.input_size, mode="bilinear", align_corners=False)
        
        return logits