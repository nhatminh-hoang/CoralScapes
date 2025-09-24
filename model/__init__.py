from model.linear_classifier import LinearClassifier
from model.UNet import UNet
from model.DINOv3 import Dinov3ForSemanticSegmentation

def get_model_and_processor(model_name, config, pretrained_model_name=None, num_labels=40):
    if model_name == "UNet":
        
        in_channels = config.in_channels
        out_channels = config.out_channels
        init_features = config.init_features
            
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels, 
            init_features=init_features
        )
        processor = None
    elif model_name == "DINOv3":
        model = Dinov3ForSemanticSegmentation(config, pretrained_model_name, num_labels)
        
        # Create the image processor separately
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    else:
        raise ValueError(f"Model {model_name} not recognized. Available models: UNet, DINOv3")
    
    return model, processor