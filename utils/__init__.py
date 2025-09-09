from utils.setup import *
from utils.dataloader import CoralSegmentationDataset, train_transform, val_transform
from utils.visualize import training_curve, visualize_predictions_with_gt
from utils.metrics import calculate_miou, calculate_pixel_accuracy