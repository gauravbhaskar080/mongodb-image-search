"""Configuration constants for the image embedding search system."""
import torch

# Device configuration - use GPU if available, otherwise CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configuration
MODEL_NAME = "resnet50"  # Default ResNet-50 for feature extraction

# Image preprocessing configuration
IMAGE_SIZE = 224  # Standard ImageNet input size
