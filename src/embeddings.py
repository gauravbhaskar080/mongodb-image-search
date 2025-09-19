from __future__ import annotations
import os
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from .config import DEVICE, MODEL_NAME, IMAGE_SIZE

# Disable WebP support to avoid libwebp.dll issues
try:
    from PIL import WebPImagePlugin
    Image.EXTENSION.pop('.webp', None)
    Image.MIME.pop('image/webp', None)
except ImportError:
    pass

# ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _build_preprocess(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

def _get_resnet_backbone(name: str = MODEL_NAME) -> Tuple[nn.Module, int]:
    """
    Returns a ResNet backbone that outputs pooled features (N, D).
    """
    name = name.lower()
    weight_map = {
        "resnet18":  models.ResNet18_Weights.IMAGENET1K_V1,
        "resnet34":  models.ResNet34_Weights.IMAGENET1K_V1,
        "resnet50":  models.ResNet50_Weights.IMAGENET1K_V2,
        "resnet101": models.ResNet101_Weights.IMAGENET1K_V2,
        "resnet152": models.ResNet152_Weights.IMAGENET1K_V2,
    }
    if name not in weight_map:
        raise ValueError(f"Unsupported model_name={name}. Choices: {list(weight_map.keys())}")

    if name == "resnet18":
        model = models.resnet18(weights=weight_map[name])
        feat_dim = 512
    elif name == "resnet34":
        model = models.resnet34(weights=weight_map[name])
        feat_dim = 512
    elif name == "resnet50":
        model = models.resnet50(weights=weight_map[name])
        feat_dim = 2048
    elif name == "resnet101":
        model = models.resnet101(weights=weight_map[name])
        feat_dim = 2048
    else:
        model = models.resnet152(weights=weight_map[name])
        feat_dim = 2048

    # Remove final FC, keep up to avgpool to get (N, 2048, 1, 1) then flatten
    backbone = nn.Sequential(*list(model.children())[:-1]).to(DEVICE).eval()
    return backbone, feat_dim
class ImageEmbedder:
    """
    Wraps a Torch backbone + preprocess to produce L2-normalized embeddings.
    """
    def __init__(self, model_name: str = MODEL_NAME, image_size: int = IMAGE_SIZE):
        self.device = DEVICE
        self.model, self.feat_dim = _get_resnet_backbone(model_name)
        self.preprocess = _build_preprocess(image_size)

        # no gradients
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.inference_mode()
    def embed_pil(self, img: Image.Image) -> np.ndarray:
        t = self.preprocess(img).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feats = self.model(t).flatten(1)  # (1, D)
        # L2 normalize
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.squeeze(0).cpu().numpy().astype(np.float32)  # (D,)

    @torch.inference_mode()
    def embed_paths(self, paths: List[str], batch_size: int = 16) -> np.ndarray:
        """
        Efficiently embed a list of image file paths in batches.
        Returns array of shape (N, D), L2-normalized.
        """
        all_embs = []
        batch = []
        batch_paths = []

        def process_batch(batch_imgs: List[torch.Tensor]):
            if not batch_imgs:
                return None
            x = torch.stack(batch_imgs, dim=0).to(self.device)
            feats = self.model(x).flatten(1)
            feats = torch.nn.functional.normalize(feats, dim=1)
            return feats.cpu().numpy().astype(np.float32)

        for p in paths:
            try:
                # Check file extension to avoid unsupported formats
                ext = os.path.splitext(p)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                
                img = Image.open(p).convert("RGB")
                batch.append(self.preprocess(img))
                batch_paths.append(p)
                if len(batch) >= batch_size:
                    embs = process_batch(batch)
                    all_embs.append(embs)
                    batch, batch_paths = [], []
            except Exception as e:
                # Skip unreadable images gracefully
                print(f"Warning: Skipping {p} - {str(e)}")
                continue

        # leftover
        if batch:
            embs = process_batch(batch)
            all_embs.append(embs)

        if not all_embs:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        return np.concatenate(all_embs, axis=0)
