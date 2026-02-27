"""
Image preprocessing pipeline (matches validation / inference transforms).
"""

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from core.config import IMG_SIZE

_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode uploaded bytes -> preprocessed tensor (1, 3, H, W)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = _transform(image=image)["image"].unsqueeze(0)  # (1,3,H,W)
    return tensor
