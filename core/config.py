"""
Centralised configuration constants for the CSIRO Biomass Prediction pipeline.
"""

import torch

# Image size must match training
IMG_SIZE: int = 800

# timm backbone used by both models
MODEL_NAME: str = "vit_huge_plus_patch16_dinov3.lvd1689m"

# Prediction target column names (order matters)
TARGETS: list[str] = [
    "Dry_Green_g",
    "Dry_Dead_g",
    "Dry_Clover_g",
    "GDM_g",
    "Dry_Total_g",
]

# Inference device – prefer CUDA when available
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use float16 on GPU for faster inference; float32 on CPU
DTYPE: torch.dtype = torch.float16 if DEVICE.type == "cuda" else torch.float32

# ---------------------------------------------------------------------------
# Local paths for ONNX models and scalers
# ---------------------------------------------------------------------------

AUX_ONNX_PATH: str = "weights/aux_onnx/aux_model.onnx"
MAIN_ONNX_PATH: str = "weights/main_onnx/main_model.onnx"
AUX_SCALER_PATH: str = "weights/scalers/aux_scaler.pth"
TARGET_SCALER_PATH: str = "weights/scalers/target_scaler.pth"
