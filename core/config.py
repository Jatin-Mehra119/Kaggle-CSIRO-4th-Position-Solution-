"""
Centralised configuration constants for the CSIRO Biomass Prediction pipeline.
"""

import os
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
# Dynamic Paths for ONNX models and scalers
# ---------------------------------------------------------------------------

def _find_file(filename: str, search_roots: list[str]) -> str:
    """Find a file by searching recursively through the provided roots."""
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for curr_dir, _, files in os.walk(root):
            # Limit depth search to prevent excessive scanning
            rel = os.path.relpath(curr_dir, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 4:
                continue
                
            if filename in files:
                return os.path.join(curr_dir, filename)
                
    # Fallback to an expected default path so errors show a sensible missing path
    default_root = search_roots[0] if search_roots else "/data"
    sub_dir = "scalers" if filename.endswith(".pth") else (
        "aux_onnx" if "aux" in filename else "main_onnx"
    )
    return os.path.join(default_root, sub_dir, filename)

_search_paths = []
if "WEIGHTS_ROOT" in os.environ:
    _search_paths.append(os.environ["WEIGHTS_ROOT"])
_search_paths.extend(["/data", "weights", "."])

AUX_ONNX_PATH: str = _find_file("aux_model.onnx", _search_paths)
MAIN_ONNX_PATH: str = _find_file("main_model.onnx", _search_paths)
AUX_SCALER_PATH: str = _find_file("aux_scaler.pth", _search_paths)
TARGET_SCALER_PATH: str = _find_file("target_scaler.pth", _search_paths)

WEIGHTS_ROOT: str = _search_paths[0] if _search_paths else "/data"