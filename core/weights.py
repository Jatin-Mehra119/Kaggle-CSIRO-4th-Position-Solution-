"""
Model weight loading utilities using ONNX and local PyTorch scalers.
"""

import os
import torch
import onnxruntime as ort

from core.config import (
    AUX_ONNX_PATH,
    MAIN_ONNX_PATH,
    AUX_SCALER_PATH,
    TARGET_SCALER_PATH,
)

# ---------------------------------------------------------------------------
# Local weight loading
# ---------------------------------------------------------------------------

def _validate_required_files() -> None:
    """Fail fast when required local/container weight files are missing."""
    missing = [
        path
        for path in (AUX_ONNX_PATH, MAIN_ONNX_PATH)
        if not os.path.isfile(path)
    ]
    if missing:
        missing_str = ", ".join(missing)
        
        err_msg = (
            f"Missing required model files: {missing_str}. "
            "Ensure your files (aux_model.onnx, main_model.onnx) are inside /data "
            "or set WEIGHTS_ROOT environment variable."
        )
        
        # Try to debug log what was actually found in /data
        data_contents = []
        if os.path.isdir("/data"):
            data_contents = os.listdir("/data")
            err_msg += f" Found in /data: {data_contents}"
            
        raise FileNotFoundError(err_msg)

def _load_aux_model() -> tuple:
    """Return (ONNX InferenceSession, tab_scaler | None)."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(AUX_ONNX_PATH, providers=providers)
    
    tab_scaler = None
    if os.path.isfile(AUX_SCALER_PATH):
        tab_scaler = torch.load(AUX_SCALER_PATH, map_location="cpu", weights_only=False)
        
    return session, tab_scaler


def _load_main_model() -> tuple:
    """Return (ONNX InferenceSession, tabular_scaler | None, target_scaler | None)."""
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    session = ort.InferenceSession(MAIN_ONNX_PATH, providers=providers)
    
    # Note: In the original implementation, the main model might have had a separate tabular_scaler
    # Here we assume no separate tabular_scaler is passed or it uses aux_scaler if needed. We return None for tabular_scaler.
    tabular_scaler = None
    target_scaler = None
    if os.path.isfile(TARGET_SCALER_PATH):
        target_scaler = torch.load(TARGET_SCALER_PATH, map_location="cpu", weights_only=False)
        
    return session, tabular_scaler, target_scaler


# ---------------------------------------------------------------------------
# Public API – lazy-loaded model registry
# ---------------------------------------------------------------------------

_models: dict = {}


def get_models() -> dict:
    """
    Load and return models.
    Returns a dict with keys:
      - ``aux_folds``  : list[(InferenceSession, tab_scaler)]
      - ``main_folds`` : list[(InferenceSession, tabular_scaler, target_scaler)]
    """
    if not _models:
        _validate_required_files()

        print("Loading AUX ONNX model and scaler...")
        _models["aux_folds"] = [_load_aux_model()]
        print("Loading Main ONNX model and scaler...")
        _models["main_folds"] = [_load_main_model()]
        print(f"All models loaded successfully.")
    return _models