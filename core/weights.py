"""
Model weight download (from Hugging Face Hub) and checkpoint loading utilities.
"""

import os

import torch
from huggingface_hub import hf_hub_download

from core.config import (
    AUX_FOLDS,
    DEVICE,
    DTYPE,
    HF_AUX_REPO,
    HF_MAIN_REPO,
    MAIN_FOLDS,
    MODEL_NAME,
)
from core.models import AuxModel, BiomassModel


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_weights(repo_id: str, filename: str, subfolder: str | None = None) -> str:
    """Download a single weight file from HF Hub (cached after first download)."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
    )


def _download_aux_weights() -> list[str]:
    """Download all AUX fold weights and return their local paths."""
    paths = []
    for fold in range(AUX_FOLDS):
        path = _download_weights(
            HF_AUX_REPO,
            f"best_aux_only_seed44_fold{fold}.pth",
            subfolder="Models_Aux_Only_v7",
        )
        paths.append(path)
    return paths


def _download_main_weights() -> list[str]:
    """Download all Main fold weights and return their local paths."""
    paths = []
    for fold in range(MAIN_FOLDS):
        path = _download_weights(
            HF_MAIN_REPO,
            f"best_model_seed42_fold{fold}.pth",
        )
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_aux_model(path: str) -> tuple:
    """Return (AuxModel, tab_scaler | None)."""
    model = AuxModel(MODEL_NAME)
    tab_scaler = None
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        tab_scaler = ckpt.get("tab_scaler")
    model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model, tab_scaler


def _load_main_model(path: str) -> tuple:
    """Return (BiomassModel, tabular_scaler | None, target_scaler | None)."""
    model = BiomassModel(MODEL_NAME)
    tabular_scaler = None
    target_scaler = None
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        tabular_scaler = ckpt.get("tabular_scaler")
        target_scaler = ckpt.get("target_scaler")
    model.to(device=DEVICE, dtype=DTYPE)
    model.eval()
    return model, tabular_scaler, target_scaler


# ---------------------------------------------------------------------------
# Public API – lazy-loaded model registry
# ---------------------------------------------------------------------------

_models: dict = {}


def get_models() -> dict:
    """
    Download (once) and return all fold models.

    Returns a dict with keys:
      - ``aux_folds``  : list[(AuxModel, tab_scaler)]
      - ``main_folds`` : list[(BiomassModel, tabular_scaler, target_scaler)]
    """
    if not _models:
        print("Downloading & loading AUX fold models from HF Hub...")
        _models["aux_folds"] = [_load_aux_model(p) for p in _download_aux_weights()]
        print("Downloading & loading Main fold models from HF Hub...")
        _models["main_folds"] = [_load_main_model(p) for p in _download_main_weights()]
        print(f"All models loaded on {DEVICE} with dtype {DTYPE}.")
    return _models
