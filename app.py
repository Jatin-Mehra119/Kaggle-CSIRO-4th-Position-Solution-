"""
CSIRO Biomass Prediction - FastAPI Backend

Two-stage inference pipeline (CPU-based):
  Stage 1: Auxiliary model predicts NDVI & Height from the uploaded image.
  Stage 2: Main model uses image + predicted tabular features to predict biomass.

Model weights are automatically downloaded from Hugging Face Hub on first run.
"""

import io
import os
import warnings
from datetime import datetime, timezone

import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
# Image size must match training
IMG_SIZE = 800
MODEL_NAME = "vit_huge_plus_patch16_dinov3.lvd1689m"
TARGETS = ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Hugging Face Hub repositories for model weights
# ---------------------------------------------------------------------------
HF_AUX_REPO = "jatinmehra/CSIRO-AUX_MODEL"
HF_MAIN_REPO = "jatinmehra/CSIRO-DinoV3-HugePlus-LB76"

AUX_FOLDS = 5   # folds 0..4
MAIN_FOLDS = 5   # folds 0..4


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


# ============================================================================
# MODEL DEFINITIONS  (mirrors training scripts exactly)
# ============================================================================
class AuxModel(nn.Module):
    """Predicts [NDVI, Height] from an image (Stage 1)."""

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        img_features = self.backbone.num_features
        self.aux_head = nn.Sequential(
            nn.Linear(img_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(image)
        return self.aux_head(feat)


class BiomassModel(nn.Module):
    """Multi-modal model combining image + tabular features (Stage 2)."""

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        with torch.no_grad():
            dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            img_features = self.backbone(dummy).shape[1]

        self.tabular_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        fusion_dim = img_features + 128
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.head_green = nn.Linear(256, 1)
        self.head_dead = nn.Linear(256, 1)
        self.head_clover = nn.Linear(256, 1)
        self.head_gdm = nn.Linear(256, 1)
        self.head_total = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        img_feat = self.backbone(image)
        tab_feat = self.tabular_encoder(tabular)
        combined = torch.cat([img_feat, tab_feat], dim=1)
        fused = self.fusion(combined)

        out = torch.cat(
            [
                self.head_green(fused),
                self.head_dead(fused),
                self.head_clover(fused),
                self.head_gdm(fused),
                self.head_total(fused),
            ],
            dim=1,
        )
        return out


# ============================================================================
# PREPROCESSING (matches validation / inference transforms)
# ============================================================================
_transform = A.Compose(
    [
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """Decode uploaded bytes → preprocessed tensor (1, 3, H, W)."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = _transform(image=image)["image"].unsqueeze(0)  # (1,3,H,W)
    return tensor


# ============================================================================
# MODEL LOADING HELPERS
# ============================================================================
def _load_aux_model(path: str) -> tuple:
    """Return (AuxModel, tab_scaler | None)."""
    model = AuxModel(MODEL_NAME)
    tab_scaler = None
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        tab_scaler = ckpt.get("tab_scaler")
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
    model.eval()
    return model, tabular_scaler, target_scaler


def _load_all_aux_models() -> list[tuple]:
    """Download & load all AUX fold models. Returns list of (model, tab_scaler)."""
    paths = _download_aux_weights()
    return [_load_aux_model(p) for p in paths]


def _load_all_main_models() -> list[tuple]:
    """Download & load all Main fold models. Returns list of (model, tab_scaler, tgt_scaler)."""
    paths = _download_main_weights()
    return [_load_main_model(p) for p in paths]


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="CSIRO Biomass Prediction API",
    description="Upload a plant image to predict biomass targets.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Lazy-load models on first request so startup is fast even without weights.
# Weights are downloaded from Hugging Face Hub and cached locally.
# ---------------------------------------------------------------------------
_models: dict = {}


def _get_models() -> dict:
    if not _models:
        print("Downloading & loading AUX fold models from HF Hub...")
        _models["aux_folds"] = _load_all_aux_models()   # list[(model, tab_scaler)]
        print("Downloading & loading Main fold models from HF Hub...")
        _models["main_folds"] = _load_all_main_models()  # list[(model, tab_scaler, tgt_scaler)]
        print("All models loaded.")
    return _models


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/labels")
async def labels():
    """Return the list of prediction target names."""
    return {"labels": TARGETS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image and return biomass predictions.

    Pipeline:
      1. Auxiliary model predicts NDVI & Height from the image.
      2. Main model predicts 5 biomass targets using image + predicted tabular.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        tensor = preprocess_image(image_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Image processing failed. Please upload a valid image.")

    models = _get_models()

    with torch.no_grad():
        # ---------------------------------------------------------------
        # Stage 1 – Ensemble auxiliary prediction (average across folds)
        # ---------------------------------------------------------------
        aux_preds = []
        for aux_model, tab_scaler in models["aux_folds"]:
            pred = aux_model(tensor).numpy()
            if tab_scaler is not None:
                pred = tab_scaler.inverse_transform(pred)
            aux_preds.append(pred)
        aux_pred = np.mean(aux_preds, axis=0)  # averaged NDVI & Height

        # ---------------------------------------------------------------
        # Stage 2 – Ensemble biomass prediction (average across folds)
        # ---------------------------------------------------------------
        main_preds = []
        for main_model, tabular_scaler, target_scaler in models["main_folds"]:
            tab_input = aux_pred.copy()
            if tabular_scaler is not None:
                tab_input = tabular_scaler.transform(tab_input)
            tab_tensor = torch.tensor(tab_input, dtype=torch.float32)

            raw = main_model(tensor, tab_tensor).numpy()
            if target_scaler is not None:
                raw = target_scaler.inverse_transform(raw)
            main_preds.append(raw)
        raw_pred = np.mean(main_preds, axis=0)  # averaged biomass

    predictions = {
        name: max(0.0, float(raw_pred[0, i])) for i, name in enumerate(TARGETS)
    }

    return JSONResponse(
        content={
            "filename": file.filename,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
