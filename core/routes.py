"""
FastAPI route handlers for the CSIRO Biomass Prediction API.
"""

from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import DEVICE, DTYPE, TARGETS
from core.preprocessing import preprocess_image
from core.weights import get_models

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/labels")
async def labels():
    """Return the list of prediction target names."""
    return {"labels": TARGETS}


@router.post("/predict")
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
        raise HTTPException(
            status_code=400,
            detail="Image processing failed. Please upload a valid image.",
        )

    models = get_models()

    # Move image tensor to the inference device & dtype
    tensor = tensor.to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        # ---------------------------------------------------------------
        # Stage 1 – Ensemble auxiliary prediction (average across folds)
        # ---------------------------------------------------------------
        aux_preds = []
        for aux_model, tab_scaler in models["aux_folds"]:
            pred = aux_model(tensor).float().cpu().numpy()
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
            tab_tensor = torch.tensor(tab_input, dtype=DTYPE, device=DEVICE)

            raw = main_model(tensor, tab_tensor).float().cpu().numpy()
            if target_scaler is not None:
                raw = target_scaler.inverse_transform(raw)
            main_preds.append(raw)
        raw_pred = np.mean(main_preds, axis=0)  # averaged biomass

    flat_pred = np.atleast_1d(raw_pred).flatten()
    predictions = {
        name: max(0.0, float(flat_pred[i])) for i, name in enumerate(TARGETS)
    }

    return JSONResponse(
        content={
            "filename": file.filename,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
