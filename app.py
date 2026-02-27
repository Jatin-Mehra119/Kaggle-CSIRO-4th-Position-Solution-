"""
CSIRO Biomass Prediction - FastAPI Backend

Two-stage inference pipeline (GPU float16 when available, CPU float32 fallback):
  Stage 1: Auxiliary model predicts NDVI & Height from the uploaded image.
  Stage 2: Main model uses image + predicted tabular features to predict biomass.

Model weights are automatically downloaded from Hugging Face Hub on first run.
"""

import warnings

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.routes import router

warnings.filterwarnings("ignore")

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

app.include_router(router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)