"""
Neural network definitions for the two-stage biomass prediction pipeline.

AuxModel   – Stage 1: predicts NDVI & Height from an image.
BiomassModel – Stage 2: predicts 5 biomass targets from image + tabular features.
"""

import torch
import torch.nn as nn
import timm

from core.config import IMG_SIZE


class AuxModel(nn.Module):
    """Predicts [NDVI, Height] from an image (Stage 1)."""

    def __init__(self, model_name: str) -> None:
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

    def __init__(self, model_name: str) -> None:
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
