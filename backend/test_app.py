"""Tests for the CSIRO Biomass Prediction FastAPI backend."""

import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Clear the lazy-loaded model cache between tests."""
    from backend.app import _models
    _models.clear()
    yield
    _models.clear()


@pytest.fixture()
def client():
    from backend.app import app
    return TestClient(app)


def _make_test_image_bytes() -> bytes:
    """Create a minimal valid JPEG in memory."""
    import cv2
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[10:50, 10:50] = (0, 200, 0)  # green square
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestLabelsEndpoint:
    def test_labels(self, client):
        resp = client.get("/labels")
        assert resp.status_code == 200
        data = resp.json()
        assert data["labels"] == [
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        ]


class TestPredictEndpoint:
    @patch("backend.app._get_models")
    def test_predict_returns_all_targets(self, mock_get_models, client):
        """POST /predict with a valid image returns all 5 biomass targets."""
        import torch

        # Build lightweight mock models
        aux_model = MagicMock()
        aux_model.return_value = torch.tensor([[0.5, 1.2]])

        main_model = MagicMock()
        main_model.return_value = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        mock_get_models.return_value = {
            "aux": aux_model,
            "tab_scaler": None,
            "main": main_model,
            "tabular_scaler": None,
            "target_scaler": None,
        }

        img_bytes = _make_test_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "timestamp" in data
        assert "filename" in data
        preds = data["predictions"]
        assert set(preds.keys()) == {
            "Dry_Green_g",
            "Dry_Dead_g",
            "Dry_Clover_g",
            "GDM_g",
            "Dry_Total_g",
        }
        # Values should be non-negative (clamped to 0)
        for v in preds.values():
            assert v >= 0.0

    @patch("backend.app._get_models")
    def test_predict_clamps_negatives(self, mock_get_models, client):
        """Negative raw predictions should be clamped to 0."""
        import torch

        aux_model = MagicMock()
        aux_model.return_value = torch.tensor([[0.0, 0.0]])

        main_model = MagicMock()
        main_model.return_value = torch.tensor([[-10.0, -5.0, 0.0, 1.0, 2.0]])

        mock_get_models.return_value = {
            "aux": aux_model,
            "tab_scaler": None,
            "main": main_model,
            "tabular_scaler": None,
            "target_scaler": None,
        }

        img_bytes = _make_test_image_bytes()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        preds = resp.json()["predictions"]
        assert preds["Dry_Green_g"] == 0.0
        assert preds["Dry_Dead_g"] == 0.0
        assert preds["Dry_Clover_g"] == 0.0
        assert preds["GDM_g"] >= 0.0

    def test_predict_rejects_non_image(self, client):
        """Non-image uploads should return 400."""
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 400
