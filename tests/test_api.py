"""API tests using FastAPI's TestClient against a freshly trained model."""

import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.train import train

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a model into a temp dir and point the app at it BEFORE importing
    # app.main, because the app loads the model once at startup.
    model_dir = tmp_path_factory.mktemp("model")
    model_path = model_dir / "model.joblib"
    train(DATA, model_path, model_dir / "metrics.json")

    previous = os.environ.get("MODEL_PATH")
    os.environ["MODEL_PATH"] = str(model_path)
    try:
        import app.main as main_module

        importlib.reload(main_module)
        with TestClient(main_module.app) as test_client:
            yield test_client
    finally:
        if previous is None:
            os.environ.pop("MODEL_PATH", None)
        else:
            os.environ["MODEL_PATH"] = previous


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert set(body["categories"]) == {"billing", "technical", "account", "shipping"}


def test_predict_billing_ticket(client):
    response = client.post(
        "/predict",
        json={"text": "My invoice shows a charge I don't recognize and I want a refund."},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "billing"
    assert body["confidence"] > 0.5
    assert set(body["probabilities"]) == {"billing", "technical", "account", "shipping"}


def test_predict_shipping_ticket(client):
    response = client.post(
        "/predict",
        json={"text": "My order has not arrived and the tracking page has not updated in days."},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "shipping"


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
