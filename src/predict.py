"""Inference helpers shared by the API and the tests."""

from __future__ import annotations

import os
from pathlib import Path

import joblib

DEFAULT_MODEL_PATH = Path("models/model.joblib")


def model_path() -> Path:
    """Model location; override with the MODEL_PATH environment variable."""
    return Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH)))


def load_model(path: str | Path | None = None):
    """Load a trained pipeline, or return None if it doesn't exist yet."""
    path = Path(path) if path is not None else model_path()
    if not path.exists():
        return None
    return joblib.load(path)


def predict(model, text: str) -> dict:
    """Classify ``text``; returns label, confidence, and per-class scores."""
    probabilities = model.predict_proba([text])[0]
    classes = list(model.classes_)
    ranked = sorted(zip(classes, probabilities), key=lambda pair: pair[1], reverse=True)
    return {
        "category": ranked[0][0],
        "confidence": round(float(ranked[0][1]), 4),
        "probabilities": {label: round(float(p), 4) for label, p in ranked},
    }
