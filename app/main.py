"""FastAPI service exposing the ticket classifier.

Run training first (``python -m src.train``), then:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import predict as predict_module

app = FastAPI(
    title="Ticket Category Classifier",
    description="Classifies support-ticket text into billing / technical / account / shipping.",
    version="1.0.0",
)

# Loaded once at startup. None until `python -m src.train` has been run.
MODEL = predict_module.load_model()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw support-ticket text")


class PredictResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "categories": list(MODEL.classes_) if MODEL is not None else [],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> dict:
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m src.train` first.",
        )
    return predict_module.predict(MODEL, request.text)
