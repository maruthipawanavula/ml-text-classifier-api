"""Train and evaluate the support-ticket classifier.

Pipeline: TF-IDF (word unigrams + bigrams, sublinear TF) feeding a
multinomial Logistic Regression classifier. Trains on the bundled
synthetic dataset, prints accuracy and a full classification report,
and saves the fitted pipeline plus a metrics JSON next to it.

Usage:
    python -m src.train [--data data/tickets.csv] [--model models/model.joblib]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

DEFAULT_DATA = Path("data/tickets.csv")
DEFAULT_MODEL = Path("models/model.joblib")
DEFAULT_METRICS = Path("models/metrics.json")


def load_dataset(path: str | Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ]
    )


def train(
    data_path: str | Path = DEFAULT_DATA,
    model_path: str | Path = DEFAULT_MODEL,
    metrics_path: str | Path = DEFAULT_METRICS,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Train on ``data_path``, evaluate on a held-out split, save artifacts.

    Returns the metrics dict (also written to ``metrics_path``).
    """
    texts, labels = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    model = build_pipeline()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

    metrics = {
        "dataset_rows": len(texts),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": round(float(accuracy), 4),
        "per_class": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
            }
            for label in sorted(set(labels))
        },
    }

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Dataset: {len(texts)} rows ({len(x_train)} train / {len(x_test)} test)")
    print(f"Test accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model saved to {model_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS))
    args = parser.parse_args()
    train(args.data, args.model, args.metrics)
