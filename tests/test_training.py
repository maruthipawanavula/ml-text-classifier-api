"""Training smoke tests: the pipeline must train, save, and learn the data."""

from pathlib import Path

from src import predict as predict_module
from src.train import load_dataset, train

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


def test_dataset_is_balanced_and_labeled():
    texts, labels = load_dataset(DATA)
    assert len(texts) == len(labels) >= 400
    assert set(labels) == {"billing", "technical", "account", "shipping"}
    for label in set(labels):
        assert labels.count(label) >= 100


def test_training_produces_accurate_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    metrics = train(DATA, model_path, metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()
    # Held-out accuracy on the synthetic dataset must clear a sane bar.
    assert metrics["accuracy"] >= 0.85

    model = predict_module.load_model(model_path)
    assert model is not None


def test_predictions_for_each_category(tmp_path):
    model_path = tmp_path / "model.joblib"
    train(DATA, model_path, tmp_path / "metrics.json")
    model = predict_module.load_model(model_path)

    examples = {
        "billing": "I was charged twice and I need a refund on my card.",
        "technical": "The app crashes with a 500 error when I open the settings page.",
        "account": "I forgot my password and want to change my account email.",
        "shipping": "My package tracking has not updated and the delivery is late.",
    }
    for expected, text in examples.items():
        result = predict_module.predict(model, text)
        assert result["category"] == expected, (text, result)
        assert 0.0 <= result["confidence"] <= 1.0
