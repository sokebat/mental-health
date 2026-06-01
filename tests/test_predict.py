"""
Integration tests for the sklearn inference path.
Skipped automatically when trained model artifacts are not present.
"""

from pathlib import Path

import pytest

from src.data.preprocess import preprocess_text
from src.inference.predict import predict_sklearn

SKLEARN_DIR = Path("models/sklearn")
VALID_LABELS = {"Anxiety", "Depression", "Normal", "Suicidal"}


@pytest.mark.skipif(
    not (SKLEARN_DIR / "logistic_regression.pkl").exists(),
    reason="Trained sklearn artifacts not found — run: python -m src.training.train --config configs/config.yaml --mode sklearn",
)
def test_predict_sklearn_returns_valid_label():
    results = predict_sklearn(["I feel hopeless and deeply sad"], SKLEARN_DIR)
    assert len(results) == 1
    assert results[0] in VALID_LABELS


@pytest.mark.skipif(
    not (SKLEARN_DIR / "logistic_regression.pkl").exists(),
    reason="Trained sklearn artifacts not found",
)
def test_predict_sklearn_batch():
    texts = [
        "I feel anxious all the time",
        "Everything is fine, I had a great day",
        "I cannot stop thinking about ending it all",
    ]
    results = predict_sklearn(texts, SKLEARN_DIR)
    assert len(results) == 3
    assert all(r in VALID_LABELS for r in results)


def test_preprocess_text_is_deterministic():
    text = "I FEEL ANXIOUS! Visit https://help.org for support. @therapist #mentalhealth"
    assert preprocess_text(text) == preprocess_text(text)


def test_preprocess_text_produces_non_empty_output():
    result = preprocess_text("I feel very sad and hopeless today")
    assert len(result) > 0
