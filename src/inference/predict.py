"""
Inference module — sklearn backend.

Usage:
  python -m src.inference.predict --text "I feel hopeless"
  python -m src.inference.predict --input data/processed/sample.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.data.preprocess import preprocess_text


def predict_sklearn(
    texts: list[str],
    model_dir: str | Path,
    model_type: str = "logistic_regression",
) -> list[str]:
    d = Path(model_dir)
    tfidf = joblib.load(d / "tfidf_vectorizer.pkl")
    le = joblib.load(d / "label_encoder.pkl")
    model = joblib.load(d / f"{model_type}.pkl")

    cleaned = [preprocess_text(t) for t in texts]
    X = tfidf.transform(cleaned)
    preds = model.predict(X)
    return le.inverse_transform(preds).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Single text input")
    parser.add_argument("--input", help="Path to CSV with a 'text' column")
    parser.add_argument("--model-type", default="logistic_regression")
    parser.add_argument("--sklearn-dir", default="models/sklearn")
    args = parser.parse_args()

    texts = []
    if args.text:
        texts = [args.text]
    elif args.input:
        texts = pd.read_csv(args.input)["text"].tolist()
    else:
        raise ValueError("Provide --text or --input")

    results = predict_sklearn(texts, args.sklearn_dir, args.model_type)

    for text, label in zip(texts, results):
        print(f"[{label}] {text[:80]}")
