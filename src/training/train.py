"""
Training entry point.

Usage:
  python -m src.training.train --config configs/config.yaml
"""

import argparse
import time
from pathlib import Path

import joblib
import yaml

from src.data.preprocess import load_and_clean
from src.evaluation.metrics import classification_metrics, print_report
from src.features.build_features import (
    build_tfidf,
    encode_labels,
    save_artifacts,
    split_data,
)
from src.models.model_factory import create_sklearn_model
from src.utils.logger import get_logger


def train_sklearn(config: dict, logger) -> dict:
    cfg_data = config["data"]
    cfg_tfidf = config["tfidf"]
    cfg_models = config["sklearn_models"]
    out_dir = Path(config["training"]["sklearn_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading and cleaning data...")
    df = load_and_clean(cfg_data["processed_path"])

    le, _ = encode_labels(df["status"])
    df["label"] = le.transform(df["status"])

    train_df, val_df = split_data(
        df,
        test_size=cfg_data["test_size"],
        random_state=config["project"]["random_state"],
    )

    logger.info("Building TF-IDF features...")
    tfidf, X_train = build_tfidf(
        train_df["clean_text"].tolist(),
        max_features=cfg_tfidf["max_features"],
        ngram_range=tuple(cfg_tfidf["ngram_range"]),
    )
    X_val = tfidf.transform(val_df["clean_text"].tolist())
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    save_artifacts(tfidf, le, out_dir)
    logger.info(f"TF-IDF matrix shape: {X_train.shape}")

    results = {}
    for model_type, params in cfg_models.items():
        logger.info(f"Training {model_type}...")
        start = time.time()
        model = create_sklearn_model(model_type, params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        elapsed = time.time() - start

        metrics = classification_metrics(y_val, preds)
        results[model_type] = {**metrics, "time_s": round(elapsed, 1)}
        logger.info(
            f"  {model_type}: acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_macro']:.4f}  [{elapsed:.1f}s]"
        )
        print_report(y_val, preds, le.classes_, model_type)

        joblib.dump(model, out_dir / f"{model_type}.pkl")

    logger.info("All sklearn models saved.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log = get_logger(__name__, cfg["logging"]["log_file"])
    train_sklearn(cfg, log)
