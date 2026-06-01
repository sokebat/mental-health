from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def build_tfidf(
    train_texts: list[str],
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
) -> tuple[TfidfVectorizer, object]:
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    X_train = tfidf.fit_transform(train_texts)
    return tfidf, X_train


def encode_labels(labels: pd.Series) -> tuple[LabelEncoder, np.ndarray]:
    le = LabelEncoder()
    encoded = le.fit_transform(labels)
    return le, encoded


def compute_weights(labels: np.ndarray) -> np.ndarray:
    return compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )


def split_data(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def save_artifacts(
    tfidf: TfidfVectorizer,
    label_encoder: LabelEncoder,
    output_dir: str | Path,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf, out / "tfidf_vectorizer.pkl")
    joblib.dump(label_encoder, out / "label_encoder.pkl")
