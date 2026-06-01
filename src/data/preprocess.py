import re
import unicodedata
from pathlib import Path

import pandas as pd


def preprocess_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text))
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[@#]\S+", "", text)
    text = text.replace("\t", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "status",
    min_word_count: int = 3,
    max_length_percentile: float = 0.99,
) -> pd.DataFrame:
    df = df[[text_col, label_col]].copy()
    df = df.drop_duplicates(subset=text_col)
    df["word_count"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    df["text_length"] = df[text_col].astype(str).apply(len)
    df = df[df["word_count"] >= min_word_count]
    max_len = df["text_length"].quantile(max_length_percentile)
    df = df[df["text_length"] <= max_len]
    df["clean_text"] = df[text_col].astype(str).apply(preprocess_text)
    return df[[text_col, label_col, "clean_text"]].reset_index(drop=True)


def load_and_clean(
    raw_path: str | Path,
    output_path: str | Path | None = None,
    min_word_count: int = 3,
    max_length_percentile: float = 0.99,
) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df = clean_dataframe(df, min_word_count=min_word_count, max_length_percentile=max_length_percentile)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
