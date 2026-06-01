import numpy as np
import pandas as pd

from src.features.build_features import build_tfidf, compute_weights, encode_labels


def test_build_tfidf_returns_correct_row_count():
    texts = ["I feel sad", "I feel great", "anxiety is real", "normal day"]
    tfidf, X = build_tfidf(texts, max_features=100)
    assert X.shape[0] == 4


def test_encode_labels_returns_array_with_correct_length():
    labels = pd.Series(["Normal", "Depression", "Suicidal", "Anxiety"])
    le, encoded = encode_labels(labels)
    assert len(encoded) == 4
    assert set(le.classes_) == {"Normal", "Depression", "Suicidal", "Anxiety"}


def test_compute_weights_gives_minority_higher_weight():
    # class 2 appears once, class 0 appears three times — class 2 should weigh more
    labels = np.array([0, 0, 0, 1, 1, 2])
    weights = compute_weights(labels)
    assert weights[2] > weights[0]


def test_build_tfidf_transform_matches_vocab():
    texts = ["feel sad hopeless", "feel great happy"]
    tfidf, X_train = build_tfidf(texts, max_features=50)
    X_new = tfidf.transform(["feel hopeless"])
    assert X_new.shape[1] == X_train.shape[1]
