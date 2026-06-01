from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def create_sklearn_model(model_type: str, params: dict | None = None):
    params = params or {}

    if model_type == "logistic_regression":
        return LogisticRegression(**{"max_iter": 1000, "class_weight": "balanced", **params})

    if model_type == "naive_bayes":
        return MultinomialNB(**params)

    if model_type == "random_forest":
        return RandomForestClassifier(
            **{"n_estimators": 100, "class_weight": "balanced", "n_jobs": -1, **params}
        )

    if model_type == "svm":
        return LinearSVC(**{"max_iter": 1000, "class_weight": "balanced", **params})

    raise ValueError(f"Unsupported sklearn model type: {model_type}")
