from __future__ import annotations

import json
import pickle
from pathlib import Path

from bootstrap import ensure_local_packages

ensure_local_packages()

from common import LABELS, MODELS_DIR, PROCESSED_DATA_PATH
from evaluate import build_metrics, save_confusion_matrix_plot, save_metrics


def ensure_training_dependencies() -> None:
    try:
        import pandas  # noqa: F401
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        from sklearn.metrics import f1_score  # noqa: F401
        from sklearn.model_selection import train_test_split  # noqa: F401
        from sklearn.naive_bayes import MultinomialNB  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401
        from sklearn.svm import LinearSVC  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing training dependencies. Install them with: pip install -r requirements.txt"
        ) from exc


def train_models(data_path: Path = PROCESSED_DATA_PATH) -> dict:
    ensure_training_dependencies()

    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    dataset = pd.read_csv(data_path)
    features = dataset["clean_comment"]
    targets = dataset["label"]

    x_train, x_temp, y_train, y_temp = train_test_split(
        features,
        targets,
        test_size=0.3,
        random_state=42,
        stratify=targets,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "linear_svm": LinearSVC(),
        "multinomial_nb": MultinomialNB(),
    }

    leaderboard = []
    best_name = None
    best_pipeline = None
    best_score = -1.0

    for name, estimator in candidate_models.items():
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        val_predictions = pipeline.predict(x_val)
        val_macro_f1 = f1_score(y_val, val_predictions, average="macro", zero_division=0)
        leaderboard.append({"model": name, "validation_macro_f1": float(val_macro_f1)})

        if val_macro_f1 > best_score:
            best_score = val_macro_f1
            best_name = name
            best_pipeline = pipeline

    test_predictions = best_pipeline.predict(x_test)
    metrics = build_metrics(y_test, test_predictions, LABELS)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "best_model.pkl"
    metadata_path = MODELS_DIR / "training_summary.json"
    metrics_path = MODELS_DIR / "test_metrics.json"
    confusion_path = MODELS_DIR / "confusion_matrix.png"

    with model_path.open("wb") as handle:
        pickle.dump(best_pipeline, handle)

    summary = {
        "best_model": best_name,
        "leaderboard": sorted(leaderboard, key=lambda item: item["validation_macro_f1"], reverse=True),
        "data_path": str(data_path),
        "saved_model_path": str(model_path),
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_metrics(metrics_path, metrics)
    save_confusion_matrix_plot(confusion_path, metrics["confusion_matrix"], LABELS)

    return {
        "summary": summary,
        "metrics": metrics,
    }


def main() -> None:
    result = train_models()
    print(f"Best model: {result['summary']['best_model']}")
    print(f"Saved model: {result['summary']['saved_model_path']}")
    print(f"Test accuracy: {result['metrics']['accuracy']:.3f}")
    print(f"Test macro F1: {result['metrics']['macro_f1']:.3f}")


if __name__ == "__main__":
    main()
