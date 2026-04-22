from __future__ import annotations

import json
from pathlib import Path

from bootstrap import ensure_local_packages

ensure_local_packages()


def ensure_ml_dependencies() -> None:
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        from sklearn.metrics import (  # noqa: F401
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing ML dependencies. Install them with: pip install -r requirements.txt"
        ) from exc


def build_metrics(y_true, y_pred, labels):
    ensure_ml_dependencies()
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_label": {},
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    for index, label in enumerate(labels):
        metrics["per_label"][label] = {
            "precision": precision[index],
            "recall": recall[index],
            "f1": f1[index],
            "support": int(support[index]),
        }
    return metrics


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_confusion_matrix_plot(path: Path, matrix, labels) -> None:
    ensure_ml_dependencies()
    import matplotlib.pyplot as plt
    import seaborn as sns

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
