from __future__ import annotations

import argparse
import pickle

from bootstrap import ensure_local_packages

ensure_local_packages()

from aggregate import aggregate_post_reaction
from common import MODELS_DIR
from preprocess import clean_text


def predict_comments(comments: list[str], model_path=MODELS_DIR / "best_model.pkl") -> dict:
    if not model_path.exists():
        raise SystemExit("Trained model not found. Run: python src/train_ml.py")

    with model_path.open("rb") as handle:
        pipeline = pickle.load(handle)

    clean_comments = [clean_text(comment) for comment in comments]
    labels = pipeline.predict(clean_comments).tolist()
    reaction = aggregate_post_reaction(labels)
    return {"comments": comments, "labels": labels, "reaction": reaction}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict comment sentiment with the trained ML model.")
    parser.add_argument("--comments", nargs="+", required=True, help="One or more comments to classify.")
    args = parser.parse_args()

    result = predict_comments(args.comments)
    for comment, label in zip(result["comments"], result["labels"], strict=True):
        print(f"[{label}] {comment}")
    print(result["reaction"])


if __name__ == "__main__":
    main()
