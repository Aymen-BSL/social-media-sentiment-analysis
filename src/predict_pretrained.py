from __future__ import annotations

import argparse

from bootstrap import ensure_local_packages

ensure_local_packages()

from aggregate import aggregate_post_reaction


def ensure_vader_dependency() -> None:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing VADER dependency. Install it with: pip install -r requirements.txt"
        ) from exc


def label_from_compound(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def predict_comments_with_vader(comments: list[str]) -> dict:
    ensure_vader_dependency()
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(comment) for comment in comments]
    labels = [label_from_compound(score["compound"]) for score in scores]
    reaction = aggregate_post_reaction(labels)
    return {"comments": comments, "scores": scores, "labels": labels, "reaction": reaction}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict comment sentiment with VADER.")
    parser.add_argument("--comments", nargs="+", required=True, help="One or more comments to classify.")
    args = parser.parse_args()

    result = predict_comments_with_vader(args.comments)
    for comment, score, label in zip(result["comments"], result["scores"], result["labels"], strict=True):
        print(f"[{label}] compound={score['compound']:.3f} :: {comment}")
    print(result["reaction"])


if __name__ == "__main__":
    main()
