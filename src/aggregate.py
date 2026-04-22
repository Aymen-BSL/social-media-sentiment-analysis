from __future__ import annotations

from collections import Counter
from typing import Iterable


def aggregate_post_reaction(
    labels: Iterable[str],
    clear_margin: int = 2,
    clear_ratio: float = 0.2,
) -> dict[str, object]:
    counts = Counter(labels)
    positive = counts.get("positive", 0)
    negative = counts.get("negative", 0)
    neutral = counts.get("neutral", 0)
    total = positive + negative + neutral

    if total == 0:
        reaction = "Mixed Reaction"
    else:
        margin = abs(positive - negative)
        threshold = max(clear_margin, int(total * clear_ratio))
        if positive > negative and margin >= threshold:
            reaction = "Liked"
        elif negative > positive and margin >= threshold:
            reaction = "Not Liked"
        else:
            reaction = "Mixed Reaction"

    return {
        "reaction": reaction,
        "counts": {
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "total": total,
        },
    }
