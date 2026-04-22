from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "comments.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "comments_clean.csv"
SAMPLE_POSTS_PATH = PROJECT_ROOT / "data" / "raw" / "sample_posts.json"
MODELS_DIR = PROJECT_ROOT / "models"

LABELS = ("positive", "negative", "neutral")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_sample_posts() -> list[dict]:
    with SAMPLE_POSTS_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["posts"]
