from __future__ import annotations

import argparse
import re
from pathlib import Path

from bootstrap import ensure_local_packages

ensure_local_packages()

from common import PROCESSED_DATA_PATH, RAW_DATA_PATH, read_csv_rows, write_csv_rows

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    cleaned = text.lower().strip()
    cleaned = URL_PATTERN.sub(" ", cleaned)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)
    cleaned = HASHTAG_SYMBOL_PATTERN.sub("", cleaned)
    cleaned = NON_WORD_PATTERN.sub(" ", cleaned)
    cleaned = MULTISPACE_PATTERN.sub(" ", cleaned)
    return cleaned.strip()


def preprocess_dataset(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH) -> int:
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows = read_csv_rows(input_path)
    processed_rows = []
    for row in rows:
        processed_rows.append(
            {
                "comment": row["comment"],
                "clean_comment": clean_text(row["comment"]),
                "label": row["label"],
            }
        )

    write_csv_rows(
        output_path,
        processed_rows,
        fieldnames=["comment", "clean_comment", "label"],
    )
    return len(processed_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and normalize social media comments.")
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--output", default=str(PROCESSED_DATA_PATH))
    args = parser.parse_args()

    count = preprocess_dataset(args.input, args.output)
    print(f"Preprocessed {count} comments into {args.output}")


if __name__ == "__main__":
    main()
