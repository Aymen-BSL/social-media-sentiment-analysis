from __future__ import annotations

from bootstrap import ensure_local_packages

ensure_local_packages()

from common import read_sample_posts


def run_ml_demo(posts: list[dict]) -> None:
    from predict_ml import predict_comments

    print("=== ML DEMO ===")
    for post in posts:
        result = predict_comments(post["comments"])
        print(f"\n{post['post_id']} - {post['title']}")
        for comment, label in zip(post["comments"], result["labels"], strict=True):
            print(f"[{label}] {comment}")
        print(result["reaction"])


def run_vader_demo(posts: list[dict]) -> None:
    from predict_pretrained import predict_comments_with_vader

    print("=== VADER DEMO ===")
    for post in posts:
        result = predict_comments_with_vader(post["comments"])
        print(f"\n{post['post_id']} - {post['title']}")
        for comment, label in zip(post["comments"], result["labels"], strict=True):
            print(f"[{label}] {comment}")
        print(result["reaction"])


def main() -> None:
    posts = read_sample_posts()
    try:
        run_ml_demo(posts)
    except SystemExit as exc:
        print(f"ML demo skipped: {exc}")

    try:
        run_vader_demo(posts)
    except SystemExit as exc:
        print(f"VADER demo skipped: {exc}")


if __name__ == "__main__":
    main()
