"""Print all unique tags (themes) found in the first 1000 lichess puzzles.

Streams the same puzzle dataset used elsewhere in this folder and collects
the space-delimited theme tokens from the themes column.

Run:
    python leela_interp/lookahead/explore_dataset.py
"""

import argparse
import os

# Column names in SubhamDB/chess-puzzles are single letters:
#   i=id, f=FEN, m=moves, r=rating, t=themes
THEMES_KEY = "Themes"


def loadDataset(name):
    """Same streaming loader as the other lookahead scripts (kept standalone
    so this script does not drag torch/deepspeed in)."""
    from datasets import load_dataset

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        print("python-dotenv not installed; skipping .env load")

    token = os.getenv("HUGGINGFACE_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token)

    return load_dataset(name, split="train", streaming=True)


def collectTags(dataset, num_samples):
    from itertools import islice

    tags = set()
    for item in islice(dataset, num_samples):
        themes = item.get(THEMES_KEY, "") or ""
        for tag in themes:
            tag = tag.strip()
            if tag:
                tags.add(tag)
    return tags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="SubhamDB/chess-puzzles",
        help="HuggingFace dataset to stream (default matches the other lookahead scripts)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="How many puzzles to pull off the stream",
    )
    args = parser.parse_args()

    dataset = loadDataset(args.dataset)
    tags = collectTags(dataset, args.num_samples)

    print(f"Unique tags found in first {args.num_samples} puzzles: {len(tags)}")
    for tag in sorted(tags):
        print(tag)


if __name__ == "__main__":
    main()
