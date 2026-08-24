"""Check the rating distribution of the chess puzzles used for lookahead probing.

Streams the puzzle dataset (same source as leela_interp_lookahead.py), collects
the rating field, and writes a seaborn figure plus a printed summary.

Two panels: every puzzle sampled, and the subset that actually survives the
lookahead filter (a puzzle needs 1 + 2*LOOKAHEAD_NUM moves to be usable), so it
is visible whether that filter skews the ratings the probe trains on.

Run:
    python leela_interp/lookahead/puzzle_rating_distribution.py --num-samples 50000
"""

import argparse
import os

# Column names in SubhamDB/chess-puzzles are single letters:
#   i=id, f=FEN, m=moves, r=rating, t=themes
RATING_KEY = "r"
MOVES_KEY = "m"

# Mirrors LOOKAHEAD_NUM in leela_interp_lookahead.py; a puzzle is only usable if
# it has a given first move plus LOOKAHEAD_NUM full move-pairs after it.
LOOKAHEAD_NUM = 2

# Tokens from the chart palette: sequential blue on a light surface.
FILL = "#3987e5"
FILL_SUBSET = "#184f95"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def loadDataset(name):
    """Same streaming loader as the training script (kept standalone so this
    script does not drag torch/deepspeed in)."""
    from datasets import load_dataset

    # The token is only needed for gated datasets, so a missing python-dotenv
    # (or a missing token) is not fatal here -- the puzzle sets are public.
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


def collectRatings(dataset, num_samples):
    """Walk the stream once, returning a row per puzzle with its rating and
    whether it is long enough for lookahead."""
    from itertools import islice

    import pandas as pd

    rows = []
    seen_fens = set()
    for item in islice(dataset, num_samples):
        fen = item.get("f", "")
        if fen in seen_fens:
            # generateActivations drops duplicate FENs too, so drop them here
            # as well or the distribution over-counts repeated positions.
            continue
        seen_fens.add(fen)

        rating = item.get(RATING_KEY)
        if rating is None:
            continue

        moves = item.get(MOVES_KEY, "").split(" ")
        rows.append(
            {
                "rating": int(rating),
                "num_moves": len(moves),
                "usable": len(moves) >= 1 + LOOKAHEAD_NUM * 2,
            }
        )

    return pd.DataFrame(rows)


def summarize(df):
    print(f"Puzzles sampled (unique FENs): {len(df)}")
    usable = df[df["usable"]]
    print(f"Usable for lookahead (>= {1 + LOOKAHEAD_NUM * 2} moves): {len(usable)}")
    print()
    print("Rating distribution -- all puzzles")
    print(df["rating"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))
    print()
    print("Rating distribution -- lookahead-usable puzzles")
    if usable.empty:
        print("(none)")
    else:
        print(usable["rating"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]))


def plot(df, out_path, bin_width):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor(SURFACE)

    usable = df[df["usable"]]
    panels = [
        (axes[0], df, FILL, f"All puzzles sampled (n={len(df)})"),
        (
            axes[1],
            usable,
            FILL_SUBSET,
            f"Lookahead-usable puzzles, >= {1 + LOOKAHEAD_NUM * 2} moves (n={len(usable)})",
        ),
    ]

    for ax, data, color, title in panels:
        ax.set_facecolor(SURFACE)
        if not data.empty:
            sns.histplot(
                data=data,
                x="rating",
                binwidth=bin_width,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.5,
                ax=ax,
            )
            median = data["rating"].median()
            ax.axvline(median, color=INK, linewidth=1.5, linestyle="--")
            # Direct-label the median instead of annotating every bar.
            ax.text(
                median,
                ax.get_ylim()[1] * 0.95,
                f" median {median:.0f}",
                color=INK,
                va="top",
                fontsize=10,
            )
        ax.set_title(title, color=INK, fontsize=12, loc="left")
        ax.set_ylabel("Puzzles", color=MUTED)
        ax.grid(color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[1].set_xlabel("Lichess puzzle rating", color=MUTED)
    fig.suptitle("Puzzle rating distribution", color=INK, fontsize=15, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="SubhamDB/chess-puzzles",
        help="HuggingFace dataset to stream (default matches the training script)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="How many rows to pull off the stream",
    )
    parser.add_argument("--bin-width", type=int, default=50, help="Rating bin width")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "puzzle_rating_distribution.png"),
        help="Where to write the figure",
    )
    parser.add_argument("--csv", default=None, help="Optional path to dump the raw ratings")
    args = parser.parse_args()

    dataset = loadDataset(args.dataset)
    df = collectRatings(dataset, args.num_samples)
    if df.empty:
        raise SystemExit("No ratings collected -- check the dataset name and rating column.")

    summarize(df)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Wrote {args.csv}")
    plot(df, args.out, args.bin_width)


if __name__ == "__main__":
    main()
