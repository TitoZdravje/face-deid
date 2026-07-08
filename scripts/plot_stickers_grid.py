import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    STICKER_GRID_POSITION_PLOT_FILE,
    STICKER_GRID_RESULTS_FILE,
    STICKER_GRID_SCALE_PLOT_FILE,
    STICKER_GRID_SUMMARY_PLOT_FILE,
)


def load_valid_results(
    results_file: Path, only_usable_clean_pairs: bool
) -> pd.DataFrame:
    df = pd.read_csv(results_file)

    valid_df = df[
        df["clean_both_faces_detected"]
        & df["stickered_face_detected"]
        & df["similarity_drop"].notna()
    ].copy()

    if only_usable_clean_pairs and "clean_pair_usable" in valid_df.columns:
        valid_df = valid_df[valid_df["clean_pair_usable"]]

    if valid_df.empty:
        raise RuntimeError("No valid sticker evaluation rows to plot.")

    return valid_df


def plot_mean_drop_by_sticker(df: pd.DataFrame, output_file: Path):
    summary = (
        df.groupby("sticker_name")["similarity_drop"]
        .mean()
        .sort_values(ascending=False)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    summary.plot(kind="bar", ax=ax)

    ax.set_title("Average Similarity Drop by Sticker")
    ax.set_xlabel("Sticker")
    ax.set_ylabel("Mean similarity drop")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_file}")


def plot_mean_drop_by_position(df: pd.DataFrame, output_file: Path):
    summary = (
        df.groupby("position_name")["similarity_drop"]
        .mean()
        .sort_values(ascending=False)
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    summary.plot(kind="bar", ax=ax)

    ax.set_title("Average Similarity Drop by Sticker Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Mean similarity drop")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_file}")


def plot_mean_drop_by_scale(df: pd.DataFrame, output_file: Path):
    summary = df.groupby("scale_factor")["similarity_drop"].mean().sort_index()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    summary.plot(kind="line", marker="o", ax=ax)

    ax.set_title("Average Similarity Drop by Sticker Scale")
    ax.set_xlabel("Scale factor relative to eye distance")
    ax.set_ylabel("Mean similarity drop")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_file}")


def print_text_summary(df: pd.DataFrame):
    print()
    print("Overall similarity drop:")
    print(df["similarity_drop"].describe())

    print()
    print("Best average stickers:")
    print(
        df.groupby("sticker_name")["similarity_drop"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    print()
    print("Best average positions:")
    print(
        df.groupby("position_name")["similarity_drop"]
        .mean()
        .sort_values(ascending=False)
    )

    print()
    print("Best average scale factors:")
    print(df.groupby("scale_factor")["similarity_drop"].mean().sort_index())

    print()
    print("Top individual rows:")
    columns = [
        "identity",
        "probe_image",
        "sticker_name",
        "position_name",
        "scale_factor",
        "clean_similarity",
        "stickered_similarity",
        "similarity_drop",
    ]

    available_columns = [col for col in columns if col in df.columns]

    print(
        df.sort_values("similarity_drop", ascending=False)[available_columns].head(10)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file", type=str, default=str(STICKER_GRID_RESULTS_FILE)
    )

    parser.add_argument(
        "--only-usable-clean-pairs",
        action="store_true",
        help="Only plot rows where clean_pair_usable is True.",
    )

    args = parser.parse_args()

    df = load_valid_results(
        results_file=Path(args.results_file),
        only_usable_clean_pairs=args.only_usable_clean_pairs,
    )

    print_text_summary(df)

    plot_mean_drop_by_sticker(
        df=df,
        output_file=STICKER_GRID_SUMMARY_PLOT_FILE,
    )

    plot_mean_drop_by_position(
        df=df,
        output_file=STICKER_GRID_POSITION_PLOT_FILE,
    )

    plot_mean_drop_by_scale(
        df=df,
        output_file=STICKER_GRID_SCALE_PLOT_FILE,
    )


if __name__ == "__main__":
    main()
