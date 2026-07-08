import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import CLEAN_BASELINE_PLOT_FILE, CLEAN_BASELINE_RESULTS_FILE


def plot_clean_baseline(results_file, output_file):
    df = pd.read_csv(results_file)

    detected_df = df[df["both_faces_detected"]].copy()

    if detected_df.empty:
        raise RuntimeError("No rows with both faces detected. Cannot plot baseline.")

    similarities = detected_df["clean_similarity"].dropna()

    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(similarities, bins=20)
    ax.set_title("Clean Same-Identity Similarity Distribution")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Number of identity pairs")

    mean_similarity = similarities.mean()
    median_similarity = similarities.median()

    ax.axvline(mean_similarity, linestyle="--", label=f"Mean: {mean_similarity:.3f}")
    ax.axvline(
        median_similarity, linestyle=":", label=f"Median: {median_similarity:.3f}"
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    print(f"Saved plot to: {output_file}")
    print()
    print("Similarity summary:")
    print(similarities.describe())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-file", type=str, default=str(CLEAN_BASELINE_RESULTS_FILE)
    )
    parser.add_argument(
        "--output-file", type=str, default=str(CLEAN_BASELINE_PLOT_FILE)
    )
    args = parser.parse_args()

    plot_clean_baseline(
        results_file=Path(args.results_file),
        output_file=Path(args.output_file),
    )


if __name__ == "__main__":
    main()
