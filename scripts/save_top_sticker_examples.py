import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.celeba_io import load_celeba_image
from src.config import (
    CELEBA_IMAGES_DIR,
    GENERATED_STICKERS_DIR,
    STICKER_GRID_ANALYZED_RESULTS_FILE,
    STICKER_GRID_RESULTS_FILE,
    TOP_STICKER_EXAMPLES_DIR,
)
from src.stickers import apply_sticker, load_sticker_library


def safe_name(value) -> str:
    value = str(value)
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_")


def as_bool(value) -> bool:
    """
    Handles booleans that may come from pandas as bools or strings.
    """

    if isinstance(value, bool):
        return value

    if pd.isna(value):
        return False

    value = str(value).strip().lower()

    return value in {"true", "1", "yes"}


def save_example_figure(
    row: pd.Series,
    stickers: dict,
    output_path: Path,
    title_prefix: str,
):
    reference_image = load_celeba_image(
        CELEBA_IMAGES_DIR,
        row["reference_image"],
    )
    probe_image = load_celeba_image(
        CELEBA_IMAGES_DIR,
        row["probe_image"],
    )

    sticker_name = row["sticker_name"]

    if sticker_name not in stickers:
        raise KeyError(
            f"Sticker '{sticker_name}' not found. "
            f"Available: {list(stickers.keys())}"
        )

    sticker = stickers[sticker_name]

    stickered_image = apply_sticker(
        base_image=probe_image,
        sticker=sticker,
        center_xy=(float(row["center_x"]), float(row["center_y"])),
        target_width=float(row["target_width_px"]),
        rotation_degrees=float(row.get("rotation_degrees", 0.0)),
        opacity=float(row.get("opacity", 1.0)),
    )

    clean_similarity = row.get("clean_similarity", None)
    stickered_similarity = row.get("stickered_similarity", None)
    similarity_drop = row.get("similarity_drop", None)

    clean_recognized = row.get("clean_recognized", None)
    stickered_recognized = row.get("stickered_recognized", None)
    attack_success = row.get("attack_success", None)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(reference_image)
    axes[0].set_title(
        f"Reference\nID {row['identity']}\n{row['reference_image']}",
        fontsize=9,
    )
    axes[0].axis("off")

    axes[1].imshow(probe_image)
    axes[1].set_title(
        (
            f"Clean probe\n{row['probe_image']}\n" f"sim={clean_similarity:.3f}"
            if pd.notna(clean_similarity)
            else f"Clean probe\n{row['probe_image']}"
        ),
        fontsize=9,
    )
    axes[1].axis("off")

    if pd.notna(stickered_similarity):
        stickered_title = (
            f"Stickered probe\n"
            f"{sticker_name}, {row['position_name']}, scale {row['scale_factor']}\n"
            f"sim={stickered_similarity:.3f}, drop={similarity_drop:.3f}"
        )
    else:
        stickered_title = (
            f"Stickered probe\n"
            f"{sticker_name}, {row['position_name']}, scale {row['scale_factor']}\n"
            f"{row.get('stickered_reason', 'unknown')}"
        )

    axes[2].imshow(stickered_image)
    axes[2].set_title(stickered_title, fontsize=9)
    axes[2].axis("off")

    extra = ""

    if clean_recognized is not None and "clean_recognized" in row:
        extra += f"clean_recognized={clean_recognized}, "

    if stickered_recognized is not None and "stickered_recognized" in row:
        extra += f"stickered_recognized={stickered_recognized}, "

    if attack_success is not None and "attack_success" in row:
        extra += f"attack_success={attack_success}"

    fig.suptitle(
        f"{title_prefix}\n" f"{extra}",
        fontsize=11,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def select_examples(
    df: pd.DataFrame,
    category: str,
    top_n: int,
) -> pd.DataFrame:
    if category == "best_similarity_drop":
        candidates = df[
            df["stickered_face_detected"] & df["similarity_drop"].notna()
        ].copy()

        return candidates.sort_values(
            "similarity_drop",
            ascending=False,
        ).head(top_n)

    if category == "worst_similarity_drop":
        candidates = df[
            df["stickered_face_detected"] & df["similarity_drop"].notna()
        ].copy()

        return candidates.sort_values(
            "similarity_drop",
            ascending=True,
        ).head(top_n)

    if category == "attack_success":
        if "attack_success" not in df.columns:
            return pd.DataFrame()

        candidates = df[df["attack_success"]].copy()

        if "similarity_drop" in candidates.columns:
            candidates = candidates.sort_values(
                "similarity_drop",
                ascending=False,
            )

        return candidates.head(top_n)

    if category == "detection_failure":
        candidates = df[
            df["clean_both_faces_detected"] & (~df["stickered_face_detected"])
        ].copy()

        return candidates.head(top_n)

    if category == "best_natural":
        candidates = df[
            df["stickered_face_detected"]
            & df["similarity_drop"].notna()
            & (~df["sticker_name"].isin(["checker_noise"]))
        ].copy()

        return candidates.sort_values(
            "similarity_drop",
            ascending=False,
        ).head(top_n)

    if category == "best_non_noise_non_skin":
        candidates = df[
            df["stickered_face_detected"]
            & df["similarity_drop"].notna()
            & (~df["sticker_name"].isin(["checker_noise", "skin_patch"]))
        ].copy()

        return candidates.sort_values(
            "similarity_drop",
            ascending=False,
        ).head(top_n)

    raise ValueError(f"Unknown category: {category}")


def save_examples(
    results_file: Path,
    sticker_dir: Path,
    output_dir: Path,
    top_n: int,
):
    df = pd.read_csv(results_file)

    stickers = load_sticker_library(sticker_dir)

    categories = [
        "best_similarity_drop",
        "worst_similarity_drop",
        "attack_success",
        "detection_failure",
        "best_natural",
        "best_non_noise_non_skin",
    ]

    index_rows = []

    for category in categories:
        selected = select_examples(
            df=df,
            category=category,
            top_n=top_n,
        )

        if selected.empty:
            print(f"Skipping {category}: no rows found.")
            continue

        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for rank, (_, row) in enumerate(selected.iterrows(), start=1):
            filename = (
                f"{rank:03d}_"
                f"id_{safe_name(row['identity'])}_"
                f"{safe_name(row['sticker_name'])}_"
                f"{safe_name(row['position_name'])}_"
                f"scale_{safe_name(row['scale_factor'])}.png"
            )

            output_path = category_dir / filename

            save_example_figure(
                row=row,
                stickers=stickers,
                output_path=output_path,
                title_prefix=category,
            )

            index_rows.append(
                {
                    "category": category,
                    "rank": rank,
                    "output_path": str(output_path),
                    "identity": row["identity"],
                    "reference_image": row["reference_image"],
                    "probe_image": row["probe_image"],
                    "sticker_name": row["sticker_name"],
                    "position_name": row["position_name"],
                    "scale_factor": row["scale_factor"],
                    "clean_similarity": row.get("clean_similarity", None),
                    "stickered_similarity": row.get("stickered_similarity", None),
                    "similarity_drop": row.get("similarity_drop", None),
                    "stickered_face_detected": row.get("stickered_face_detected", None),
                    "stickered_reason": row.get("stickered_reason", None),
                    "attack_success": row.get("attack_success", None),
                }
            )

            print(f"Saved: {output_path}")

    index_df = pd.DataFrame(index_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    index_file = output_dir / "example_index.csv"
    index_df.to_csv(index_file, index=False)

    print()
    print(f"Saved example index to: {index_file}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-file",
        type=str,
        default=str(STICKER_GRID_ANALYZED_RESULTS_FILE),
        help=(
            "Prefer sticker_grid_facenet_analyzed.csv. "
            "If it does not exist yet, use sticker_grid_facenet.csv, "
            "but attack_success examples will be unavailable."
        ),
    )

    parser.add_argument("--sticker-dir", type=str, default=str(GENERATED_STICKERS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(TOP_STICKER_EXAMPLES_DIR))
    parser.add_argument("--top-n", type=int, default=10)

    args = parser.parse_args()

    results_file = Path(args.results_file)

    if not results_file.exists():
        fallback = STICKER_GRID_RESULTS_FILE

        if fallback.exists():
            print(
                f"Analyzed results file not found: {results_file}\n"
                f"Using fallback: {fallback}"
            )
            results_file = fallback
        else:
            raise FileNotFoundError(
                f"Could not find analyzed results file or fallback grid file."
            )

    save_examples(
        results_file=results_file,
        sticker_dir=Path(args.sticker_dir),
        output_dir=Path(args.output_dir),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
