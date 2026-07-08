import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import (
    STICKER_GRID_ANALYSIS_JSON_FILE,
    STICKER_GRID_ANALYSIS_SUMMARY_FILE,
    STICKER_GRID_ANALYZED_RESULTS_FILE,
    STICKER_GRID_RESULTS_FILE,
    VERIFICATION_THRESHOLD_FILE,
)


def load_threshold(
    threshold_file: Path,
    threshold_name: str,
    threshold_value: float | None,
) -> tuple[float, str, dict]:
    with open(threshold_file, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)

    if threshold_value is not None:
        return threshold_value, "manual", threshold_data

    thresholds = threshold_data["thresholds"]

    if threshold_name not in thresholds:
        raise ValueError(
            f"Unknown threshold name '{threshold_name}'. "
            f"Available: {list(thresholds.keys())}"
        )

    return (
        float(thresholds[threshold_name]["threshold"]),
        threshold_name,
        threshold_data,
    )


def add_attack_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()

    df["clean_recognized"] = (
        df["clean_both_faces_detected"]
        & df["clean_similarity"].notna()
        & (df["clean_similarity"] >= threshold)
    )

    df["stickered_recognized"] = (
        df["stickered_face_detected"]
        & df["stickered_similarity"].notna()
        & (df["stickered_similarity"] >= threshold)
    )

    df["detection_failure_success"] = df["clean_recognized"] & (
        ~df["stickered_face_detected"]
    )

    df["embedding_failure_success"] = (
        df["clean_recognized"]
        & df["stickered_face_detected"]
        & df["stickered_similarity"].notna()
        & (df["stickered_similarity"] < threshold)
    )

    df["attack_success"] = (
        df["detection_failure_success"] | df["embedding_failure_success"]
    )

    df["recognized_to_unrecognized"] = df["attack_success"]

    return df


def summarize_group(group: pd.DataFrame) -> pd.Series:
    clean_recognized = group[group["clean_recognized"]]

    n_rows = len(group)
    n_clean_recognized = len(clean_recognized)

    if n_clean_recognized > 0:
        attack_success_rate = clean_recognized["attack_success"].mean()
        detection_failure_rate = clean_recognized["detection_failure_success"].mean()
        embedding_failure_rate = clean_recognized["embedding_failure_success"].mean()
        stickered_recognized_rate = clean_recognized["stickered_recognized"].mean()
    else:
        attack_success_rate = None
        detection_failure_rate = None
        embedding_failure_rate = None
        stickered_recognized_rate = None

    valid_similarity_drop = group[
        group["clean_both_faces_detected"]
        & group["stickered_face_detected"]
        & group["similarity_drop"].notna()
    ]["similarity_drop"]

    return pd.Series(
        {
            "n_rows": n_rows,
            "n_clean_recognized_rows": n_clean_recognized,
            "attack_success_rate": attack_success_rate,
            "detection_failure_success_rate": detection_failure_rate,
            "embedding_failure_success_rate": embedding_failure_rate,
            "stickered_recognized_rate_after_clean_recognition": stickered_recognized_rate,
            "mean_similarity_drop": (
                valid_similarity_drop.mean() if len(valid_similarity_drop) > 0 else None
            ),
            "median_similarity_drop": (
                valid_similarity_drop.median()
                if len(valid_similarity_drop) > 0
                else None
            ),
            "mean_clean_similarity": group["clean_similarity"].mean(),
            "mean_stickered_similarity": group["stickered_similarity"].mean(),
            "mean_sticker_area_ratio": group["sticker_area_ratio"].mean(),
        }
    )


def build_summary_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summaries = {
        "by_sticker": (
            df.groupby("sticker_name")
            .apply(summarize_group)
            .reset_index()
            .sort_values("attack_success_rate", ascending=False)
        ),
        "by_position": (
            df.groupby("position_name")
            .apply(summarize_group)
            .reset_index()
            .sort_values("attack_success_rate", ascending=False)
        ),
        "by_scale": (
            df.groupby("scale_factor")
            .apply(summarize_group)
            .reset_index()
            .sort_values("scale_factor")
        ),
        "by_sticker_position": (
            df.groupby(["sticker_name", "position_name"])
            .apply(summarize_group)
            .reset_index()
            .sort_values("attack_success_rate", ascending=False)
        ),
        "by_sticker_scale": (
            df.groupby(["sticker_name", "scale_factor"])
            .apply(summarize_group)
            .reset_index()
            .sort_values("attack_success_rate", ascending=False)
        ),
    }

    return summaries


def save_summary_csv(summary_tables: dict[str, pd.DataFrame], output_file: Path):
    """
    Saves one flat CSV with a column telling which summary type each row belongs to.
    """

    parts = []

    for summary_name, table in summary_tables.items():
        temp = table.copy()
        temp.insert(0, "summary_type", summary_name)
        parts.append(temp)

    combined = pd.concat(parts, ignore_index=True, sort=False)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_file, index=False)


def save_summary_json(
    df: pd.DataFrame,
    summary_tables: dict[str, pd.DataFrame],
    threshold: float,
    threshold_name: str,
    threshold_file: Path,
    output_file: Path,
):
    clean_recognized_rows = df[df["clean_recognized"]]

    if len(clean_recognized_rows) > 0:
        overall_attack_success_rate = float(
            clean_recognized_rows["attack_success"].mean()
        )
        detection_failure_rate = float(
            clean_recognized_rows["detection_failure_success"].mean()
        )
        embedding_failure_rate = float(
            clean_recognized_rows["embedding_failure_success"].mean()
        )
    else:
        overall_attack_success_rate = None
        detection_failure_rate = None
        embedding_failure_rate = None

    output = {
        "threshold": float(threshold),
        "threshold_name": threshold_name,
        "threshold_file": str(threshold_file),
        "n_rows": int(len(df)),
        "n_clean_recognized_rows": int(len(clean_recognized_rows)),
        "overall_attack_success_rate": overall_attack_success_rate,
        "overall_detection_failure_success_rate": detection_failure_rate,
        "overall_embedding_failure_success_rate": embedding_failure_rate,
        "mean_similarity_drop_detected_only": (
            float(
                df[
                    df["clean_both_faces_detected"]
                    & df["stickered_face_detected"]
                    & df["similarity_drop"].notna()
                ]["similarity_drop"].mean()
            )
        ),
        "best_stickers_by_attack_success": summary_tables["by_sticker"]
        .head(10)
        .to_dict(orient="records"),
        "best_positions_by_attack_success": summary_tables["by_position"]
        .head(10)
        .to_dict(orient="records"),
        "scale_summary": summary_tables["by_scale"].to_dict(orient="records"),
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def analyze_sticker_grid_with_threshold(
    sticker_grid_file: Path,
    threshold_file: Path,
    analyzed_output_file: Path,
    summary_output_file: Path,
    summary_json_file: Path,
    threshold_name: str,
    threshold_value: float | None,
):
    threshold, used_threshold_name, _ = load_threshold(
        threshold_file=threshold_file,
        threshold_name=threshold_name,
        threshold_value=threshold_value,
    )

    df = pd.read_csv(sticker_grid_file)

    df = add_attack_columns(df, threshold=threshold)

    analyzed_output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(analyzed_output_file, index=False)

    summary_tables = build_summary_tables(df)

    save_summary_csv(
        summary_tables=summary_tables,
        output_file=summary_output_file,
    )

    save_summary_json(
        df=df,
        summary_tables=summary_tables,
        threshold=threshold,
        threshold_name=used_threshold_name,
        threshold_file=threshold_file,
        output_file=summary_json_file,
    )

    clean_recognized_rows = df[df["clean_recognized"]]

    print()
    print(f"Using threshold: {threshold:.4f} ({used_threshold_name})")
    print(f"Saved analyzed rows to: {analyzed_output_file}")
    print(f"Saved summary CSV to: {summary_output_file}")
    print(f"Saved summary JSON to: {summary_json_file}")
    print()
    print(f"Total rows: {len(df)}")
    print(f"Clean-recognized rows: {len(clean_recognized_rows)}")

    if len(clean_recognized_rows) > 0:
        print()
        print("Overall:")
        print(
            f"Attack success rate: {clean_recognized_rows['attack_success'].mean():.4f}"
        )
        print(
            f"Detection-failure success rate: {clean_recognized_rows['detection_failure_success'].mean():.4f}"
        )
        print(
            f"Embedding-failure success rate: {clean_recognized_rows['embedding_failure_success'].mean():.4f}"
        )

    print()
    print("Best stickers by attack success:")
    print(
        summary_tables["by_sticker"][
            [
                "sticker_name",
                "n_clean_recognized_rows",
                "attack_success_rate",
                "mean_similarity_drop",
                "mean_sticker_area_ratio",
            ]
        ].head(10)
    )

    print()
    print("Best positions by attack success:")
    print(
        summary_tables["by_position"][
            [
                "position_name",
                "n_clean_recognized_rows",
                "attack_success_rate",
                "mean_similarity_drop",
                "mean_sticker_area_ratio",
            ]
        ]
    )

    print()
    print("Scale summary:")
    print(
        summary_tables["by_scale"][
            [
                "scale_factor",
                "n_clean_recognized_rows",
                "attack_success_rate",
                "mean_similarity_drop",
                "mean_sticker_area_ratio",
            ]
        ]
    )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sticker-grid-file", type=str, default=str(STICKER_GRID_RESULTS_FILE)
    )
    parser.add_argument(
        "--threshold-file", type=str, default=str(VERIFICATION_THRESHOLD_FILE)
    )
    parser.add_argument(
        "--analyzed-output-file",
        type=str,
        default=str(STICKER_GRID_ANALYZED_RESULTS_FILE),
    )
    parser.add_argument(
        "--summary-output-file",
        type=str,
        default=str(STICKER_GRID_ANALYSIS_SUMMARY_FILE),
    )
    parser.add_argument(
        "--summary-json-file", type=str, default=str(STICKER_GRID_ANALYSIS_JSON_FILE)
    )

    parser.add_argument(
        "--threshold-name",
        type=str,
        default="best_accuracy",
        help=(
            "Threshold name from verification_threshold_facenet.json. "
            "Examples: best_accuracy, eer, far_10_percent, far_1_percent, far_0_percent."
        ),
    )

    parser.add_argument(
        "--threshold-value",
        type=float,
        default=None,
        help="Optional manual threshold. Overrides --threshold-name.",
    )

    args = parser.parse_args()

    analyze_sticker_grid_with_threshold(
        sticker_grid_file=Path(args.sticker_grid_file),
        threshold_file=Path(args.threshold_file),
        analyzed_output_file=Path(args.analyzed_output_file),
        summary_output_file=Path(args.summary_output_file),
        summary_json_file=Path(args.summary_json_file),
        threshold_name=args.threshold_name,
        threshold_value=args.threshold_value,
    )


if __name__ == "__main__":
    main()
