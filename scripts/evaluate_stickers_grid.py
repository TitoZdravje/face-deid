import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.celeba_io import load_celeba_image, load_landmarks, load_subset
from src.config import (
    CELEBA_IMAGES_DIR,
    DEFAULT_SUBSET_FILE,
    GENERATED_STICKERS_DIR,
    LANDMARKS_FILE,
    STICKER_GRID_RESULTS_FILE,
)
from src.face_model import FaceNetPytorchModel
from src.landmarks import get_candidate_positions, get_default_sticker_width
from src.metrics import cosine_similarity
from src.stickers import (
    apply_sticker,
    load_sticker_library,
    sticker_visible_area_ratio,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_string_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def evaluate_sticker_grid(
    subset_file: Path,
    output_file: Path,
    sticker_dir: Path,
    max_identities: int | None = None,
    scale_factors: list[float] | None = None,
    position_names: list[str] | None = None,
    min_clean_similarity: float | None = None,
):
    if scale_factors is None:
        scale_factors = [0.7, 1.0, 1.3, 1.6]

    if position_names is None:
        # Avoid duplicate between_eyes / eye_region for now.
        position_names = [
            "between_eyes",
            "nose",
            "mouth_center",
            "lower_face",
        ]

    subset = load_subset(subset_file)
    all_landmarks = load_landmarks(LANDMARKS_FILE)
    stickers = load_sticker_library(sticker_dir)

    pairs = list(subset["pairs"].items())

    if max_identities is not None:
        pairs = pairs[:max_identities]

    model = FaceNetPytorchModel()

    rows = []

    total_configs = (
        len(pairs) * len(stickers) * len(position_names) * len(scale_factors)
    )

    progress = tqdm(total=total_configs, desc="Evaluating sticker grid")

    for identity, pair in pairs:
        reference_name = pair["reference"]
        probe_name = pair["probe"]

        reference_image = load_celeba_image(CELEBA_IMAGES_DIR, reference_name)
        probe_image = load_celeba_image(CELEBA_IMAGES_DIR, probe_name)

        reference_result = model.get_embedding_result(reference_image)
        clean_probe_result = model.get_embedding_result(probe_image)

        clean_both_faces_detected = (
            reference_result.face_detected and clean_probe_result.face_detected
        )

        clean_similarity = None

        if clean_both_faces_detected:
            clean_similarity = cosine_similarity(
                reference_result.embedding,
                clean_probe_result.embedding,
            )

        clean_pair_usable = (
            clean_both_faces_detected
            and clean_similarity is not None
            and (
                min_clean_similarity is None or clean_similarity >= min_clean_similarity
            )
        )

        landmarks = all_landmarks[probe_name]
        candidate_positions = get_candidate_positions(
            landmarks,
            include_unreliable=False,
        )

        for sticker_name, sticker in stickers.items():
            for position_name in position_names:
                if position_name not in candidate_positions:
                    raise ValueError(
                        f"Unknown position '{position_name}'. "
                        f"Available positions: {list(candidate_positions.keys())}"
                    )

                center_xy = candidate_positions[position_name]

                for scale_factor in scale_factors:
                    target_width = get_default_sticker_width(
                        landmarks,
                        scale_factor=scale_factor,
                    )

                    stickered_face_detected = False
                    stickered_reason = "not_evaluated"
                    stickered_similarity = None
                    similarity_drop = None

                    if clean_both_faces_detected:
                        stickered_image = apply_sticker(
                            base_image=probe_image,
                            sticker=sticker,
                            center_xy=center_xy,
                            target_width=target_width,
                            rotation_degrees=0.0,
                            opacity=1.0,
                        )

                        stickered_result = model.get_embedding_result(stickered_image)

                        stickered_face_detected = stickered_result.face_detected
                        stickered_reason = stickered_result.reason

                        if stickered_result.face_detected:
                            stickered_similarity = cosine_similarity(
                                reference_result.embedding,
                                stickered_result.embedding,
                            )

                            similarity_drop = clean_similarity - stickered_similarity
                        else:
                            stickered_similarity = None
                            similarity_drop = None

                    area_ratio = sticker_visible_area_ratio(
                        sticker=sticker,
                        base_image=probe_image,
                        target_width=target_width,
                    )

                    rows.append(
                        {
                            "identity": identity,
                            "reference_image": reference_name,
                            "probe_image": probe_name,
                            "model": model.name,
                            "clean_reference_face_detected": reference_result.face_detected,
                            "clean_probe_face_detected": clean_probe_result.face_detected,
                            "clean_both_faces_detected": clean_both_faces_detected,
                            "clean_similarity": clean_similarity,
                            "clean_pair_usable": clean_pair_usable,
                            "sticker_name": sticker_name,
                            "position_name": position_name,
                            "center_x": center_xy[0],
                            "center_y": center_xy[1],
                            "scale_factor": scale_factor,
                            "target_width_px": target_width,
                            "rotation_degrees": 0.0,
                            "opacity": 1.0,
                            "sticker_area_ratio": area_ratio,
                            "stickered_face_detected": stickered_face_detected,
                            "stickered_reason": stickered_reason,
                            "stickered_similarity": stickered_similarity,
                            "similarity_drop": similarity_drop,
                        }
                    )

                    progress.update(1)

    progress.close()

    df = pd.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print()
    print(f"Saved sticker grid results to: {output_file}")
    print(f"Rows: {len(df)}")
    print(f"Identities: {len(pairs)}")
    print(f"Stickers: {len(stickers)}")
    print(f"Positions: {position_names}")
    print(f"Scale factors: {scale_factors}")

    valid_df = df[
        df["clean_both_faces_detected"]
        & df["stickered_face_detected"]
        & df["similarity_drop"].notna()
    ]

    print()
    print(f"Rows with valid clean and stickered embeddings: {len(valid_df)}")

    if len(valid_df) > 0:
        print()
        print("Similarity drop summary:")
        print(valid_df["similarity_drop"].describe())

        print()
        print("Top average drops by sticker:")
        print(
            valid_df.groupby("sticker_name")["similarity_drop"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-file", type=str, default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument(
        "--output-file", type=str, default=str(STICKER_GRID_RESULTS_FILE)
    )
    parser.add_argument("--sticker-dir", type=str, default=str(GENERATED_STICKERS_DIR))
    parser.add_argument("--max-identities", type=int, default=None)

    parser.add_argument(
        "--scale-factors",
        type=str,
        default="0.7,1.0,1.3,1.6",
        help="Comma-separated scale factors, relative to inter-eye distance.",
    )

    parser.add_argument(
        "--positions",
        type=str,
        default="between_eyes,nose,mouth_center,lower_face",
        help="Comma-separated candidate position names.",
    )

    parser.add_argument(
        "--min-clean-similarity",
        type=float,
        default=None,
        help=(
            "Optional. Marks clean_pair_usable=False when clean similarity is below this. "
            "Rows are still saved."
        ),
    )

    args = parser.parse_args()

    evaluate_sticker_grid(
        subset_file=Path(args.subset_file),
        output_file=Path(args.output_file),
        sticker_dir=Path(args.sticker_dir),
        max_identities=args.max_identities,
        scale_factors=parse_float_list(args.scale_factors),
        position_names=parse_string_list(args.positions),
        min_clean_similarity=args.min_clean_similarity,
    )


if __name__ == "__main__":
    main()
