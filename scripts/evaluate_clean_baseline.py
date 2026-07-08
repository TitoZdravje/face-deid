import argparse
import time

import pandas as pd
from tqdm import tqdm

from pathlib import Path

from src.celeba_io import load_celeba_image, load_subset
from src.config import (
    CELEBA_IMAGES_DIR,
    CLEAN_BASELINE_RESULTS_FILE,
    DEFAULT_SUBSET_FILE,
)
from src.face_model import FaceNetPytorchModel
from src.metrics import cosine_similarity


def evaluate_clean_baseline(subset_file, output_file, max_identities=None):
    subset = load_subset(subset_file)

    pairs = list(subset["pairs"].items())

    if max_identities is not None:
        pairs = pairs[:max_identities]

    model = FaceNetPytorchModel()

    rows = []

    start_time = time.time()

    for identity, pair in tqdm(pairs, desc="Evaluating clean pairs"):
        reference_name = pair["reference"]
        probe_name = pair["probe"]

        reference_image = load_celeba_image(CELEBA_IMAGES_DIR, reference_name)
        probe_image = load_celeba_image(CELEBA_IMAGES_DIR, probe_name)

        reference_result = model.get_embedding_result(reference_image)
        probe_result = model.get_embedding_result(probe_image)

        both_faces_detected = (
            reference_result.face_detected and probe_result.face_detected
        )

        similarity = None

        if both_faces_detected:
            similarity = cosine_similarity(
                reference_result.embedding,
                probe_result.embedding,
            )

        rows.append(
            {
                "identity": identity,
                "reference_image": reference_name,
                "probe_image": probe_name,
                "model": model.name,
                "reference_face_detected": reference_result.face_detected,
                "probe_face_detected": probe_result.face_detected,
                "reference_reason": reference_result.reason,
                "probe_reason": probe_result.reason,
                "both_faces_detected": both_faces_detected,
                "clean_similarity": similarity,
            }
        )

    elapsed = time.time() - start_time

    df = pd.DataFrame(rows)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    detected_df = df[df["both_faces_detected"]]

    print()
    print(f"Saved clean baseline results to: {output_file}")
    print(f"Total pairs: {len(df)}")
    print(f"Pairs with both faces detected: {len(detected_df)}")
    print(f"Pairs with detection failure: {len(df) - len(detected_df)}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

    if len(detected_df) > 0:
        print()
        print("Clean similarity statistics:")
        print(detected_df["clean_similarity"].describe())

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-file", type=str, default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument(
        "--output-file", type=str, default=str(CLEAN_BASELINE_RESULTS_FILE)
    )
    parser.add_argument("--max-identities", type=int, default=None)
    args = parser.parse_args()

    evaluate_clean_baseline(
        subset_file=Path(args.subset_file),
        output_file=Path(args.output_file),
        max_identities=args.max_identities,
    )


if __name__ == "__main__":
    main()
