import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.celeba_io import load_celeba_image, load_subset
from src.config import (
    CELEBA_IMAGES_DIR,
    DEFAULT_SUBSET_FILE,
    VERIFICATION_PAIRS_RESULTS_FILE,
    VERIFICATION_THRESHOLD_FILE,
)
from src.face_model import FaceNetPytorchModel
from src.metrics import cosine_similarity


def summarize(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
    }


def compute_threshold_metrics(
    positive_similarities: np.ndarray,
    negative_similarities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Positive pair:
        same identity

    Negative pair:
        different identity

    Decision rule:
        similarity >= threshold => same identity
    """

    positive_similarities = np.asarray(positive_similarities, dtype=float)
    negative_similarities = np.asarray(negative_similarities, dtype=float)

    true_positive = np.sum(positive_similarities >= threshold)
    false_negative = np.sum(positive_similarities < threshold)

    false_positive = np.sum(negative_similarities >= threshold)
    true_negative = np.sum(negative_similarities < threshold)

    n_positive = len(positive_similarities)
    n_negative = len(negative_similarities)
    n_total = n_positive + n_negative

    accuracy = (true_positive + true_negative) / n_total

    tpr = true_positive / n_positive if n_positive > 0 else 0.0
    frr = false_negative / n_positive if n_positive > 0 else 0.0

    # FAR = false accept rate = negative pairs incorrectly accepted as same identity.
    far = false_positive / n_negative if n_negative > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "tpr": float(tpr),
        "frr": float(frr),
        "far": float(far),
        "true_positive": int(true_positive),
        "false_negative": int(false_negative),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
    }


def candidate_thresholds(
    positive_similarities: np.ndarray,
    negative_similarities: np.ndarray,
) -> np.ndarray:
    """
    Creates threshold candidates from observed similarities.

    We also include values just below min and just above max so that
    all-negative and all-positive decisions are possible.
    """

    all_values = np.concatenate([positive_similarities, negative_similarities])
    unique_values = np.unique(all_values)

    eps = 1e-6

    candidates = []

    candidates.append(float(np.min(unique_values) - eps))
    candidates.extend(float(x) for x in unique_values)
    candidates.append(float(np.max(unique_values) + eps))

    return np.array(candidates, dtype=float)


def find_best_accuracy_threshold(
    positive_similarities: np.ndarray,
    negative_similarities: np.ndarray,
) -> dict:
    best = None

    for threshold in candidate_thresholds(positive_similarities, negative_similarities):
        metrics = compute_threshold_metrics(
            positive_similarities,
            negative_similarities,
            threshold,
        )

        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = metrics

    return best


def find_eer_threshold(
    positive_similarities: np.ndarray,
    negative_similarities: np.ndarray,
) -> dict:
    """
    Finds threshold where FAR and FRR are closest.

    EER is approximated because we scan observed similarity thresholds.
    """

    best = None
    best_gap = None

    for threshold in candidate_thresholds(positive_similarities, negative_similarities):
        metrics = compute_threshold_metrics(
            positive_similarities,
            negative_similarities,
            threshold,
        )

        gap = abs(metrics["far"] - metrics["frr"])

        if best is None or gap < best_gap:
            best = metrics
            best_gap = gap

    best["eer_approx"] = float((best["far"] + best["frr"]) / 2.0)

    return best


def find_far_limited_threshold(
    positive_similarities: np.ndarray,
    negative_similarities: np.ndarray,
    target_far: float,
) -> dict:
    """
    Finds the lowest threshold whose FAR is <= target_far.

    Lowest threshold is chosen because among thresholds satisfying the
    FAR constraint, it usually preserves the highest TPR.
    """

    valid = []

    for threshold in candidate_thresholds(positive_similarities, negative_similarities):
        metrics = compute_threshold_metrics(
            positive_similarities,
            negative_similarities,
            threshold,
        )

        if metrics["far"] <= target_far:
            valid.append(metrics)

    if not valid:
        raise RuntimeError(f"No threshold found for FAR <= {target_far}")

    valid = sorted(valid, key=lambda item: item["threshold"])

    result = valid[0]
    result["target_far"] = float(target_far)

    return result


def compute_embeddings_for_subset(subset_file: Path, max_identities: int | None):
    subset = load_subset(subset_file)
    pairs = list(subset["pairs"].items())

    if max_identities is not None:
        pairs = pairs[:max_identities]

    model = FaceNetPytorchModel()

    embeddings = {}

    for identity, pair in tqdm(pairs, desc="Computing embeddings"):
        reference_name = pair["reference"]
        probe_name = pair["probe"]

        reference_image = load_celeba_image(CELEBA_IMAGES_DIR, reference_name)
        probe_image = load_celeba_image(CELEBA_IMAGES_DIR, probe_name)

        reference_result = model.get_embedding_result(reference_image)
        probe_result = model.get_embedding_result(probe_image)

        embeddings[identity] = {
            "reference_image": reference_name,
            "probe_image": probe_name,
            "reference_embedding": reference_result.embedding,
            "probe_embedding": probe_result.embedding,
            "reference_face_detected": reference_result.face_detected,
            "probe_face_detected": probe_result.face_detected,
            "reference_reason": reference_result.reason,
            "probe_reason": probe_result.reason,
        }

    return model.name, embeddings


def create_pair_rows(model_name: str, embeddings: dict) -> list[dict]:
    rows = []

    identities = list(embeddings.keys())

    # Positive pairs: reference(A) vs probe(A)
    for identity in identities:
        item = embeddings[identity]

        both_detected = item["reference_face_detected"] and item["probe_face_detected"]

        similarity = None

        if both_detected:
            similarity = cosine_similarity(
                item["reference_embedding"],
                item["probe_embedding"],
            )

        rows.append(
            {
                "pair_type": "positive",
                "reference_identity": identity,
                "probe_identity": identity,
                "reference_image": item["reference_image"],
                "probe_image": item["probe_image"],
                "model": model_name,
                "reference_face_detected": item["reference_face_detected"],
                "probe_face_detected": item["probe_face_detected"],
                "reference_reason": item["reference_reason"],
                "probe_reason": item["probe_reason"],
                "both_faces_detected": both_detected,
                "similarity": similarity,
            }
        )

    # Negative pairs: reference(A) vs probe(B), A != B
    for reference_identity in tqdm(identities, desc="Creating negative pairs"):
        reference_item = embeddings[reference_identity]

        for probe_identity in identities:
            if reference_identity == probe_identity:
                continue

            probe_item = embeddings[probe_identity]

            both_detected = (
                reference_item["reference_face_detected"]
                and probe_item["probe_face_detected"]
            )

            similarity = None

            if both_detected:
                similarity = cosine_similarity(
                    reference_item["reference_embedding"],
                    probe_item["probe_embedding"],
                )

            rows.append(
                {
                    "pair_type": "negative",
                    "reference_identity": reference_identity,
                    "probe_identity": probe_identity,
                    "reference_image": reference_item["reference_image"],
                    "probe_image": probe_item["probe_image"],
                    "model": model_name,
                    "reference_face_detected": reference_item[
                        "reference_face_detected"
                    ],
                    "probe_face_detected": probe_item["probe_face_detected"],
                    "reference_reason": reference_item["reference_reason"],
                    "probe_reason": probe_item["probe_reason"],
                    "both_faces_detected": both_detected,
                    "similarity": similarity,
                }
            )

    return rows


def evaluate_verification_threshold(
    subset_file: Path,
    pairs_output_file: Path,
    threshold_output_file: Path,
    max_identities: int | None = None,
):
    model_name, embeddings = compute_embeddings_for_subset(
        subset_file=subset_file,
        max_identities=max_identities,
    )

    rows = create_pair_rows(model_name=model_name, embeddings=embeddings)

    df = pd.DataFrame(rows)

    pairs_output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(pairs_output_file, index=False)

    valid_df = df[df["both_faces_detected"] & df["similarity"].notna()].copy()

    positive = valid_df[valid_df["pair_type"] == "positive"]["similarity"].to_numpy()
    negative = valid_df[valid_df["pair_type"] == "negative"]["similarity"].to_numpy()

    if len(positive) == 0:
        raise RuntimeError("No valid positive pairs.")
    if len(negative) == 0:
        raise RuntimeError("No valid negative pairs.")

    thresholds = {
        "best_accuracy": find_best_accuracy_threshold(positive, negative),
        "eer": find_eer_threshold(positive, negative),
        "far_10_percent": find_far_limited_threshold(
            positive, negative, target_far=0.10
        ),
        "far_1_percent": find_far_limited_threshold(
            positive, negative, target_far=0.01
        ),
        "far_0_percent": find_far_limited_threshold(
            positive, negative, target_far=0.00
        ),
    }

    output = {
        "model": model_name,
        "subset_file": str(subset_file),
        "pairs_file": str(pairs_output_file),
        "n_total_pairs": int(len(df)),
        "n_valid_pairs": int(len(valid_df)),
        "n_positive": int(len(positive)),
        "n_negative": int(len(negative)),
        "positive_similarity_summary": summarize(positive),
        "negative_similarity_summary": summarize(negative),
        "thresholds": thresholds,
        "recommended_threshold_name": "best_accuracy",
        "notes": (
            "Decision rule: cosine_similarity >= threshold means same identity. "
            "Use best_accuracy for early experiments, and FAR-limited thresholds "
            "for stricter security-style evaluation."
        ),
    }

    threshold_output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(threshold_output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved verification pairs to: {pairs_output_file}")
    print(f"Saved threshold data to: {threshold_output_file}")
    print()
    print("Positive similarity summary:")
    print(pd.Series(positive).describe())
    print()
    print("Negative similarity summary:")
    print(pd.Series(negative).describe())
    print()
    print("Thresholds:")

    for name, metrics in thresholds.items():
        print(
            f"{name:>15}: "
            f"threshold={metrics['threshold']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"TPR={metrics['tpr']:.4f}, "
            f"FAR={metrics['far']:.4f}, "
            f"FRR={metrics['frr']:.4f}"
        )

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-file", type=str, default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument(
        "--pairs-output-file", type=str, default=str(VERIFICATION_PAIRS_RESULTS_FILE)
    )
    parser.add_argument(
        "--threshold-output-file", type=str, default=str(VERIFICATION_THRESHOLD_FILE)
    )
    parser.add_argument("--max-identities", type=int, default=None)
    args = parser.parse_args()

    evaluate_verification_threshold(
        subset_file=Path(args.subset_file),
        pairs_output_file=Path(args.pairs_output_file),
        threshold_output_file=Path(args.threshold_output_file),
        max_identities=args.max_identities,
    )


if __name__ == "__main__":
    main()
