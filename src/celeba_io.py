import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image


def load_identity_map(identity_file: Path) -> dict[str, list[str]]:
    """
    Loads CelebA identity annotations.

    identity_CelebA.txt format:
        image_name identity_id

    Example:
        000001.jpg 2880

    Returns:
        {
            "2880": ["000001.jpg", ...],
            ...
        }
    """

    identity_to_images = defaultdict(list)

    with open(identity_file, "r", encoding="utf-8") as f:
        for line in f:
            image_name, identity = line.strip().split()
            identity_to_images[identity].append(image_name)

    return dict(identity_to_images)


def load_landmarks(landmarks_file: Path) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Loads CelebA aligned landmark annotations.

    list_landmarks_align_celeba.txt has:
        first line: number of images
        second line: header
        following lines:
            image_name lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y

    Returns:
        {
            "000001.jpg": {
                "left_eye": (x, y),
                "right_eye": (x, y),
                "nose": (x, y),
                "left_mouth": (x, y),
                "right_mouth": (x, y),
            }
        }
    """

    landmarks = {}

    with open(landmarks_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip first two lines: image count and header
    for line in lines[2:]:
        parts = line.strip().split()

        image_name = parts[0]
        values = list(map(int, parts[1:]))

        landmarks[image_name] = {
            "left_eye": (values[0], values[1]),
            "right_eye": (values[2], values[3]),
            "nose": (values[4], values[5]),
            "left_mouth": (values[6], values[7]),
            "right_mouth": (values[8], values[9]),
        }

    return landmarks


def create_identity_pair_subset(
    identity_file: Path,
    output_file: Path,
    n_identities: int = 50,
    seed: int = 42,
) -> dict:
    """
    Selects a fixed set of CelebA identities.

    For each identity, chooses:
        - one reference image
        - one probe image

    The result is saved as JSON so future runs use the exact same data.
    """

    identity_to_images = load_identity_map(identity_file)

    valid_identities = [
        identity for identity, images in identity_to_images.items() if len(images) >= 2
    ]

    if n_identities > len(valid_identities):
        raise ValueError(
            f"Requested {n_identities} identities, but only "
            f"{len(valid_identities)} identities have at least 2 images."
        )

    rng = random.Random(seed)
    selected_identities = rng.sample(valid_identities, n_identities)

    subset = {
        "metadata": {
            "dataset": "CelebA",
            "n_identities": n_identities,
            "seed": seed,
            "description": "Fixed identity-pair subset for privacy sticker experiments.",
        },
        "pairs": {},
    }

    for identity in selected_identities:
        images = sorted(identity_to_images[identity])
        reference, probe = rng.sample(images, 2)

        subset["pairs"][identity] = {
            "reference": reference,
            "probe": probe,
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=2)

    return subset


def load_subset(subset_file: Path) -> dict:
    with open(subset_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_celeba_image(images_dir: Path, image_name: str) -> Image.Image:
    image_path = images_dir / image_name

    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    return Image.open(image_path).convert("RGB")
