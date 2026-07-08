import argparse

import matplotlib.pyplot as plt

from src.config import (
    CELEBA_IMAGES_DIR,
    LANDMARKS_FILE,
    SANITY_CHECKS_DIR,
    DEFAULT_SUBSET_FILE,
)
from src.celeba_io import load_celeba_image, load_landmarks, load_subset


def draw_landmarks(ax, landmarks):
    for name, (x, y) in landmarks.items():
        ax.scatter(x, y, s=20)
        ax.text(x + 2, y + 2, name, fontsize=7)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-file", type=str, default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument("--max-examples", type=int, default=10)
    args = parser.parse_args()

    subset = load_subset(args.subset_file)
    landmarks = load_landmarks(LANDMARKS_FILE)

    SANITY_CHECKS_DIR.mkdir(parents=True, exist_ok=True)

    pairs = list(subset["pairs"].items())[: args.max_examples]

    for idx, (identity, pair) in enumerate(pairs):
        reference_name = pair["reference"]
        probe_name = pair["probe"]

        reference_img = load_celeba_image(CELEBA_IMAGES_DIR, reference_name)
        probe_img = load_celeba_image(CELEBA_IMAGES_DIR, probe_name)

        probe_landmarks = landmarks[probe_name]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(reference_img)
        axes[0].set_title(f"Reference\nID {identity}\n{reference_name}")
        axes[0].axis("off")

        axes[1].imshow(probe_img)
        axes[1].set_title(f"Probe\nID {identity}\n{probe_name}")
        axes[1].axis("off")

        draw_landmarks(axes[1], probe_landmarks)

        output_path = SANITY_CHECKS_DIR / f"subset_check_{idx:03d}_id_{identity}.png"

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
