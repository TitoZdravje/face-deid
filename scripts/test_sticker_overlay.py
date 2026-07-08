import argparse

import matplotlib.pyplot as plt

from src.celeba_io import load_celeba_image, load_landmarks, load_subset
from src.config import (
    CELEBA_IMAGES_DIR,
    DEFAULT_SUBSET_FILE,
    GENERATED_STICKERS_DIR,
    LANDMARKS_FILE,
    STICKER_OVERLAY_CHECKS_DIR,
)
from src.landmarks import get_candidate_positions, get_default_sticker_width
from src.stickers import apply_sticker, load_sticker_library


def draw_landmarks(ax, landmarks):
    for name, (x, y) in landmarks.items():
        ax.scatter(x, y, s=16)
        ax.text(x + 2, y + 2, name, fontsize=6)


def save_grid(images, titles, output_path, columns=4):
    rows = (len(images) + columns - 1) // columns

    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))

    if rows == 1:
        axes = [axes]
    axes_flat = []

    for row in axes:
        if isinstance(row, (list, tuple)):
            axes_flat.extend(row)
        else:
            try:
                axes_flat.extend(list(row))
            except TypeError:
                axes_flat.append(row)

    for ax in axes_flat:
        ax.axis("off")

    for ax, image, title in zip(axes_flat, images, titles):
        ax.imshow(image)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def create_position_check(
    probe_img,
    landmarks,
    stickers,
    identity,
    probe_name,
    output_dir,
):
    """
    Uses one sticker and places it at every candidate position.
    """

    positions = get_candidate_positions(landmarks, include_unreliable=True)
    target_width = get_default_sticker_width(landmarks, scale_factor=1.1)

    sticker_name = (
        "yellow_star" if "yellow_star" in stickers else sorted(stickers.keys())[0]
    )
    sticker = stickers[sticker_name]

    images = [probe_img]
    titles = [f"clean\n{probe_name}"]

    for position_name, center_xy in positions.items():
        stickered = apply_sticker(
            base_image=probe_img,
            sticker=sticker,
            center_xy=center_xy,
            target_width=target_width,
            rotation_degrees=0.0,
            opacity=1.0,
        )

        images.append(stickered)
        titles.append(position_name)

    output_path = output_dir / f"id_{identity}_positions_{sticker_name}.png"

    save_grid(
        images=images,
        titles=titles,
        output_path=output_path,
        columns=4,
    )


def create_sticker_check(
    probe_img,
    landmarks,
    stickers,
    identity,
    probe_name,
    output_dir,
):
    """
    Places every sticker at one default position.

    This helps visually compare sticker types.
    """

    positions = get_candidate_positions(landmarks, include_unreliable=True)
    center_xy = positions["forehead"]

    target_width = get_default_sticker_width(landmarks, scale_factor=1.2)

    images = [probe_img]
    titles = [f"clean\n{probe_name}"]

    for sticker_name, sticker in stickers.items():
        stickered = apply_sticker(
            base_image=probe_img,
            sticker=sticker,
            center_xy=center_xy,
            target_width=target_width,
            rotation_degrees=0.0,
            opacity=1.0,
        )

        images.append(stickered)
        titles.append(sticker_name)

    output_path = output_dir / f"id_{identity}_stickers_left_cheek.png"

    save_grid(
        images=images,
        titles=titles,
        output_path=output_path,
        columns=4,
    )


def create_landmark_debug_image(
    probe_img,
    landmarks,
    identity,
    probe_name,
    output_dir,
):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    ax.imshow(probe_img)
    ax.set_title(f"Landmarks\nID {identity}\n{probe_name}")
    ax.axis("off")

    draw_landmarks(ax, landmarks)

    output_path = output_dir / f"id_{identity}_landmarks.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-file", type=str, default=str(DEFAULT_SUBSET_FILE))
    parser.add_argument("--max-identities", type=int, default=5)
    args = parser.parse_args()

    subset = load_subset(args.subset_file)
    all_landmarks = load_landmarks(LANDMARKS_FILE)
    stickers = load_sticker_library(GENERATED_STICKERS_DIR)

    output_dir = STICKER_OVERLAY_CHECKS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = list(subset["pairs"].items())[: args.max_identities]

    for identity, pair in pairs:
        probe_name = pair["probe"]

        probe_img = load_celeba_image(CELEBA_IMAGES_DIR, probe_name)
        landmarks = all_landmarks[probe_name]

        create_landmark_debug_image(
            probe_img=probe_img,
            landmarks=landmarks,
            identity=identity,
            probe_name=probe_name,
            output_dir=output_dir,
        )

        create_position_check(
            probe_img=probe_img,
            landmarks=landmarks,
            stickers=stickers,
            identity=identity,
            probe_name=probe_name,
            output_dir=output_dir,
        )

        create_sticker_check(
            probe_img=probe_img,
            landmarks=landmarks,
            stickers=stickers,
            identity=identity,
            probe_name=probe_name,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
