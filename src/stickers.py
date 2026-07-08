from pathlib import Path

import numpy as np
from PIL import Image

RESAMPLING_LANCZOS = Image.Resampling.LANCZOS
RESAMPLING_BICUBIC = Image.Resampling.BICUBIC


def load_sticker(path: Path) -> Image.Image:
    """
    Loads one sticker as RGBA.

    Stickers should normally be transparent PNGs.
    """

    if not path.exists():
        raise FileNotFoundError(f"Sticker not found: {path}")

    return Image.open(path).convert("RGBA")


def load_sticker_library(sticker_dir: Path) -> dict[str, Image.Image]:
    """
    Loads all PNG stickers from a directory.

    Returns:
        {
            "heart": PIL.Image,
            "star": PIL.Image,
            ...
        }
    """

    if not sticker_dir.exists():
        raise FileNotFoundError(f"Sticker directory not found: {sticker_dir}")

    stickers = {}

    for path in sorted(sticker_dir.glob("*.png")):
        stickers[path.stem] = load_sticker(path)

    if not stickers:
        raise RuntimeError(f"No PNG stickers found in: {sticker_dir}")

    return stickers


def resize_sticker_to_width(sticker: Image.Image, target_width: float) -> Image.Image:
    """
    Resizes a sticker to a target width while preserving aspect ratio.
    """

    target_width = int(round(target_width))

    if target_width <= 0:
        raise ValueError("target_width must be positive.")

    aspect = sticker.height / sticker.width
    target_height = max(1, int(round(target_width * aspect)))

    return sticker.resize(
        (target_width, target_height),
        RESAMPLING_LANCZOS,
    )


def transform_sticker(
    sticker: Image.Image,
    target_width: float,
    rotation_degrees: float = 0.0,
    opacity: float = 1.0,
) -> Image.Image:
    """
    Applies size, rotation, and opacity to a sticker.

    target_width:
        Desired sticker width in pixels.

    rotation_degrees:
        Rotation angle in degrees.

    opacity:
        1.0 means fully opaque according to the sticker's alpha.
        0.5 means half as visible.
    """

    if not (0.0 <= opacity <= 1.0):
        raise ValueError("opacity must be between 0 and 1.")

    sticker = sticker.convert("RGBA")
    sticker = resize_sticker_to_width(sticker, target_width)

    if rotation_degrees != 0:
        sticker = sticker.rotate(
            rotation_degrees,
            expand=True,
            resample=RESAMPLING_BICUBIC,
        )

    if opacity < 1.0:
        r, g, b, a = sticker.split()
        a = a.point(lambda p: int(p * opacity))
        sticker.putalpha(a)

    return sticker


def apply_sticker(
    base_image: Image.Image,
    sticker: Image.Image,
    center_xy: tuple[float, float],
    target_width: float,
    rotation_degrees: float = 0.0,
    opacity: float = 1.0,
) -> Image.Image:
    """
    Alpha-composites a sticker onto an RGB image.

    Mathematically:

        I_out = (1 - M) * I_in + M * sticker

    where M is the sticker alpha mask.

    The sticker is centered at center_xy.
    """

    base = base_image.convert("RGBA")

    sticker = transform_sticker(
        sticker=sticker,
        target_width=target_width,
        rotation_degrees=rotation_degrees,
        opacity=opacity,
    )

    center_x, center_y = center_xy

    x0 = int(round(center_x - sticker.width / 2))
    y0 = int(round(center_y - sticker.height / 2))
    x1 = x0 + sticker.width
    y1 = y0 + sticker.height

    # Clip sticker to the base image bounds.
    paste_x0 = max(0, x0)
    paste_y0 = max(0, y0)
    paste_x1 = min(base.width, x1)
    paste_y1 = min(base.height, y1)

    # Sticker is fully outside image.
    if paste_x0 >= paste_x1 or paste_y0 >= paste_y1:
        return base.convert("RGB")

    crop_left = paste_x0 - x0
    crop_top = paste_y0 - y0
    crop_right = crop_left + (paste_x1 - paste_x0)
    crop_bottom = crop_top + (paste_y1 - paste_y0)

    sticker_crop = sticker.crop((crop_left, crop_top, crop_right, crop_bottom))

    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    layer.alpha_composite(sticker_crop, dest=(paste_x0, paste_y0))

    output = Image.alpha_composite(base, layer)

    return output.convert("RGB")


def sticker_visible_area_ratio(
    sticker: Image.Image,
    base_image: Image.Image,
    target_width: float,
) -> float:
    """
    Estimates how much of the image area the visible part of the sticker occupies.

    This is useful later as a simple naturalness/occlusion metric.
    """

    transformed = transform_sticker(
        sticker=sticker,
        target_width=target_width,
        rotation_degrees=0.0,
        opacity=1.0,
    )

    alpha = np.array(transformed.getchannel("A"))
    visible_pixels = np.count_nonzero(alpha > 0)

    image_pixels = base_image.width * base_image.height

    return float(visible_pixels / image_pixels)
