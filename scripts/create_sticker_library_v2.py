import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from src.config import (
    CURATED_STICKERS_V2_DIR,
    EXTERNAL_SELECTED_STICKERS_DIR,
    GENERATED_STICKERS_V2_DIR,
    STICKER_METADATA_GENERATED_V2_FILE,
    STICKER_METADATA_V2_FILE,
)

CANVAS = 512
OUTPUT_SIZE = 256
RNG_SEED = 42


def rgba(hex_color: str, alpha: int = 255):
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
        alpha,
    )


def new_canvas():
    return Image.new("RGBA", (CANVAS, CANVAS), (0, 0, 0, 0))


def downsample(img: Image.Image) -> Image.Image:
    return img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)


def save_sticker(img: Image.Image, name: str, metadata: dict):
    GENERATED_STICKERS_V2_DIR.mkdir(parents=True, exist_ok=True)

    output = downsample(img)
    path = GENERATED_STICKERS_V2_DIR / f"{name}.png"
    output.save(path)

    metadata[name] = {
        "name": name,
        **metadata[name],
        "source_type": "generated",
        "source": "procedural",
        "license": "own_generated",
        "file": str(path),
    }

    print(f"Generated: {path}")


def add_metadata(
    metadata: dict,
    name: str,
    category: str,
    motif: str,
    color_family: str,
    pattern_type: str,
):
    metadata[name] = {
        "category": category,
        "motif": motif,
        "color_family": color_family,
        "pattern_type": pattern_type,
    }


# -------------------------
# Noise baselines
# -------------------------


def create_checker_noise():
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    rng = random.Random(RNG_SEED)

    margin = 80
    cell = 32

    for y in range(margin, CANVAS - margin, cell):
        for x in range(margin, CANVAS - margin, cell):
            color = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
                235,
            )
            draw.rectangle((x, y, x + cell, y + cell), fill=color)

    return img


def create_random_noise():
    rng = np.random.default_rng(RNG_SEED)

    arr = rng.integers(0, 256, size=(CANVAS, CANVAS, 4), dtype=np.uint8)
    alpha = np.zeros((CANVAS, CANVAS), dtype=np.uint8)

    margin = 96
    alpha[margin:-margin, margin:-margin] = 225

    arr[:, :, 3] = alpha

    return Image.fromarray(arr, mode="RGBA")


def create_stripe_noise():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    colors = [
        rgba("#ff006e", 230),
        rgba("#3a86ff", 230),
        rgba("#ffbe0b", 230),
        rgba("#8338ec", 230),
    ]

    margin = 80
    stripe_width = 22

    for i, x in enumerate(range(margin - CANVAS, CANVAS + margin, stripe_width)):
        color = colors[i % len(colors)]
        draw.polygon(
            [
                (x, margin),
                (x + stripe_width, margin),
                (x + CANVAS // 2 + stripe_width, CANVAS - margin),
                (x + CANVAS // 2, CANVAS - margin),
            ],
            fill=color,
        )

    return img


def create_soft_noise_blob():
    rng = np.random.default_rng(RNG_SEED)

    arr = rng.normal(loc=140, scale=60, size=(CANVAS, CANVAS, 3))
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    alpha = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(alpha)
    draw.ellipse((90, 110, 420, 400), fill=210)
    alpha = alpha.filter(ImageFilter.GaussianBlur(18))

    rgba_arr = np.dstack([arr, np.array(alpha)])
    img = Image.fromarray(rgba_arr, mode="RGBA")

    return img


# -------------------------
# Occlusion baselines
# -------------------------


def create_black_bar():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (52, 190, 460, 320),
        radius=40,
        fill=(0, 0, 0, 235),
    )

    return img


def create_pixel_patch():
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    rng = random.Random(RNG_SEED + 1)

    margin = 100
    cell = 42

    for y in range(margin, CANVAS - margin, cell):
        for x in range(margin, CANVAS - margin, cell):
            v = rng.randint(80, 220)
            draw.rectangle(
                (x, y, x + cell + 2, y + cell + 2),
                fill=(v, v, v, 225),
            )

    return img


def create_skin_patch(color_hex: str):
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse(
        (95, 110, 420, 405),
        fill=rgba(color_hex, 220),
    )

    img = img.filter(ImageFilter.GaussianBlur(1.5))

    return img


def create_soft_circle(color_hex: str, alpha_value: int):
    img = new_canvas()

    mask = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((95, 95, 417, 417), fill=alpha_value)
    mask = mask.filter(ImageFilter.GaussianBlur(12))

    color = Image.new("RGBA", (CANVAS, CANVAS), rgba(color_hex, 255))
    color.putalpha(mask)

    return color


# -------------------------
# Makeup / skin-like
# -------------------------


def create_blush_patch():
    img = new_canvas()

    mask = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((90, 160, 425, 355), fill=115)
    mask = mask.filter(ImageFilter.GaussianBlur(32))

    color = Image.new("RGBA", (CANVAS, CANVAS), rgba("#ff7f9f", 255))
    color.putalpha(mask)

    return color


def create_freckles_cluster():
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    rng = random.Random(RNG_SEED)

    cx, cy = CANVAS // 2, CANVAS // 2

    for _ in range(55):
        x = int(rng.gauss(cx, 85))
        y = int(rng.gauss(cy, 55))
        r = rng.randint(3, 8)
        alpha = rng.randint(120, 210)

        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(95, 55, 35, alpha),
        )

    img = img.filter(ImageFilter.GaussianBlur(0.4))

    return img


def create_beauty_marks():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    marks = [
        (220, 210, 10),
        (285, 260, 7),
        (250, 310, 5),
        (320, 225, 4),
    ]

    for x, y, r in marks:
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            fill=(45, 25, 18, 220),
        )

    return img.filter(ImageFilter.GaussianBlur(0.3))


def create_glitter_patch():
    img = new_canvas()
    draw = ImageDraw.Draw(img)
    rng = random.Random(RNG_SEED)

    for _ in range(35):
        x = rng.randint(120, 395)
        y = rng.randint(140, 365)
        r = rng.randint(5, 13)
        alpha = rng.randint(120, 220)

        points = []
        for i in range(8):
            angle = i * math.pi / 4
            radius = r if i % 2 == 0 else r * 0.35
            points.append(
                (
                    x + radius * math.cos(angle),
                    y + radius * math.sin(angle),
                )
            )

        draw.polygon(points, fill=(255, 215, 80, alpha))

    return img.filter(ImageFilter.GaussianBlur(0.2))


def create_contour_stroke():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    for offset, alpha in [(0, 150), (12, 100), (24, 60)]:
        draw.arc(
            (120 + offset, 125, 470 + offset, 430),
            start=115,
            end=245,
            fill=(130, 75, 45, alpha),
            width=32,
        )

    img = img.filter(ImageFilter.GaussianBlur(4))

    return img


def create_highlight_patch():
    img = new_canvas()

    mask = Image.new("L", (CANVAS, CANVAS), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((110, 120, 390, 350), fill=95)
    mask = mask.filter(ImageFilter.GaussianBlur(35))

    color = Image.new("RGBA", (CANVAS, CANVAS), rgba("#ffe0a3", 255))
    color.putalpha(mask)

    return color


# -------------------------
# Face-part graphics
# -------------------------


def create_moustache():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse((85, 210, 275, 340), fill=(35, 20, 15, 235))
    draw.ellipse((237, 210, 427, 340), fill=(35, 20, 15, 235))
    draw.polygon(
        [(256, 265), (70, 310), (120, 360)],
        fill=(35, 20, 15, 235),
    )
    draw.polygon(
        [(256, 265), (442, 310), (392, 360)],
        fill=(35, 20, 15, 235),
    )

    # center cutout for a more moustache-like shape
    draw.ellipse((225, 245, 287, 320), fill=(0, 0, 0, 0))

    return img.filter(ImageFilter.GaussianBlur(0.8))


def create_eyebrow_pair():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.arc(
        (70, 145, 235, 260),
        start=200,
        end=340,
        fill=(40, 25, 18, 235),
        width=34,
    )
    draw.arc(
        (277, 145, 442, 260),
        start=200,
        end=340,
        fill=(40, 25, 18, 235),
        width=34,
    )

    return img.filter(ImageFilter.GaussianBlur(0.5))


def create_cartoon_eye():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse((110, 120, 402, 392), fill=(255, 255, 255, 240))
    draw.ellipse((190, 180, 322, 312), fill=(70, 140, 220, 245))
    draw.ellipse((225, 215, 287, 277), fill=(20, 20, 30, 245))
    draw.ellipse((235, 200, 260, 225), fill=(255, 255, 255, 220))

    draw.arc(
        (95, 110, 417, 402),
        start=180,
        end=360,
        fill=(30, 20, 20, 230),
        width=18,
    )

    return img


def create_lips():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse((90, 180, 260, 285), fill=(190, 30, 70, 230))
    draw.ellipse((252, 180, 422, 285), fill=(190, 30, 70, 230))
    draw.polygon(
        [(95, 245), (417, 245), (256, 355)],
        fill=(160, 20, 55, 230),
    )
    draw.line((130, 252, 382, 252), fill=(90, 10, 35, 200), width=8)

    return img.filter(ImageFilter.GaussianBlur(0.5))


# -------------------------
# Accessories
# -------------------------


def create_sunglasses():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (60, 175, 220, 315),
        radius=36,
        fill=(10, 10, 12, 240),
    )
    draw.rounded_rectangle(
        (292, 175, 452, 315),
        radius=36,
        fill=(10, 10, 12, 240),
    )
    draw.rectangle((220, 230, 292, 260), fill=(10, 10, 12, 240))

    # subtle highlight
    draw.line((90, 205, 160, 185), fill=(255, 255, 255, 80), width=10)
    draw.line((322, 205, 392, 185), fill=(255, 255, 255, 80), width=10)

    return img


def create_bandage():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (65, 190, 447, 320),
        radius=38,
        fill=(232, 188, 145, 235),
    )

    draw.rounded_rectangle(
        (198, 170, 314, 340),
        radius=28,
        fill=(220, 170, 125, 245),
    )

    for x in [105, 145, 370, 410]:
        for y in [230, 280]:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(180, 125, 95, 160))

    return img.filter(ImageFilter.GaussianBlur(0.3))


def create_nose_strip():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (155, 120, 357, 390),
        radius=55,
        fill=(235, 200, 170, 215),
    )

    draw.line((210, 160, 302, 350), fill=(255, 235, 210, 120), width=18)

    return img.filter(ImageFilter.GaussianBlur(0.8))


def create_face_gems():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    gems = [
        (256, 130, 18),
        (205, 200, 14),
        (307, 200, 14),
        (170, 285, 11),
        (342, 285, 11),
    ]

    for x, y, r in gems:
        points = [
            (x, y - r),
            (x + r, y),
            (x, y + r),
            (x - r, y),
        ]
        draw.polygon(points, fill=(180, 235, 255, 220))
        draw.line(points + [points[0]], fill=(60, 120, 160, 190), width=3)

    return img


def create_tattoo_line():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.arc(
        (120, 100, 420, 420),
        start=210,
        end=330,
        fill=(25, 25, 35, 210),
        width=20,
    )

    for i in range(6):
        x = 160 + i * 38
        y = 265 + int(25 * math.sin(i))
        draw.line((x, y, x + 20, y + 42), fill=(25, 25, 35, 190), width=10)

    return img.filter(ImageFilter.GaussianBlur(0.4))


def generate_procedural_stickers():
    metadata = {}

    specs = [
        # noise_baseline
        (
            "checker_noise",
            create_checker_noise,
            "noise_baseline",
            "checker",
            "mixed",
            "checker",
        ),
        (
            "random_noise",
            create_random_noise,
            "noise_baseline",
            "random_noise",
            "mixed",
            "noise",
        ),
        (
            "stripe_noise",
            create_stripe_noise,
            "noise_baseline",
            "stripes",
            "mixed",
            "stripes",
        ),
        (
            "soft_noise_blob",
            create_soft_noise_blob,
            "noise_baseline",
            "noise_blob",
            "mixed",
            "soft_noise",
        ),
        # occlusion_baseline
        ("black_bar", create_black_bar, "occlusion_baseline", "bar", "black", "solid"),
        (
            "pixel_patch",
            create_pixel_patch,
            "occlusion_baseline",
            "pixel_patch",
            "gray",
            "pixelated",
        ),
        (
            "skin_patch_light",
            lambda: create_skin_patch("#f0c7a1"),
            "occlusion_baseline",
            "skin_patch",
            "skin_light",
            "solid_soft",
        ),
        (
            "skin_patch_medium",
            lambda: create_skin_patch("#c88b61"),
            "occlusion_baseline",
            "skin_patch",
            "skin_medium",
            "solid_soft",
        ),
        (
            "skin_patch_dark",
            lambda: create_skin_patch("#7a4a32"),
            "occlusion_baseline",
            "skin_patch",
            "skin_dark",
            "solid_soft",
        ),
        (
            "soft_blue_circle",
            lambda: create_soft_circle("#4a90e2", 185),
            "occlusion_baseline",
            "soft_circle",
            "blue",
            "soft_blob",
        ),
        # makeup_skin
        (
            "blush_patch",
            create_blush_patch,
            "makeup_skin",
            "blush",
            "pink",
            "soft_blob",
        ),
        (
            "freckles_cluster",
            create_freckles_cluster,
            "makeup_skin",
            "freckles",
            "brown",
            "small_dots",
        ),
        (
            "beauty_marks",
            create_beauty_marks,
            "makeup_skin",
            "beauty_marks",
            "brown",
            "dots",
        ),
        (
            "glitter_patch",
            create_glitter_patch,
            "makeup_skin",
            "glitter",
            "gold",
            "sparkles",
        ),
        (
            "contour_stroke",
            create_contour_stroke,
            "makeup_skin",
            "contour",
            "brown",
            "stroke",
        ),
        (
            "highlight_patch",
            create_highlight_patch,
            "makeup_skin",
            "highlight",
            "gold",
            "soft_blob",
        ),
        # face_part_graphic
        (
            "moustache",
            create_moustache,
            "face_part_graphic",
            "moustache",
            "black",
            "graphic",
        ),
        (
            "eyebrow_pair",
            create_eyebrow_pair,
            "face_part_graphic",
            "eyebrow",
            "brown",
            "graphic",
        ),
        (
            "cartoon_eye",
            create_cartoon_eye,
            "face_part_graphic",
            "eye",
            "blue",
            "graphic",
        ),
        ("lips", create_lips, "face_part_graphic", "lips", "red", "graphic"),
        # accessory
        (
            "sunglasses",
            create_sunglasses,
            "accessory",
            "sunglasses",
            "black",
            "graphic",
        ),
        ("bandage", create_bandage, "accessory", "bandage", "skin_medium", "graphic"),
        (
            "nose_strip",
            create_nose_strip,
            "accessory",
            "nose_strip",
            "skin_light",
            "soft_solid",
        ),
        ("face_gems", create_face_gems, "accessory", "gems", "blue", "sparkles"),
        ("tattoo_line", create_tattoo_line, "accessory", "tattoo", "black", "line"),
    ]

    for name, fn, category, motif, color_family, pattern_type in specs:
        add_metadata(
            metadata=metadata,
            name=name,
            category=category,
            motif=motif,
            color_family=color_family,
            pattern_type=pattern_type,
        )

        img = fn()

        save_sticker(
            img=img,
            name=name,
            metadata=metadata,
        )

    STICKER_METADATA_GENERATED_V2_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STICKER_METADATA_GENERATED_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved generated metadata: {STICKER_METADATA_GENERATED_V2_FILE}")

    return metadata


def copy_pngs_to_curated(source_dir: Path, target_dir: Path):
    if not source_dir.exists():
        return []

    target_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    for path in sorted(source_dir.glob("*.png")):
        target_path = target_dir / path.name
        shutil.copy2(path, target_path)
        copied.append(target_path)

    return copied


def build_combined_metadata(generated_metadata: dict):
    """
    External stickers are copied into curated_v2, but their metadata should be
    added manually in sticker_metadata_external_v2.json or edited later.

    This function creates combined metadata for generated stickers only.
    You can extend it after selecting your 5 emoji stickers.
    """

    combined = {}

    for name, item in generated_metadata.items():
        item = dict(item)
        item["file"] = str(CURATED_STICKERS_V2_DIR / f"{name}.png")
        combined[name] = item

    with open(STICKER_METADATA_V2_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved combined metadata: {STICKER_METADATA_V2_FILE}")


def main():
    GENERATED_STICKERS_V2_DIR.mkdir(parents=True, exist_ok=True)
    CURATED_STICKERS_V2_DIR.mkdir(parents=True, exist_ok=True)

    generated_metadata = generate_procedural_stickers()

    copied_generated = copy_pngs_to_curated(
        GENERATED_STICKERS_V2_DIR,
        CURATED_STICKERS_V2_DIR,
    )

    copied_external = copy_pngs_to_curated(
        EXTERNAL_SELECTED_STICKERS_DIR,
        CURATED_STICKERS_V2_DIR,
    )

    build_combined_metadata(generated_metadata)

    print()
    print(f"Copied generated stickers to curated: {len(copied_generated)}")
    print(f"Copied external selected stickers to curated: {len(copied_external)}")
    print(f"Curated sticker directory: {CURATED_STICKERS_V2_DIR}")
    print()
    print("If you added external emoji stickers, add their metadata manually to:")
    print(STICKER_METADATA_V2_FILE)


if __name__ == "__main__":
    main()
