import math
import random

from PIL import Image, ImageDraw

from src.config import GENERATED_STICKERS_DIR

CANVAS_SIZE = 128


def new_canvas():
    return Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))


def save(img, name):
    GENERATED_STICKERS_DIR.mkdir(parents=True, exist_ok=True)
    path = GENERATED_STICKERS_DIR / f"{name}.png"
    img.save(path)
    print(f"Saved: {path}")


def create_censor_bar():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (8, 46, 120, 82),
        radius=10,
        fill=(0, 0, 0, 235),
    )

    return img


def create_red_heart():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse((20, 18, 68, 66), fill=(220, 20, 80, 235))
    draw.ellipse((60, 18, 108, 66), fill=(220, 20, 80, 235))
    draw.polygon(
        [
            (18, 48),
            (110, 48),
            (64, 116),
        ],
        fill=(220, 20, 80, 235),
    )

    return img


def star_points(cx, cy, outer_r, inner_r, n=5):
    points = []

    for i in range(n * 2):
        angle = -math.pi / 2 + i * math.pi / n
        radius = outer_r if i % 2 == 0 else inner_r

        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)

        points.append((x, y))

    return points


def create_yellow_star():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    points = star_points(64, 64, 56, 24)
    draw.polygon(points, fill=(245, 200, 40, 235))

    return img


def create_blue_circle():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse(
        (18, 18, 110, 110),
        fill=(60, 150, 230, 210),
    )

    return img


def create_skin_colored_patch():
    """
    This is a useful baseline.

    If this works similarly to visible stickers, the effect may be mostly
    caused by occlusion rather than sticker semantics.
    """

    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse(
        (18, 18, 110, 110),
        fill=(210, 165, 130, 225),
    )

    return img


def create_checker_noise():
    """
    A deterministic noisy baseline.
    """

    img = new_canvas()
    draw = ImageDraw.Draw(img)
    rng = random.Random(42)

    cell = 8

    for y in range(16, 112, cell):
        for x in range(16, 112, cell):
            color = (
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(0, 255),
                230,
            )
            draw.rectangle((x, y, x + cell, y + cell), fill=color)

    return img


def create_simple_sunglasses():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        (12, 42, 54, 78),
        radius=10,
        fill=(10, 10, 10, 235),
    )
    draw.rounded_rectangle(
        (74, 42, 116, 78),
        radius=10,
        fill=(10, 10, 10, 235),
    )
    draw.rectangle(
        (54, 56, 74, 64),
        fill=(10, 10, 10, 235),
    )

    return img


def create_blush_pair():
    img = new_canvas()
    draw = ImageDraw.Draw(img)

    draw.ellipse((16, 42, 54, 82), fill=(255, 120, 150, 150))
    draw.ellipse((74, 42, 112, 82), fill=(255, 120, 150, 150))

    return img


def main():
    stickers = {
        "censor_bar": create_censor_bar(),
        "red_heart": create_red_heart(),
        "yellow_star": create_yellow_star(),
        "blue_circle": create_blue_circle(),
        "skin_patch": create_skin_colored_patch(),
        "checker_noise": create_checker_noise(),
        "simple_sunglasses": create_simple_sunglasses(),
        "blush_pair": create_blush_pair(),
    }

    for name, img in stickers.items():
        save(img, name)

    print()
    print(f"Generated {len(stickers)} stickers in {GENERATED_STICKERS_DIR}")


if __name__ == "__main__":
    main()
