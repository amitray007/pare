"""Animated content — 4-frame variants exercising codec features.

Each pattern targets a different optimization axis used by APNG/animated
WebP/GIF encoders:

- translation:    sprite shifted +8 px per frame on a static gradient bg.
                  Tests dirty-rect coding (encoder should re-emit only
                  the changed strip, not full frames).
- fade:           single full-frame opacity ramp 0 → 0.5 → 1 → 0.5.
                  Tests blend-mode handling (alpha-blend across frames).
- sprite_static:  bouncing 32×32 ball over an unchanged background.
                  Tests `dispose=2` / `blend=over` semantics.
- redraw:         every frame is independent random content.
                  Worst case: forces encoder to emit full frames.

Synthesizers return `list[PIL.Image.Image]` — the conversion stage
encodes those frames into APNG / animated WebP / GIF as appropriate.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from bench.corpus.synthesis._common import (
    array_to_rgb,
    make_rng,
    register_kind,
    smooth_field,
)

FRAMES = 4


def _gradient_bg(seed: int, width: int, height: int) -> Image.Image:
    r = smooth_field(seed, height, width, alpha=1.6)
    g = smooth_field(seed + 17, height, width, alpha=1.6)
    b = smooth_field(seed + 31, height, width, alpha=1.6)
    return array_to_rgb((r, g, b))


@register_kind("animated_translation")
def animated_translation(
    *,
    seed: int,
    width: int,
    height: int,
    sprite_size: int = 32,
    step: int = 8,
) -> list[Image.Image]:
    """Sprite shifted +step px per frame; otherwise identical background."""
    py_rng, _ = make_rng(seed)
    bg = _gradient_bg(seed, width, height)
    color = (
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
    )
    y = max(0, height // 2 - sprite_size // 2)
    frames: list[Image.Image] = []
    for i in range(FRAMES):
        frame = bg.copy()
        draw = ImageDraw.Draw(frame)
        x = (i * step) % max(1, width - sprite_size)
        draw.rectangle([x, y, x + sprite_size, y + sprite_size], fill=color)
        frames.append(frame)
    return frames


@register_kind("animated_fade")
def animated_fade(
    *,
    seed: int,
    width: int,
    height: int,
) -> list[Image.Image]:
    """Full-frame opacity ramp over a fixed scene."""
    bg = _gradient_bg(seed, width, height)
    py_rng, _ = make_rng(seed)
    overlay_color = np.array([py_rng.uniform(0.2, 0.95) for _ in range(3)])

    bg_arr = np.asarray(bg, dtype=np.float64) / 255.0
    overlay = np.broadcast_to(overlay_color[None, None, :], bg_arr.shape)

    alphas = [0.0, 0.5, 1.0, 0.5]
    frames: list[Image.Image] = []
    for a in alphas:
        blended = bg_arr * (1 - a) + overlay * a
        frames.append(Image.fromarray((blended * 255).astype(np.uint8), "RGB"))
    return frames


@register_kind("animated_sprite_static")
def animated_sprite_static(
    *,
    seed: int,
    width: int,
    height: int,
    sprite_size: int = 32,
) -> list[Image.Image]:
    """Bouncing ball; everything outside the ball region is unchanged."""
    py_rng, _ = make_rng(seed)
    bg = _gradient_bg(seed, width, height)
    color = (
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
    )

    positions = [
        (width // 4, height // 3),
        (width // 2, height // 4),
        (3 * width // 4, height // 3),
        (width // 2, 2 * height // 3),
    ]
    frames: list[Image.Image] = []
    for cx, cy in positions:
        frame = bg.copy()
        draw = ImageDraw.Draw(frame)
        r = sprite_size // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        frames.append(frame)
    return frames


@register_kind("animated_redraw")
def animated_redraw(
    *,
    seed: int,
    width: int,
    height: int,
) -> list[Image.Image]:
    """Each frame is independent — worst case for inter-frame compression."""
    frames: list[Image.Image] = []
    for i in range(FRAMES):
        r = smooth_field(seed + i, height, width, alpha=1.5)
        g = smooth_field(seed + i + 100, height, width, alpha=1.5)
        b = smooth_field(seed + i + 200, height, width, alpha=1.5)
        frames.append(array_to_rgb((r, g, b)))
    return frames
