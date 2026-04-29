"""Graphic content — sharp edges, limited palettes.

This is the territory where palette codecs (PNG-8, GIF) and lossless
modes (WebP-LL, JXL-LL) win. The geometric variant draws a deterministic
set of overlapping shapes; the palette variant quantizes a smooth field
to a small number of discrete steps.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from bench.corpus.synthesis._common import (
    make_rng,
    register_kind,
    smooth_field,
)


def _palette(py_rng, n: int) -> list[tuple[int, int, int]]:
    """Pick `n` distinct-ish saturated colors deterministically."""
    out: list[tuple[int, int, int]] = []
    for _ in range(n):
        h = py_rng.random()
        s = py_rng.uniform(0.55, 0.95)
        v = py_rng.uniform(0.55, 0.95)
        out.append(_hsv_to_rgb(h, s, v))
    return out


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


@register_kind("graphic_geometric")
def graphic_geometric(
    *,
    seed: int,
    width: int,
    height: int,
    n_shapes: int = 60,
    palette_size: int = 6,
) -> Image.Image:
    """Overlapping rectangles, ellipses, and polygons from a small palette."""
    py_rng, _ = make_rng(seed)
    palette = _palette(py_rng, palette_size)

    img = Image.new("RGB", (width, height), palette[0])
    draw = ImageDraw.Draw(img)

    for _ in range(n_shapes):
        shape = py_rng.choice(("rect", "ellipse", "triangle"))
        color = py_rng.choice(palette[1:])
        x0 = py_rng.randrange(0, width)
        y0 = py_rng.randrange(0, height)
        w = py_rng.randrange(width // 16, max(width // 4, width // 16 + 1))
        h = py_rng.randrange(height // 16, max(height // 4, height // 16 + 1))
        x1 = min(x0 + w, width - 1)
        y1 = min(y0 + h, height - 1)

        if shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        else:
            mid_x = (x0 + x1) // 2
            draw.polygon([(x0, y1), (x1, y1), (mid_x, y0)], fill=color)

    return img


@register_kind("graphic_palette")
def graphic_palette(
    *,
    seed: int,
    width: int,
    height: int,
    levels: int = 6,
) -> Image.Image:
    """Smooth field quantized to `levels` discrete bands per channel.

    The result has soft contours (good for PNG-8 / GIF) without being
    purely geometric.
    """
    r = smooth_field(seed, height, width, alpha=1.4)
    g = smooth_field(seed + 7, height, width, alpha=1.4)
    b = smooth_field(seed + 13, height, width, alpha=1.4)
    stack = np.stack((r, g, b), axis=-1)
    quantized = np.digitize(stack, np.linspace(0.0, 1.0, levels + 1)[1:-1])
    rgb = (quantized * (255 // (levels - 1))).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")
