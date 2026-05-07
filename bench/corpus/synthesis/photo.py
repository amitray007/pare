"""Photographic content — smooth gradients, pink noise, white noise.

These are the "best case" inputs for transform coders (JPEG/AVIF/HEIC/JXL).
The pink-noise variant produces natural-looking textures without depending
on any external corpus or noise library; the white-noise variant gives
a hard upper bound on encoded size.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from bench.corpus.synthesis._common import (
    array_to_rgb,
    make_rng,
    register_kind,
    smooth_field,
)


@register_kind("photo_gradient")
def photo_gradient(
    *,
    seed: int,
    width: int,
    height: int,
    angle_deg: float = 35.0,
    stops: int = 4,
) -> Image.Image:
    """Multi-stop linear gradient with a faint pink-noise texture overlay."""
    py_rng, np_rng = make_rng(seed)

    angle = np.deg2rad(angle_deg)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    proj = (proj - proj.min()) / (proj.max() - proj.min() + 1e-9)

    palette = []
    for _ in range(stops):
        palette.append(np.array([py_rng.uniform(0.05, 0.95) for _ in range(3)]))
    palette_arr = np.stack(palette, axis=0)

    seg = proj * (stops - 1)
    lo = np.floor(seg).astype(np.int64)
    hi = np.clip(lo + 1, 0, stops - 1)
    t = (seg - lo)[..., None]

    base = palette_arr[lo] * (1 - t) + palette_arr[hi] * t

    texture = smooth_field(seed + 1, height, width, alpha=1.8) - 0.5
    base = base + (texture * 0.05)[..., None]
    base = np.clip(base, 0.0, 1.0)

    rgb = (base * 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")


@register_kind("photo_perlin")
def photo_perlin(
    *,
    seed: int,
    width: int,
    height: int,
    alpha: float = 1.6,
) -> Image.Image:
    """Pink-noise field treated as three slightly-offset color channels.

    Looks like a soft cloud / terrain photo. Independent channels keep the
    hue varying naturally instead of producing a pure grayscale image.
    """
    r = smooth_field(seed, height, width, alpha=alpha)
    g = smooth_field(seed + 17, height, width, alpha=alpha)
    b = smooth_field(seed + 31, height, width, alpha=alpha)
    return array_to_rgb((r, g, b))


@register_kind("photo_noise")
def photo_noise(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Uniform white noise — the worst case for any transform-based codec.

    Useful as a compression floor: if a format can't shrink this, it can't
    shrink anything. Also exposes encoder overhead (header sizes, etc.).
    """
    _, np_rng = make_rng(seed)
    arr = np_rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")
