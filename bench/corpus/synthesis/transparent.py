"""Transparent content — RGBA composites with alpha gradients.

Exercises the alpha-aware paths in PNG-32, WebP, AVIF, JXL, HEIC. GIF and
JPEG ignore alpha (they'll either matte to a background or strip it),
which is itself useful test signal for the optimizer.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from bench.corpus.synthesis._common import (
    make_rng,
    register_kind,
    smooth_field,
)


@register_kind("transparent_overlay")
def transparent_overlay(
    *,
    seed: int,
    width: int,
    height: int,
    n_blobs: int = 12,
) -> Image.Image:
    """Soft colored blobs over a transparent background.

    Alpha is built from a smooth field so the edges are not binary —
    this exercises the encoder's alpha-quality path, not just the
    one-bit transparency case.
    """
    py_rng, _ = make_rng(seed)

    rgb = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(n_blobs):
        cx = py_rng.uniform(0.0, 1.0) * width
        cy = py_rng.uniform(0.0, 1.0) * height
        radius = py_rng.uniform(0.08, 0.25) * min(width, height)
        color = np.array([py_rng.uniform(0.2, 1.0) for _ in range(3)])

        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        d = np.hypot(xx - cx, yy - cy)
        falloff = np.clip(1.0 - d / radius, 0.0, 1.0) ** 2
        rgb += falloff[..., None] * color

    rgb = np.clip(rgb / max(1.0, rgb.max()), 0.0, 1.0)

    alpha_field = smooth_field(seed + 99, height, width, alpha=1.4)
    alpha = (np.clip(alpha_field, 0.0, 1.0) * 255).astype(np.uint8)

    rgb_u8 = (rgb * 255).astype(np.uint8)
    rgba = np.concatenate([rgb_u8, alpha[..., None]], axis=-1)
    return Image.fromarray(rgba, "RGBA")


@register_kind("transparent_sprite")
def transparent_sprite(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Solid-colored shape with a 1-pixel anti-aliased transparent edge.

    The 1px alpha boundary is the case that exposes binary-only alpha
    paths in some legacy GIF tooling and in poorly-tuned WebP encoders.
    """
    py_rng, _ = make_rng(seed)
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    color = (
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
        py_rng.randint(40, 240),
        255,
    )
    pad = max(2, min(width, height) // 16)
    draw.ellipse([pad, pad, width - pad - 1, height - pad - 1], fill=color)
    return img
