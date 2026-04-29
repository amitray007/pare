"""Pathological cases that expose codec-specific failure modes.

Each entry here targets a known artifact / quality bug:

- thin_gradient        — banding at low bit depth (q-floor bugs).
- sharp_diagonal       — mosquito noise on JPEG / AVIF I-frames.
- block_aligned_check  — DCT block-boundary bleeding.
- text_on_flat         — chroma subsampling artifacts (4:2:0).
- white_noise          — compression floor for transform coders.
- solid_fill           — palette codecs win huge; lossy may quantize wrong.
- alpha_edge_sprite    — 1-px alpha edge preservation through WebP/AVIF.
- chroma_clash         — saturated red on saturated blue (4:2:0 worst).
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from bench.corpus.synthesis._common import make_rng, register_kind


@register_kind("path_thin_gradient")
def path_thin_gradient(
    *,
    seed: int,
    width: int,
    height: int,
    lo: int = 80,
    hi: int = 120,
) -> Image.Image:
    """Vertical 80→120 grayscale band — exposes 8-bit banding."""
    ramp = np.linspace(lo, hi, height, dtype=np.float64)
    band = np.repeat(ramp[:, None], width, axis=1)
    arr = np.repeat(band[..., None], 3, axis=-1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


@register_kind("path_sharp_diagonal")
def path_sharp_diagonal(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Alternating 1-pixel black/white diagonals — JPEG mosquito noise."""
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((xx + yy) % 2 == 0).astype(np.uint8) * 255
    arr = np.repeat(pattern[..., None], 3, axis=-1)
    return Image.fromarray(arr, "RGB")


@register_kind("path_block_aligned_check")
def path_block_aligned_check(
    *,
    seed: int,
    width: int,
    height: int,
    block: int = 8,
) -> Image.Image:
    """8×8 checker aligned to JPEG's DCT grid — block-boundary bleeding."""
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = (((xx // block) + (yy // block)) % 2 == 0).astype(np.uint8) * 255
    arr = np.repeat(pattern[..., None], 3, axis=-1)
    return Image.fromarray(arr, "RGB")


@register_kind("path_text_on_flat")
def path_text_on_flat(
    *,
    seed: int,
    width: int,
    height: int,
    bg: tuple[int, int, int] = (255, 220, 220),
    fg: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Black text on a flat colored background — chroma-subsample artifacts."""
    from PIL import ImageFont

    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(12, height // 16))
    except TypeError:
        font = ImageFont.load_default()

    line = "The quick brown fox jumps over the lazy dog 0123456789"
    line_h = max(16, height // 14)
    y = max(8, height // 20)
    while y + line_h < height:
        draw.text((max(8, width // 30), y), line, font=font, fill=fg)
        y += line_h
    return img


@register_kind("path_white_noise")
def path_white_noise(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Uniform white noise — compression floor; alias of photo_noise."""
    _, np_rng = make_rng(seed)
    arr = np_rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@register_kind("path_solid_fill")
def path_solid_fill(
    *,
    seed: int,
    width: int,
    height: int,
    color: tuple[int, int, int] = (123, 200, 64),
) -> Image.Image:
    """Single-color fill — palette codecs win; lossy may quantize wrong."""
    return Image.new("RGB", (width, height), color)


@register_kind("path_alpha_edge_sprite")
def path_alpha_edge_sprite(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Filled rectangle with a 1-pixel transparent border."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([1, 1, width - 2, height - 2], fill=(220, 80, 40, 255))
    return img


@register_kind("path_chroma_clash")
def path_chroma_clash(
    *,
    seed: int,
    width: int,
    height: int,
) -> Image.Image:
    """Saturated red text on saturated blue — 4:2:0 chroma worst case."""
    from PIL import ImageFont

    img = Image.new("RGB", (width, height), (0, 0, 220))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=max(14, height // 12))
    except TypeError:
        font = ImageFont.load_default()
    line = "RED ON BLUE - chroma clash"
    line_h = max(20, height // 10)
    y = max(8, height // 16)
    while y + line_h < height:
        draw.text((max(8, width // 30), y), line, font=font, fill=(255, 0, 0))
        y += line_h
    return img
