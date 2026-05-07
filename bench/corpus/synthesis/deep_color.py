"""10/12-bit deep-color content.

Pillow has no native mode for 10/12-bit RGB, so synthesizers here return
a `numpy.uint16` array with values in the *raw* bit-depth range
(`[0, 1023]` for 10-bit, `[0, 4095]` for 12-bit) — *never* stretched to
65535. Encoders that take a 10-bit array and assume 16-bit produce
clipped output; encoders that take a 16-bit array as 10-bit will quantize.

The conversion stage (`bench.corpus.conversion`) is responsible for
routing these to `pillow_heif` / `jxlpy` / `pillow-avif-plugin` with the
correct `bit_depth` parameter.
"""

from __future__ import annotations

import numpy as np

from bench.corpus.synthesis._common import (
    register_kind,
    smooth_field,
)


def _depth_to_max(bit_depth: int) -> int:
    if bit_depth not in (10, 12, 16):
        raise ValueError(f"unsupported deep-color bit_depth: {bit_depth}")
    return (1 << bit_depth) - 1


def _to_uint16(field01: np.ndarray, bit_depth: int) -> np.ndarray:
    """Map a [0, 1] float field to integer values in [0, (1<<bit_depth) - 1]."""
    max_val = _depth_to_max(bit_depth)
    return np.clip(field01 * max_val, 0, max_val).astype(np.uint16)


@register_kind("deep_color_smooth")
def deep_color_smooth(
    *,
    seed: int,
    width: int,
    height: int,
    bit_depth: int = 10,
) -> np.ndarray:
    """Smooth pink-noise field at the requested bit depth.

    Returns shape `(H, W, 3)` uint16 with values in `[0, (1<<bit_depth)-1]`.
    """
    r = _to_uint16(smooth_field(seed, height, width, alpha=1.6), bit_depth)
    g = _to_uint16(smooth_field(seed + 17, height, width, alpha=1.6), bit_depth)
    b = _to_uint16(smooth_field(seed + 31, height, width, alpha=1.6), bit_depth)
    return np.stack((r, g, b), axis=-1)


@register_kind("deep_color_thin_gradient")
def deep_color_thin_gradient(
    *,
    seed: int,
    width: int,
    height: int,
    bit_depth: int = 10,
    lo_frac: float = 0.40,
    hi_frac: float = 0.60,
) -> np.ndarray:
    """A narrow gradient band — 10/12-bit codecs should preserve smoothness.

    Banding visible in this case at 8-bit is a regression signal: the
    encoder either truncated bit depth, or the lossy quantizer is too
    aggressive.
    """
    max_val = _depth_to_max(bit_depth)
    lo = int(lo_frac * max_val)
    hi = int(hi_frac * max_val)
    ramp = np.linspace(lo, hi, height, dtype=np.float64)
    band = np.repeat(ramp[:, None], width, axis=1).astype(np.uint16)
    return np.repeat(band[..., None], 3, axis=-1)
