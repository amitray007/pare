"""Size-bucket targeting and validation.

The corpus declares a target bucket per entry (`tiny`, `small`, ...,
`xlarge`). Synthesis happens at a chosen `(width, height)`; the encoded
file size depends on content kind and format. This module:

1.  Validates that an actual encoded file landed in its declared bucket.
2.  Estimates dimensions to hit a target byte size, given a content kind
    and an encoder. Uses a one-shot bytes-per-pixel probe followed by
    optional 1–2 step refinement.

Aspect ratio defaults to 4:3 but can be overridden (e.g. 1:1 for sprites).
"""

from __future__ import annotations

import io
import math
from typing import Callable

import numpy as np
from PIL import Image

from bench.corpus.manifest import BUCKET_RANGES, Bucket, bucket_for_size

EncoderFn = Callable[[Image.Image], bytes]


def in_bucket(byte_size: int, bucket: Bucket | str) -> bool:
    """Return True if `byte_size` falls inside the declared bucket's range."""
    name = bucket.value if isinstance(bucket, Bucket) else bucket
    if name not in BUCKET_RANGES:
        raise ValueError(f"unknown bucket: {name!r}")
    lo, hi = BUCKET_RANGES[name]
    return byte_size >= lo and (hi is None or byte_size < hi)


def bucket_center(bucket: Bucket | str) -> int:
    """Geometric center of a bucket — useful as a target for refinement."""
    name = bucket.value if isinstance(bucket, Bucket) else bucket
    lo, hi = BUCKET_RANGES[name]
    if hi is None:
        return lo * 2
    return int(math.sqrt(max(lo, 1) * hi))


def png_encoder(image: Image.Image) -> bytes:
    """Pillow PNG with default compression — used as a generic probe."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def jpeg_encoder(quality: int = 75) -> EncoderFn:
    def encode(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    return encode


def webp_encoder(quality: int = 75) -> EncoderFn:
    def encode(image: Image.Image) -> bytes:
        buf = io.BytesIO()
        image.save(buf, format="WEBP", quality=quality)
        return buf.getvalue()

    return encode


def fit_bpp(
    synth_fn: Callable[..., Image.Image],
    encoder: EncoderFn,
    *,
    probe_w: int = 256,
    probe_h: int = 256,
    seed: int = 0,
    **synth_params,
) -> float:
    """Encode a probe synthesis and return bytes-per-pixel.

    A single 256×256 probe is enough for a useful estimate because encoded
    size scales near-linearly with pixel count for a given content class
    and quality. `target_dimensions()` does the inverse mapping.
    """
    if probe_w <= 0 or probe_h <= 0:
        raise ValueError(f"probe dims must be positive: ({probe_w}, {probe_h})")

    probe = synth_fn(seed=seed, width=probe_w, height=probe_h, **synth_params)
    if not isinstance(probe, Image.Image):
        raise TypeError("fit_bpp only supports synthesizers returning Image.Image")

    encoded = encoder(probe)
    return len(encoded) / (probe_w * probe_h)


def target_dimensions(
    bpp: float,
    target_bytes: int,
    *,
    aspect: float = 4 / 3,
    min_dim: int = 16,
) -> tuple[int, int]:
    """Compute (width, height) to roughly hit `target_bytes` when encoded.

    Assumes encoded size scales linearly with pixel count and the linear
    coefficient is `bpp`. Aspect ratio is width / height.
    """
    if bpp <= 0:
        raise ValueError(f"bpp must be positive: {bpp}")
    target_pixels = target_bytes / bpp
    height = max(min_dim, int(round((target_pixels / aspect) ** 0.5)))
    width = max(min_dim, int(round(height * aspect)))
    return width, height


def refine_to_bucket(
    synth_fn: Callable[..., Image.Image],
    encoder: EncoderFn,
    target_bucket: Bucket,
    *,
    aspect: float = 4 / 3,
    max_iters: int = 3,
    seed: int = 0,
    **synth_params,
) -> tuple[Image.Image, int, int, int]:
    """Iteratively pick dimensions to land an encoded file inside `target_bucket`.

    Strategy: probe at 256×256 to get an initial bpp estimate, predict
    dimensions that hit the bucket center, encode, and either accept or
    rescale by sqrt(actual / target) area ratio. Up to `max_iters` rescales.

    Returns (image, width, height, encoded_bytes). Raises if the target
    bucket cannot be reached (e.g., synthesizer's minimum dims overshoot
    the tiny bucket — happens for `text_screenshot` and similar).
    """
    bpp = fit_bpp(synth_fn, encoder, seed=seed, **synth_params)
    target = bucket_center(target_bucket)
    width, height = target_dimensions(bpp, target, aspect=aspect)

    last_size = 0
    for _ in range(max_iters + 1):
        image = synth_fn(seed=seed, width=width, height=height, **synth_params)
        encoded = encoder(image)
        size = len(encoded)
        last_size = size
        if in_bucket(size, target_bucket):
            return image, width, height, size

        lo, hi = BUCKET_RANGES[target_bucket.value]
        if hi is not None and size >= hi:
            new_target = (lo + hi) / 2
        else:
            new_target = max(lo, target)

        scale = (new_target / max(size, 1)) ** 0.5
        width = max(8, int(round(width * scale)))
        height = max(8, int(round(height * scale)))

    raise SizingConvergenceError(
        f"could not land inside {target_bucket.value} after {max_iters + 1} attempts; "
        f"last size={last_size} dims={width}x{height} actual_bucket={bucket_for_size(last_size).value}"
    )


class SizingConvergenceError(Exception):
    """`refine_to_bucket` exhausted iterations without hitting the target."""


def ndarray_byte_size(arr: np.ndarray) -> int:
    """For deep-color content where there's no encoded form yet."""
    return arr.nbytes
