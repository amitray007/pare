"""Shared primitives for seeded synthesizers.

Determinism rules:

1.  Always pass an explicit `random.Random(seed)` instance — never call
    `random.seed()` (other libraries also use the global PRNG).
2.  Use `numpy.random.default_rng(seed)` for vectorized work; never use
    the legacy `numpy.random` module-level functions.
3.  Pixel data is the contract. Encoder output may vary across builds.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np
from PIL import Image

SynthFn = Callable[..., Image.Image]


_REGISTRY: dict[str, SynthFn] = {}


def register_kind(name: str) -> Callable[[SynthFn], SynthFn]:
    """Register a synthesizer under a content_kind name."""

    def decorator(fn: SynthFn) -> SynthFn:
        if name in _REGISTRY:
            raise ValueError(f"content_kind already registered: {name!r}")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_synth(name: str) -> SynthFn:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown content_kind: {name!r}") from None


def known_kinds() -> list[str]:
    return sorted(_REGISTRY)


def make_rng(seed: int) -> tuple[random.Random, np.random.Generator]:
    """Pair of seeded PRNGs — one for Python, one for numpy.

    Same seed; never touches the globals.
    """
    return random.Random(seed), np.random.default_rng(seed)


def smooth_field(seed: int, height: int, width: int, alpha: float = 1.6) -> np.ndarray:
    """Generate a 1/f^alpha pink-noise field, normalized to [0, 1].

    Used as the base layer for "photographic" content — has the smooth,
    natural-looking spectrum of real photos without the encoder-specific
    artifacts of stock images.
    """
    _, np_rng = make_rng(seed)
    noise = np_rng.standard_normal((height, width))
    fft = np.fft.fft2(noise)
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fy, fx)
    radius[0, 0] = 1.0  # placeholder; mask will zero this bin
    mask = 1.0 / (radius**alpha)
    mask[0, 0] = 0.0  # kill DC
    smoothed = np.fft.ifft2(fft * mask).real
    smoothed -= smoothed.min()
    peak = smoothed.max()
    if peak > 0:
        smoothed /= peak
    return smoothed


def array_to_rgb(channels: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Image.Image:
    """Stack three [0, 1] float arrays into an RGB Pillow image."""
    rgb = np.stack(channels, axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, "RGB")
