"""Animated + deep-color synthesizer tests."""

from __future__ import annotations

import numpy as np
from PIL import Image

from bench.corpus.manifest import Bucket, ManifestEntry, pixel_sha256
from bench.corpus.synthesis import synthesize


def _entry(kind: str, **overrides) -> ManifestEntry:
    base = dict(
        name=f"t_{kind}",
        bucket=Bucket.SMALL,
        content_kind=kind,
        seed=1,
        width=128,
        height=96,
        output_formats=["png"],
    )
    base.update(overrides)
    return ManifestEntry(**base)


def test_animated_translation_returns_four_frames():
    frames = synthesize(_entry("animated_translation"))
    assert isinstance(frames, list)
    assert len(frames) == 4
    assert all(isinstance(f, Image.Image) for f in frames)
    assert all(f.size == (128, 96) for f in frames)


def test_animated_fade_changes_pixels_across_frames():
    frames = synthesize(_entry("animated_fade"))
    h0 = pixel_sha256(frames[0])
    h2 = pixel_sha256(frames[2])
    assert h0 != h2, "alpha=0 and alpha=1 frames must differ"


def test_animated_sprite_static_keeps_corners_unchanged():
    """The bouncing ball changes a small region; corners should match."""
    frames = synthesize(_entry("animated_sprite_static", width=200, height=200))
    a = np.asarray(frames[0])
    b = np.asarray(frames[2])
    # 16x16 corner patch should be identical (background only, no sprite there)
    assert np.array_equal(a[:16, :16], b[:16, :16])


def test_animated_redraw_every_frame_differs():
    frames = synthesize(_entry("animated_redraw"))
    hashes = {pixel_sha256(f) for f in frames}
    assert len(hashes) == 4


def test_animated_pixel_sha256_handles_frame_list():
    frames = synthesize(_entry("animated_translation"))
    h = pixel_sha256(frames)
    assert isinstance(h, str) and len(h) == 64


def test_animated_pixel_sha256_is_deterministic():
    a = pixel_sha256(synthesize(_entry("animated_translation", seed=99)))
    b = pixel_sha256(synthesize(_entry("animated_translation", seed=99)))
    assert a == b


def test_deep_color_smooth_returns_uint16_array_in_10bit_range():
    arr = synthesize(_entry("deep_color_smooth", params={"bit_depth": 10}))
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint16
    assert arr.shape == (96, 128, 3)
    assert arr.max() <= (1 << 10) - 1, "values must not exceed 10-bit max"


def test_deep_color_smooth_supports_12bit():
    arr = synthesize(_entry("deep_color_smooth", params={"bit_depth": 12}, width=64, height=64))
    assert arr.max() <= (1 << 12) - 1


def test_deep_color_smooth_rejects_unsupported_depth():
    import pytest

    with pytest.raises(ValueError, match="bit_depth"):
        synthesize(_entry("deep_color_smooth", params={"bit_depth": 8}))


def test_deep_color_thin_gradient_is_seed_independent():
    a = pixel_sha256(synthesize(_entry("deep_color_thin_gradient", seed=1)))
    b = pixel_sha256(synthesize(_entry("deep_color_thin_gradient", seed=2)))
    assert a == b, "thin gradient is fixed pattern; seed is irrelevant"


def test_pixel_sha256_distinguishes_bit_depths():
    """A 10-bit and a 12-bit array of the same shape must hash differently."""
    a = synthesize(_entry("deep_color_smooth", params={"bit_depth": 10}))
    b = synthesize(_entry("deep_color_smooth", params={"bit_depth": 12}))
    assert pixel_sha256(a) != pixel_sha256(b)


def test_pixel_sha256_handles_ndarray_input():
    arr = np.zeros((4, 4, 3), dtype=np.uint16)
    h = pixel_sha256(arr)
    assert len(h) == 64
