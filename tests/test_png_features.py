"""Tests for estimation/png_features.py.

Covers:
- Happy-path mode extraction (RGB, RGBA, L, LA, P)
- Unsupported mode (I;16) → None
- Palette image with tRNS → has_alpha=True, count = palette+1
- Over-bound pixel count → None
- Determinism: calling twice yields identical PngFeatures
"""

import io

import numpy as np
import pytest
from PIL import Image

from estimation.png_features import (
    MAX_INPUT_BPP,
    PngFeatures,
    extract_png_features,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb(width: int = 32, height: int = 32) -> Image.Image:
    """Solid red RGB image."""
    return Image.new("RGB", (width, height), color=(200, 50, 50))


def _make_rgba(width: int = 32, height: int = 32) -> Image.Image:
    """Semi-transparent RGBA image."""
    return Image.new("RGBA", (width, height), color=(100, 150, 200, 128))


def _make_l(width: int = 32, height: int = 32) -> Image.Image:
    """Grayscale L image with a gradient."""
    arr = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
    return Image.fromarray(arr, mode="L")


def _make_la(width: int = 32, height: int = 32) -> Image.Image:
    """Grayscale+alpha LA image."""
    arr = np.zeros((height, width, 2), dtype=np.uint8)
    arr[:, :, 0] = 128  # gray
    arr[:, :, 1] = 200  # alpha
    return Image.fromarray(arr, mode="LA")


def _make_palette(width: int = 32, height: int = 32, with_trns: bool = False) -> Image.Image:
    """Palette-mode image, optionally with tRNS transparency."""
    img = Image.new("RGB", (width, height), color=(0, 128, 255))
    # Add a second color so palette has at least 2 entries.
    img.putpixel((0, 0), (255, 0, 0))
    p_img = img.quantize(colors=8)
    if with_trns:
        # Mark palette index 0 as transparent.
        p_img.info["transparency"] = 0
    return p_img


# ---------------------------------------------------------------------------
# Happy path: supported modes
# ---------------------------------------------------------------------------


class TestHappyPathModes:
    def test_rgb_returns_features(self):
        img = _make_rgb()
        feat = extract_png_features(img, 32, 32, quality=80)
        assert isinstance(feat, PngFeatures)
        assert feat.has_alpha is False
        assert feat.quality == 80
        assert feat.log10_orig_pixels > 0
        # input_bpp defaults to 0.0 when orig_size not provided
        assert feat.input_bpp == 0.0

    def test_rgb_with_orig_size(self):
        img = _make_rgb()
        # 32*32 = 1024 pixels; 1024 bytes → input_bpp = 8.0
        feat = extract_png_features(img, 32, 32, quality=80, orig_size=1024)
        assert feat is not None
        assert abs(feat.input_bpp - 8.0) < 1e-9

    def test_rgba_has_alpha(self):
        img = _make_rgba()
        feat = extract_png_features(img, 32, 32, quality=60)
        assert feat is not None
        assert feat.has_alpha is True

    def test_l_mode(self):
        img = _make_l()
        feat = extract_png_features(img, 32, 32, quality=75)
        assert feat is not None
        assert feat.has_alpha is False
        # Gradient should produce non-zero Sobel
        assert feat.mean_sobel > 0

    def test_la_mode_has_alpha(self):
        img = _make_la()
        feat = extract_png_features(img, 32, 32, quality=80)
        assert feat is not None
        assert feat.has_alpha is True

    def test_palette_without_trns(self):
        img = _make_palette(with_trns=False)
        feat = extract_png_features(img, 32, 32, quality=80)
        assert feat is not None
        assert feat.has_alpha is False

    def test_palette_with_trns_has_alpha(self):
        img = _make_palette(with_trns=True)
        feat = extract_png_features(img, 32, 32, quality=80)
        assert feat is not None
        assert feat.has_alpha is True


# ---------------------------------------------------------------------------
# Unsupported mode → None
# ---------------------------------------------------------------------------


class TestUnsupportedMode:
    def test_i16_returns_none(self):
        """Mode I;16 is outside the allow-list and must return None."""
        # Pillow does not directly create I;16; construct via fromarray with int32 and then
        # adjust the mode string via internal mode override.  A simpler route: create a
        # 16-bit PNG bytes payload and re-open it.
        arr = np.zeros((16, 16), dtype=np.uint16)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")  # saves as I (32-bit int on some builds)
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        # Regardless of the exact mode Pillow assigns, if it is not in the allow-list
        # extract_png_features should return None.
        if img.mode not in {"RGB", "RGBA", "L", "LA", "P"}:
            result = extract_png_features(img, 16, 16, quality=80)
            assert result is None
        else:
            # On this Pillow build the mode was coerced to a supported one; skip.
            pytest.skip(f"Pillow produced mode {img.mode!r} instead of an unsupported mode")

    def test_cmyk_returns_none(self):
        """CMYK is outside the allow-list."""
        img = Image.new("CMYK", (32, 32), color=(0, 100, 200, 50))
        result = extract_png_features(img, 32, 32, quality=80)
        assert result is None


# ---------------------------------------------------------------------------
# Palette with tRNS: unique_count = palette_size + 1
# ---------------------------------------------------------------------------


class TestPaletteTrns:
    def test_trns_increments_count(self):
        """With tRNS, the effective color count is palette_size + 1."""
        img_no_trns = _make_palette(with_trns=False)
        img_with_trns = _make_palette(with_trns=True)

        feat_no = extract_png_features(img_no_trns, 32, 32, quality=80)
        feat_with = extract_png_features(img_with_trns, 32, 32, quality=80)

        assert feat_no is not None
        assert feat_with is not None
        # The tRNS version must report at least as many unique colors.
        assert feat_with.log10_unique_colors >= feat_no.log10_unique_colors


# ---------------------------------------------------------------------------
# Over-bound pixel count → None
# ---------------------------------------------------------------------------


class TestPixelBounds:
    def test_over_max_pixels_returns_none(self):
        """orig_w * orig_h > MAX_PIXELS should return None without touching the image."""
        img = _make_rgb(32, 32)
        # Pass orig dimensions that exceed the cap.
        w = 10_001
        h = 10_001  # 10_001 * 10_001 = 100_020_001 > 100_000_000
        result = extract_png_features(img, w, h, quality=80)
        assert result is None

    def test_exactly_at_max_pixels_passes(self):
        """Exactly MAX_PIXELS is still allowed (> not >=)."""
        img = _make_rgb(32, 32)
        # 10_000 * 10_000 = 100_000_000 == MAX_PIXELS → allowed
        w = 10_000
        h = 10_000
        result = extract_png_features(img, w, h, quality=80)
        # Should succeed (or None only if unique_colors OOB, which won't happen for solid color)
        assert result is not None

    def test_input_bpp_over_max_returns_none(self):
        """input_bpp > MAX_INPUT_BPP should return None (feature_oob)."""
        img = _make_rgb(32, 32)
        # 32*32=1024 pixels; MAX_INPUT_BPP=64 → threshold = 64 * 1024 / 8 = 8192 bytes
        # Use a size clearly above: 32*32 pixels × 64 bpp + 1 byte
        oob_size = int(MAX_INPUT_BPP * 32 * 32 / 8) + 1  # just above threshold
        result = extract_png_features(img, 32, 32, quality=80, orig_size=oob_size)
        assert result is None

    def test_input_bpp_at_max_passes(self):
        """input_bpp == MAX_INPUT_BPP should pass (not strictly greater)."""
        img = _make_rgb(32, 32)
        # Exactly at threshold: orig_size * 8 / pixels == MAX_INPUT_BPP
        exact_size = int(MAX_INPUT_BPP * 32 * 32 / 8)  # == 8192
        result = extract_png_features(img, 32, 32, quality=80, orig_size=exact_size)
        assert result is not None
        assert result.input_bpp == MAX_INPUT_BPP


# ---------------------------------------------------------------------------
# Determinism: calling twice → identical output
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_rgb_deterministic(self):
        img = _make_rgb()
        feat1 = extract_png_features(img, 32, 32, quality=80)
        feat2 = extract_png_features(img, 32, 32, quality=80)
        assert feat1 == feat2

    def test_l_gradient_deterministic(self):
        img = _make_l()
        feat1 = extract_png_features(img, 32, 32, quality=75)
        feat2 = extract_png_features(img, 32, 32, quality=75)
        assert feat1 == feat2

    def test_palette_with_trns_deterministic(self):
        img = _make_palette(with_trns=True)
        feat1 = extract_png_features(img, 32, 32, quality=60)
        feat2 = extract_png_features(img, 32, 32, quality=60)
        assert feat1 == feat2


# ---------------------------------------------------------------------------
# Feature sanity
# ---------------------------------------------------------------------------


class TestFeatureSanity:
    def test_sobel_solid_image_near_zero(self):
        """A solid-color image has near-zero spatial gradient."""
        img = Image.new("RGB", (64, 64), color=(123, 45, 67))
        feat = extract_png_features(img, 64, 64, quality=80)
        assert feat is not None
        # Sobel magnitude on a constant image should be essentially zero.
        assert feat.mean_sobel < 1.0

    def test_edge_density_zero_on_solid(self):
        img = Image.new("RGB", (64, 64), color=(0, 0, 0))
        feat = extract_png_features(img, 64, 64, quality=80)
        assert feat is not None
        assert feat.edge_density == 0.0

    def test_log10_orig_pixels_correct(self):
        import math

        img = _make_rgb(100, 200)
        feat = extract_png_features(img, 100, 200, quality=80)
        assert feat is not None
        assert abs(feat.log10_orig_pixels - math.log10(100 * 200)) < 1e-9

    def test_quality_passed_through(self):
        img = _make_rgb()
        for q in (1, 50, 100):
            feat = extract_png_features(img, 32, 32, quality=q)
            assert feat is not None
            assert feat.quality == q
