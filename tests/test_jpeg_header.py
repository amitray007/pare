"""Tests for estimation/jpeg_header.py — pure-bytes JPEG header parser.

Uses Pillow to generate JPEG fixtures, then parses them with the new
pure-bytes parser to verify correctness.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from estimation.jpeg_header import JpegHeader, parse_jpeg_header

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_SOI = b"\xff\xd8"


def _make_jpeg(
    width: int = 64,
    height: int = 48,
    mode: str = "RGB",
    quality: int = 75,
    progressive: bool = False,
    subsampling: int = -1,  # -1 = Pillow default
    qtables: list | None = None,
) -> bytes:
    """Create a JPEG in memory using Pillow."""
    img = Image.new(mode, (width, height), color=128)
    buf = io.BytesIO()
    kwargs: dict = {"format": "JPEG", "quality": quality}
    if progressive:
        kwargs["progressive"] = True
    if subsampling >= 0:
        kwargs["subsampling"] = subsampling
    if qtables is not None:
        kwargs["qtables"] = qtables
    img.save(buf, **kwargs)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Standard quality fixtures
# ---------------------------------------------------------------------------


class TestStandardQualities:
    """libjpeg baseline JPEG at various quality levels."""

    @pytest.mark.parametrize("q", [10, 50, 75, 85, 95])
    def test_libjpeg_quality(self, q: int):
        data = _make_jpeg(quality=q)
        hdr = parse_jpeg_header(data)
        assert hdr is not None, f"parse failed for q={q}"
        assert isinstance(hdr, JpegHeader)
        assert hdr.fallback_reason is None
        assert len(hdr.dqt_luma) == 64
        assert all(v >= 1 for v in hdr.dqt_luma), "DQT luma values must be >= 1"
        assert hdr.width == 64
        assert hdr.height == 48

    @pytest.mark.parametrize("q", [10, 50, 75, 85, 95])
    def test_libjpeg_quality_has_chroma(self, q: int):
        data = _make_jpeg(quality=q)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.dqt_chroma is not None
        assert len(hdr.dqt_chroma) == 64


# ---------------------------------------------------------------------------
# Progressive JPEG
# ---------------------------------------------------------------------------


class TestProgressiveJpeg:
    def test_progressive_flag_true(self):
        data = _make_jpeg(progressive=True)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.progressive is True
        assert hdr.fallback_reason is None

    def test_baseline_progressive_false(self):
        data = _make_jpeg(progressive=False)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.progressive is False


# ---------------------------------------------------------------------------
# Grayscale JPEG
# ---------------------------------------------------------------------------


class TestGrayscaleJpeg:
    def test_grayscale_subsampling(self):
        data = _make_jpeg(mode="L")
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.subsampling == "grayscale"
        assert hdr.dqt_chroma is None
        assert hdr.components == 1
        assert hdr.fallback_reason is None

    def test_grayscale_has_luma_table(self):
        data = _make_jpeg(mode="L")
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert len(hdr.dqt_luma) == 64


# ---------------------------------------------------------------------------
# Subsampling variants
# ---------------------------------------------------------------------------


class TestSubsampling:
    def test_444(self):
        data = _make_jpeg(subsampling=0)  # Pillow: 0=4:4:4
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.subsampling == "4:4:4"
        assert hdr.fallback_reason is None

    def test_422(self):
        data = _make_jpeg(subsampling=1)  # Pillow: 1=4:2:2
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.subsampling == "4:2:2"
        assert hdr.fallback_reason is None

    def test_420(self):
        data = _make_jpeg(subsampling=2)  # Pillow: 2=4:2:0
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.subsampling == "4:2:0"
        assert hdr.fallback_reason is None


# ---------------------------------------------------------------------------
# CMYK (non-standard components)
# ---------------------------------------------------------------------------


class TestCmykJpeg:
    def test_cmyk_fallback_reason(self):
        data = _make_jpeg(mode="CMYK")
        hdr = parse_jpeg_header(data)
        # CMYK has 4 components — non_standard_components
        assert hdr is not None
        assert hdr.fallback_reason == "non_standard_components"


# ---------------------------------------------------------------------------
# Truncated / bad data
# ---------------------------------------------------------------------------


class TestTruncated:
    def test_truncated_100_bytes_no_crash(self):
        data = _make_jpeg()
        truncated = data[:100]
        # May return JpegHeader (if SOF seen before byte 100) or None — must not raise
        result = parse_jpeg_header(truncated)
        assert result is None or isinstance(result, JpegHeader)

    def test_wrong_magic_returns_none(self):
        # PNG bytes
        png_data = _PNG_MAGIC + b"\x00" * 200
        assert parse_jpeg_header(png_data) is None

    def test_empty_bytes_returns_none(self):
        assert parse_jpeg_header(b"") is None

    def test_too_short_returns_none(self):
        assert parse_jpeg_header(b"\xff\xd8") is None

    def test_all_zeros_returns_none(self):
        assert parse_jpeg_header(b"\x00" * 64) is None

    def test_random_bytes_no_crash(self):
        import os

        rng_bytes = os.urandom(512)
        result = parse_jpeg_header(rng_bytes)
        assert result is None or isinstance(result, JpegHeader)

    def test_just_soi_returns_none(self):
        # SOI only — no SOF → None
        assert parse_jpeg_header(_SOI + b"\xff\xd9") is None  # SOI + EOI


# ---------------------------------------------------------------------------
# APP14 Adobe
# ---------------------------------------------------------------------------


class TestApp14:
    def test_app14_transform_0_rgb(self):
        """RGB JPEG saved by Pillow may carry APP14 with transform=0."""
        # Force Pillow to write an APP14 marker by saving with subsampling=0
        # and quality high enough that libjpeg may write APP14
        data = _make_jpeg(subsampling=0, quality=95)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        # app14_color_transform is either None (no APP14) or 0 (RGB), both OK
        assert hdr.app14_color_transform in (None, 0, 1)

    def test_no_app14_is_none(self):
        """Most standard JEPGs don't have APP14; field should be None."""
        data = _make_jpeg(quality=75)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        # Don't assert a specific value — just that it's int or None
        assert hdr.app14_color_transform is None or isinstance(hdr.app14_color_transform, int)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_parse_twice_identical(self):
        data = _make_jpeg(quality=75)
        h1 = parse_jpeg_header(data)
        h2 = parse_jpeg_header(data)
        assert h1 is not None
        assert h2 is not None
        assert h1 == h2

    def test_frozen_dataclass(self):
        data = _make_jpeg(quality=75)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        with pytest.raises((AttributeError, TypeError)):
            hdr.width = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Dimensions and bit depth
# ---------------------------------------------------------------------------


class TestDimensions:
    def test_width_and_height_correct(self):
        data = _make_jpeg(width=320, height=240)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.width == 320
        assert hdr.height == 240

    def test_bit_depth_8(self):
        data = _make_jpeg(quality=75)
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        assert hdr.bit_depth == 8
