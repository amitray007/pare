"""Tests for estimation/png_header.py — pure-bytes PNG IHDR parser."""

from __future__ import annotations

import struct

import pytest

from estimation.png_header import PngHeader, parse_png_header

# ---------------------------------------------------------------------------
# Helpers to build synthetic valid PNG headers
# ---------------------------------------------------------------------------

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _make_png_header(
    width: int = 100,
    height: int = 80,
    bit_depth: int = 8,
    color_type: int = 2,  # RGB
) -> bytes:
    """Build a synthetic 33-byte PNG header (signature + IHDR chunk)."""
    ihdr_data = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    # IHDR chunk: length(4) + type(4) + data(13) + CRC(4) = 25 bytes
    length = struct.pack(">I", 13)
    ihdr_type = b"IHDR"
    # Fake CRC (not verified by parser)
    crc = b"\x00\x00\x00\x00"
    return _PNG_SIG + length + ihdr_type + ihdr_data + crc


# ---------------------------------------------------------------------------
# Valid fixture tests
# ---------------------------------------------------------------------------


class TestValidHeaders:
    def test_rgb(self):
        data = _make_png_header(width=200, height=150, bit_depth=8, color_type=2)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert isinstance(hdr, PngHeader)
        assert hdr.width == 200
        assert hdr.height == 150
        assert hdr.bit_depth == 8
        assert hdr.color_type == 2
        assert hdr.has_alpha is False
        assert hdr.is_palette is False

    def test_rgba(self):
        data = _make_png_header(width=64, height=64, bit_depth=8, color_type=6)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.color_type == 6
        assert hdr.has_alpha is True
        assert hdr.is_palette is False

    def test_grayscale(self):
        data = _make_png_header(width=32, height=32, bit_depth=8, color_type=0)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.color_type == 0
        assert hdr.has_alpha is False
        assert hdr.is_palette is False

    def test_grayscale_alpha(self):
        data = _make_png_header(width=10, height=10, bit_depth=8, color_type=4)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.color_type == 4
        assert hdr.has_alpha is True
        assert hdr.is_palette is False

    def test_palette(self):
        data = _make_png_header(width=256, height=256, bit_depth=8, color_type=3)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.color_type == 3
        assert hdr.has_alpha is False  # tRNS not in IHDR
        assert hdr.is_palette is True

    def test_16bit_depth(self):
        data = _make_png_header(width=100, height=100, bit_depth=16, color_type=2)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.bit_depth == 16

    def test_1bit_depth(self):
        data = _make_png_header(width=8, height=8, bit_depth=1, color_type=0)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.bit_depth == 1

    def test_extra_bytes_ignored(self):
        """Data longer than 33 bytes should parse correctly (tail ignored)."""
        data = _make_png_header(width=10, height=10) + b"\x00" * 1000
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.width == 10


# ---------------------------------------------------------------------------
# Invalid / truncated input tests
# ---------------------------------------------------------------------------


class TestInvalidHeaders:
    def test_too_short_returns_none(self):
        data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10  # only 18 bytes
        assert parse_png_header(data) is None

    def test_empty_returns_none(self):
        assert parse_png_header(b"") is None

    def test_exactly_32_bytes_returns_none(self):
        data = _make_png_header()[:32]
        assert parse_png_header(data) is None

    def test_wrong_signature_returns_none(self):
        bad = b"\x89PNX\r\n\x1a\n" + b"\x00" * 25
        assert parse_png_header(bad) is None

    def test_all_zeros_returns_none(self):
        assert parse_png_header(b"\x00" * 33) is None

    def test_wrong_ihdr_type_returns_none(self):
        data = _make_png_header()
        # Overwrite bytes 12-15 (IHDR type) with garbage
        data_ba = bytearray(data)
        data_ba[12:16] = b"XXXX"
        assert parse_png_header(bytes(data_ba)) is None

    def test_ihdr_length_not_13_returns_none(self):
        data = _make_png_header()
        data_ba = bytearray(data)
        # Overwrite chunk length (bytes 8-11) with 14
        data_ba[8:12] = struct.pack(">I", 14)
        assert parse_png_header(bytes(data_ba)) is None

    def test_invalid_color_type_returns_none(self):
        # color_type = 1 is not valid
        data = _make_png_header(color_type=1)
        assert parse_png_header(data) is None

    def test_invalid_color_type_5_returns_none(self):
        data = _make_png_header(color_type=5)
        assert parse_png_header(data) is None

    def test_invalid_bit_depth_returns_none(self):
        data = _make_png_header(bit_depth=3)
        assert parse_png_header(data) is None

    def test_zero_width_returns_none(self):
        data = _make_png_header(width=0)
        assert parse_png_header(data) is None

    def test_zero_height_returns_none(self):
        data = _make_png_header(height=0)
        assert parse_png_header(data) is None

    def test_max_dimension_returns_none(self):
        # 2^31 is exactly the threshold — should be rejected (>= check)
        data = _make_png_header(width=2**31)
        assert parse_png_header(data) is None

    def test_large_but_valid_dimension(self):
        # 2^31 - 1 is the largest valid dimension
        data = _make_png_header(width=2**31 - 1, height=1)
        hdr = parse_png_header(data)
        assert hdr is not None
        assert hdr.width == 2**31 - 1


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_parse_twice_identical(self):
        data = _make_png_header(width=320, height=240, bit_depth=8, color_type=6)
        result1 = parse_png_header(data)
        result2 = parse_png_header(data)
        assert result1 is not None
        assert result2 is not None
        assert result1 == result2

    def test_frozen_dataclass(self):
        data = _make_png_header()
        hdr = parse_png_header(data)
        assert hdr is not None
        with pytest.raises((AttributeError, TypeError)):
            hdr.width = 999  # type: ignore[misc]
