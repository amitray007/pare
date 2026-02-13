"""Tests for format detection via magic bytes."""

import gzip
import io
import struct

import pytest
from PIL import Image

from exceptions import UnsupportedFormatError
from utils.format_detect import (
    ImageFormat,
    _is_svg_content,
    detect_format,
    is_apng,
)

# --- detect_format ---


def test_detect_png():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    assert detect_format(buf.getvalue()) == ImageFormat.PNG


def test_detect_jpeg():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    assert detect_format(buf.getvalue()) == ImageFormat.JPEG


def test_detect_gif():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    assert detect_format(buf.getvalue()) == ImageFormat.GIF


def test_detect_webp():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    assert detect_format(buf.getvalue()) == ImageFormat.WEBP


def test_detect_bmp():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    assert detect_format(buf.getvalue()) == ImageFormat.BMP


def test_detect_tiff_little_endian():
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    assert detect_format(buf.getvalue()) == ImageFormat.TIFF


def test_detect_tiff_big_endian():
    """Big-endian TIFF magic bytes."""
    data = b"MM\x00\x2a" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.TIFF


def test_detect_svg():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    assert detect_format(svg) == ImageFormat.SVG


def test_detect_svg_with_xml_prolog():
    svg = b'<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    assert detect_format(svg) == ImageFormat.SVG


def test_detect_svg_with_bom():
    svg = b'\xef\xbb\xbf<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    assert detect_format(svg) == ImageFormat.SVG


def test_detect_svgz():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
    compressed = gzip.compress(svg)
    assert detect_format(compressed) == ImageFormat.SVGZ


def test_detect_gzip_non_svg():
    """Gzip data that is not SVG should raise."""
    compressed = gzip.compress(b"not an svg at all, just text")
    with pytest.raises(UnsupportedFormatError):
        detect_format(compressed)


def test_detect_too_small():
    with pytest.raises(UnsupportedFormatError, match="too small"):
        detect_format(b"\x89PN")


def test_detect_unknown():
    with pytest.raises(UnsupportedFormatError):
        detect_format(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")


# --- AVIF/HEIC via ISO BMFF ---


def test_detect_avif():
    # ftyp box: size(4) + "ftyp"(4) + brand(4) = 12 bytes
    data = struct.pack(">I", 20) + b"ftyp" + b"avif" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.AVIF


def test_detect_avif_avis():
    data = struct.pack(">I", 20) + b"ftyp" + b"avis" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.AVIF


def test_detect_heic():
    data = struct.pack(">I", 20) + b"ftyp" + b"heic" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.HEIC


def test_detect_heic_mif1():
    data = struct.pack(">I", 20) + b"ftyp" + b"mif1" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.HEIC


def test_detect_heic_heix():
    data = struct.pack(">I", 20) + b"ftyp" + b"heix" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.HEIC


def test_detect_avif_compatible_brand():
    """AVIF in compatible brands list, not major brand."""
    data = struct.pack(">I", 24) + b"ftyp" + b"isom" + b"\x00\x00\x00\x00" + b"avif" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.AVIF


def test_detect_heic_compatible_brand():
    """HEIC in compatible brands list."""
    data = struct.pack(">I", 24) + b"ftyp" + b"isom" + b"\x00\x00\x00\x00" + b"heic" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.HEIC


def test_detect_isobmff_unknown_brand():
    """Unknown ISO BMFF brand -> UnsupportedFormatError."""
    data = struct.pack(">I", 16) + b"ftyp" + b"isom" + b"\x00\x00\x00\x00"
    with pytest.raises(UnsupportedFormatError, match="unrecognized brand"):
        detect_format(data)


# --- is_apng ---


def test_is_apng_false_for_static_png(sample_png):
    assert is_apng(sample_png) is False


def test_is_apng_false_for_non_png():
    assert is_apng(b"not a png") is False


def test_is_apng_truncated():
    """Truncated PNG data: should return False gracefully."""
    assert is_apng(b"\x89PNG\r\n\x1a\n\x00\x00") is False


# --- _is_svg_content ---


def test_svg_content_xml_prolog():
    assert _is_svg_content(b'<?xml version="1.0"?><svg></svg>') is True


def test_svg_content_svg_tag():
    assert _is_svg_content(b'<svg xmlns="http://www.w3.org/2000/svg"></svg>') is True


def test_svg_content_with_bom():
    assert _is_svg_content(b"\xef\xbb\xbf<svg></svg>") is True


def test_svg_content_with_whitespace():
    assert _is_svg_content(b"   \n  <svg></svg>") is True


def test_svg_content_not_svg():
    assert _is_svg_content(b"<html><body>not svg</body></html>") is False


# --- JXL detection ---


def test_detect_jxl_bare_codestream():
    """Cover JXL bare codestream detection."""
    data = b"\xff\x0a" + b"\x00" * 100
    assert detect_format(data) == ImageFormat.JXL


def test_detect_jxl_isobmff_major():
    """Cover JXL ISO BMFF major brand detection."""
    box_size = struct.pack(">I", 20)
    ftyp = b"ftyp"
    major_brand = b"jxl "
    minor_version = b"\x00\x00\x00\x00"
    data = box_size + ftyp + major_brand + minor_version + b"\x00" * 100

    assert detect_format(data) == ImageFormat.JXL


def test_detect_jxl_isobmff_compat():
    """Cover JXL ISO BMFF compatible brand detection."""
    box_size = struct.pack(">I", 24)
    ftyp = b"ftyp"
    major_brand = b"unkn"
    minor_version = b"\x00\x00\x00\x00"
    compat_brand = b"jxl "
    data = box_size + ftyp + major_brand + minor_version + compat_brand + b"\x00" * 100

    assert detect_format(data) == ImageFormat.JXL


def test_is_apng_truncated_chunk():
    """Cover is_apng with truncated PNG chunk data."""
    png_sig = b"\x89PNG\r\n\x1a\n"
    chunk = struct.pack(">I", 1000) + b"IHDR" + b"\x00\x00"
    data = png_sig + chunk

    result = is_apng(data)
    assert result is False
