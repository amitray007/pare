"""Tests for metadata stripping — JPEG, PNG, TIFF paths."""

import io
import struct

import pytest
from PIL import Image

from utils.metadata import (
    strip_metadata_selective,
    _strip_jpeg_metadata,
    _strip_png_metadata,
    _strip_pillow_metadata,
)
from utils.format_detect import ImageFormat


def _make_jpeg_with_exif():
    """Create JPEG with EXIF orientation and ICC profile."""
    img = Image.new("RGB", (50, 50), (128, 64, 32))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation = Rotate 90
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def _make_jpeg_with_icc():
    """Create JPEG with ICC profile."""
    from PIL import ImageCms
    srgb = ImageCms.createProfile("sRGB")
    icc_data = ImageCms.ImageCmsProfile(srgb).tobytes()
    img = Image.new("RGB", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", icc_profile=icc_data)
    return buf.getvalue()


def _make_png_with_text():
    """Create PNG with tEXt chunks."""
    from PIL import PngImagePlugin
    img = Image.new("RGB", (10, 10))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Author", "test")
    pnginfo.add_text("Description", "test description")
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def _make_tiff_with_exif():
    """Create TIFF with EXIF data."""
    img = Image.new("RGB", (50, 50))
    exif = Image.Exif()
    exif[0x0112] = 3  # Orientation
    buf = io.BytesIO()
    img.save(buf, format="TIFF", exif=exif.tobytes())
    return buf.getvalue()


# --- strip_metadata_selective dispatch ---


def test_strip_dispatch_jpeg():
    data = _make_jpeg_with_exif()
    result = strip_metadata_selective(data, ImageFormat.JPEG)
    assert len(result) > 0


def test_strip_dispatch_png():
    data = _make_png_with_text()
    result = strip_metadata_selective(data, ImageFormat.PNG)
    assert len(result) > 0


def test_strip_dispatch_apng():
    data = _make_png_with_text()
    result = strip_metadata_selective(data, ImageFormat.APNG)
    assert len(result) > 0


def test_strip_dispatch_tiff():
    data = _make_tiff_with_exif()
    result = strip_metadata_selective(data, ImageFormat.TIFF)
    assert len(result) > 0


def test_strip_dispatch_webp_passthrough():
    """WebP metadata stripping returns data unchanged."""
    data = b"fake webp data"
    result = strip_metadata_selective(data, ImageFormat.WEBP)
    assert result == data


def test_strip_dispatch_gif_passthrough():
    """GIF metadata stripping returns data unchanged."""
    data = b"fake gif data"
    result = strip_metadata_selective(data, ImageFormat.GIF)
    assert result == data


# --- JPEG metadata ---


def test_jpeg_strip_preserves_orientation():
    """JPEG strip preserves orientation tag."""
    data = _make_jpeg_with_exif()
    result = _strip_jpeg_metadata(data, preserve_orientation=True, preserve_icc=True)
    # Verify orientation is still present
    img = Image.open(io.BytesIO(result))
    exif = img.getexif()
    assert exif.get(0x0112) == 6


def test_jpeg_strip_no_orientation():
    """JPEG strip without preserving orientation."""
    data = _make_jpeg_with_exif()
    result = _strip_jpeg_metadata(data, preserve_orientation=False, preserve_icc=True)
    img = Image.open(io.BytesIO(result))
    exif = img.getexif()
    assert 0x0112 not in exif


def test_jpeg_strip_preserves_icc():
    """JPEG strip preserves ICC profile."""
    data = _make_jpeg_with_icc()
    result = _strip_jpeg_metadata(data, preserve_orientation=True, preserve_icc=True)
    img = Image.open(io.BytesIO(result))
    assert "icc_profile" in img.info


def test_jpeg_strip_no_icc():
    """JPEG strip without preserving ICC."""
    data = _make_jpeg_with_icc()
    result = _strip_jpeg_metadata(data, preserve_orientation=True, preserve_icc=False)
    img = Image.open(io.BytesIO(result))
    assert "icc_profile" not in img.info


# --- PNG metadata ---


def test_png_strip_removes_text_chunks():
    """PNG strip removes tEXt chunks."""
    data = _make_png_with_text()
    result = _strip_png_metadata(data, preserve_icc=True)
    # Verify tEXt chunks are gone
    assert b"tEXt" not in result[8:]  # Skip PNG signature


def test_png_strip_preserves_essential_chunks():
    """PNG strip preserves IHDR, IDAT, IEND, pHYs."""
    data = _make_png_with_text()
    result = _strip_png_metadata(data, preserve_icc=True)
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
    assert b"IHDR" in result
    assert b"IDAT" in result
    assert b"IEND" in result


def test_png_strip_non_png():
    """Non-PNG data: returned as-is."""
    data = b"not a png"
    result = _strip_png_metadata(data, preserve_icc=True)
    assert result == data


def test_png_strip_no_icc():
    """PNG strip without preserving ICC removes iCCP."""
    data = _make_png_with_text()
    result = _strip_png_metadata(data, preserve_icc=False)
    assert len(result) > 0


def test_png_strip_incomplete_chunk():
    """PNG with truncated last chunk: remaining data preserved."""
    # Valid PNG header + IHDR chunk + truncated extra
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    # Append a truncated chunk (only length, no type/data/crc)
    truncated = data + b"\x00\x00\x00\x05"
    result = _strip_png_metadata(truncated, preserve_icc=True)
    assert len(result) > 0


# --- TIFF metadata ---


def test_tiff_strip_preserves_orientation():
    """TIFF strip preserves orientation."""
    data = _make_tiff_with_exif()
    result = _strip_pillow_metadata(
        data, ImageFormat.TIFF, preserve_orientation=True, preserve_icc=True
    )
    img = Image.open(io.BytesIO(result))
    exif = img.getexif()
    assert exif.get(0x0112) == 3


def test_tiff_strip_no_orientation():
    """TIFF strip without orientation."""
    data = _make_tiff_with_exif()
    result = _strip_pillow_metadata(
        data, ImageFormat.TIFF, preserve_orientation=False, preserve_icc=True
    )
    img = Image.open(io.BytesIO(result))
    exif = img.getexif()
    assert 0x0112 not in exif


def test_tiff_strip_with_icc():
    """TIFF strip preserves ICC if present."""
    img = Image.new("RGB", (50, 50))
    from PIL import ImageCms
    srgb = ImageCms.createProfile("sRGB")
    icc_data = ImageCms.ImageCmsProfile(srgb).tobytes()
    buf = io.BytesIO()
    img.save(buf, format="TIFF", icc_profile=icc_data)
    data = buf.getvalue()
    result = _strip_pillow_metadata(
        data, ImageFormat.TIFF, preserve_orientation=True, preserve_icc=True
    )
    img2 = Image.open(io.BytesIO(result))
    assert "icc_profile" in img2.info
