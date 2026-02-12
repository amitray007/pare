"""Tests for header analysis — format-specific header parsing."""

import gzip
import io

from PIL import Image

from estimation.header_analysis import (
    _compute_svg_bloat_ratio,
    _flat_pixel_ratio,
    analyze_header,
    estimate_jpeg_quality_from_qtable,
)
from utils.format_detect import ImageFormat

# --- analyze_header basics ---


def test_header_png():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.dimensions["width"] == 100
    assert info.dimensions["height"] == 80
    assert info.color_type == "rgb"
    assert info.bit_depth == 8


def test_header_png_palette():
    img = Image.new("P", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.is_palette_mode is True
    assert info.color_type == "palette"


def test_header_png_rgba():
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.color_type == "rgba"


def test_header_png_grayscale():
    img = Image.new("L", (50, 50), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.color_type == "grayscale"


def test_header_jpeg():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.dimensions["width"] == 100
    assert info.dimensions["height"] == 80
    assert info.estimated_quality is not None
    assert 60 <= info.estimated_quality <= 90


def test_header_jpeg_progressive():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75, progressive=True)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.is_progressive


def test_header_bmp():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.BMP)
    assert info.dimensions["width"] == 100
    assert info.dimensions["height"] == 80


def test_header_tiff():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.TIFF)
    assert info.dimensions["width"] == 100
    assert info.flat_pixel_ratio is not None


def test_header_gif():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="GIF")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.GIF)
    assert info.dimensions["width"] == 100


def test_header_webp():
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.WEBP)
    assert info.dimensions["width"] == 100


def test_header_small_file_raw_data():
    """Small files should have raw_data stored."""
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.raw_data is not None


def test_header_icc_profile():
    """ICC profile detected."""
    img = Image.new("RGB", (10, 10))
    # Create a minimal ICC profile
    icc = b"\x00" * 128  # minimal header
    buf = io.BytesIO()
    img.save(buf, format="PNG", icc_profile=icc)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.has_icc_profile is True


def test_header_corrupt_data():
    """Corrupt data returns partial info without crashing."""
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
    info = analyze_header(data, ImageFormat.PNG)
    assert info.format == ImageFormat.PNG


# --- SVG analysis ---


def test_header_svg():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.dimensions["width"] == 800
    assert info.dimensions["height"] == 600
    assert info.svg_bloat_ratio is not None


def test_header_svgz():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300"><rect/></svg>'
    data = gzip.compress(svg)
    info = analyze_header(data, ImageFormat.SVGZ)
    assert info.dimensions["width"] == 400


def test_header_svg_with_metadata():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><!-- comment --><metadata>data</metadata><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.has_metadata_chunks is True


def test_header_svg_with_editor_namespaces():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.has_metadata_chunks is True


# --- estimate_jpeg_quality_from_qtable ---


def test_qtable_high_quality():
    """Very low avg quantization -> high quality."""
    quality = estimate_jpeg_quality_from_qtable(0.3)
    assert quality == 100


def test_qtable_mid_quality():
    """Moderate avg quantization."""
    quality = estimate_jpeg_quality_from_qtable(30.0)
    assert 70 <= quality <= 90


def test_qtable_low_quality():
    """High avg quantization -> low quality."""
    quality = estimate_jpeg_quality_from_qtable(150.0)
    assert quality < 30


def test_qtable_scale_above_100():
    """Scale > 100 uses the 5000/scale formula."""
    quality = estimate_jpeg_quality_from_qtable(200.0)
    assert 1 <= quality <= 100


# --- _flat_pixel_ratio ---


def test_flat_ratio_solid():
    """Solid color image -> very high flat ratio."""
    img = Image.new("RGB", (10, 10), (128, 128, 128))
    ratio = _flat_pixel_ratio(img)
    assert ratio > 0.95


def test_flat_ratio_noise():
    """Noisy image -> low flat ratio."""
    import random

    random.seed(99)
    img = Image.new("RGB", (10, 10))
    for x in range(10):
        for y in range(10):
            img.putpixel(
                (x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            )
    ratio = _flat_pixel_ratio(img)
    assert ratio < 0.5


def test_flat_ratio_tiny():
    """1x1 image -> 0.0."""
    img = Image.new("RGB", (1, 1))
    assert _flat_pixel_ratio(img) == 0.0


# --- _compute_svg_bloat_ratio ---


def test_svg_bloat_with_comments():
    text = "<!-- A comment --><svg><rect/></svg>"
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_with_metadata():
    text = "<svg><metadata>lots of data here</metadata><rect/></svg>"
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_with_editor_ns():
    text = '<svg xmlns:inkscape="http://www.inkscape.org" inkscape:version="1.0"><rect/></svg>'
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_with_long_ids():
    text = '<svg><rect id="very_long_element_id_here"/></svg>'
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_with_redundant_attrs():
    text = '<svg><rect stroke="none" stroke-width="0" opacity="1"/></svg>'
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_with_xml_prolog():
    text = '<?xml version="1.0"?><svg><rect/></svg>'
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio > 0


def test_svg_bloat_empty():
    assert _compute_svg_bloat_ratio("") == 0.0


def test_svg_bloat_clean():
    """Clean SVG with minimal bloat."""
    text = '<svg><rect width="10" height="10"/></svg>'
    ratio = _compute_svg_bloat_ratio(text)
    assert ratio < 0.1
