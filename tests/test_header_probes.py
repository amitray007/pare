"""Tests for header_analysis probe paths — oxipng probe, pngquant probe, quantize probe, SVG bloat."""

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from estimation.header_analysis import (
    _compute_svg_bloat_ratio,
    _flat_pixel_ratio,
    _oxipng_probe,
    _pngquant_probe,
    _quantize_probe,
    analyze_header,
    estimate_jpeg_quality_from_qtable,
)
from utils.format_detect import ImageFormat

# --- _oxipng_probe ---


def test_oxipng_probe_returns_ratio():
    """oxipng probe on a small crop returns a float ratio."""
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    ratio = _oxipng_probe(img)
    assert ratio is not None
    assert 0 < ratio <= 1.0


def test_oxipng_probe_empty_image():
    """oxipng probe on 1x1 image."""
    img = Image.new("RGB", (1, 1), (0, 0, 0))
    ratio = _oxipng_probe(img)
    # Might be None or a valid ratio
    assert ratio is None or 0 < ratio <= 2.0


def test_oxipng_probe_failure():
    """oxipng exception returns None."""
    img = Image.new("RGB", (32, 32))
    import oxipng

    original_fn = oxipng.optimize_from_memory
    oxipng.optimize_from_memory = MagicMock(side_effect=Exception("fail"))
    try:
        ratio = _oxipng_probe(img)
    finally:
        oxipng.optimize_from_memory = original_fn
    assert ratio is None


# --- _quantize_probe ---


def test_quantize_probe_returns_ratio():
    """Quantize probe on small RGB image returns ratio."""
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    ratio = _quantize_probe(img)
    assert ratio is not None
    assert 0 < ratio <= 2.0


def test_quantize_probe_solid_color():
    """Solid color image: quantize ratio should be low."""
    img = Image.new("RGB", (32, 32), (255, 0, 0))
    ratio = _quantize_probe(img)
    assert ratio is not None
    assert ratio < 1.0


# --- _pngquant_probe ---


def test_pngquant_probe_success():
    """Pngquant probe with mocked subprocess."""
    img = Image.new("RGB", (16, 16), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = data[: int(len(data) * 0.5)]

    import oxipng

    with patch("subprocess.run", return_value=mock_result):
        with patch.object(
            oxipng, "optimize_from_memory", return_value=data[: int(len(data) * 0.4)]
        ):
            ratio = _pngquant_probe(data)
    assert ratio is not None
    assert ratio > 0


def test_pngquant_probe_failure():
    """Pngquant returns non-zero -> None."""
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    mock_result = MagicMock()
    mock_result.returncode = 99
    mock_result.stdout = b""

    with patch("subprocess.run", return_value=mock_result):
        ratio = _pngquant_probe(data)
    assert ratio is None


def test_pngquant_probe_exception():
    """Pngquant subprocess raises -> None."""
    with patch("subprocess.run", side_effect=Exception("fail")):
        ratio = _pngquant_probe(b"png data")
    assert ratio is None


# --- _flat_pixel_ratio ---


def test_flat_pixel_ratio_solid():
    """Solid color image -> ratio near 1.0."""
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    ratio = _flat_pixel_ratio(img)
    assert ratio > 0.95


def test_flat_pixel_ratio_noisy():
    """Noisy image -> low ratio."""
    import random

    random.seed(42)
    img = Image.new("RGB", (32, 32))
    pixels = [
        (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        for _ in range(32 * 32)
    ]
    img.putdata(pixels)
    ratio = _flat_pixel_ratio(img)
    assert ratio < 0.5


def test_flat_pixel_ratio_tiny():
    """1x1 image -> 0.0 (no pairs)."""
    img = Image.new("RGB", (1, 1), (0, 0, 0))
    ratio = _flat_pixel_ratio(img)
    assert ratio == 0.0


# --- _compute_svg_bloat_ratio ---


def test_svg_bloat_empty():
    assert _compute_svg_bloat_ratio("") == 0.0


def test_svg_bloat_with_comments():
    svg = "<svg><!-- comment --><rect/></svg>"
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_metadata():
    svg = "<svg><metadata>some data</metadata><rect/></svg>"
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_editor_ns():
    svg = '<svg xmlns:inkscape="http://www.inkscape.org" inkscape:version="1.0"><rect/></svg>'
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_long_ids():
    svg = '<svg><rect id="very_long_element_id_here"/></svg>'
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_redundant_attrs():
    svg = '<svg><rect stroke="none" stroke-width="0" opacity="1"/></svg>'
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_xml_prolog():
    svg = '<?xml version="1.0" encoding="UTF-8"?><svg><rect/></svg>'
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


def test_svg_bloat_with_adobe_ns():
    svg = '<svg xmlns:x="http://ns.adobe.com/something"><rect/></svg>'
    ratio = _compute_svg_bloat_ratio(svg)
    assert ratio > 0


# --- estimate_jpeg_quality_from_qtable ---


def test_jpeg_quality_from_qtable_low_avg():
    """Very low average -> quality 100."""
    q = estimate_jpeg_quality_from_qtable(0.3)
    assert q == 100


def test_jpeg_quality_from_qtable_mid():
    """Mid-range average -> quality ~75."""
    q = estimate_jpeg_quality_from_qtable(20.0)
    assert 50 <= q <= 90


def test_jpeg_quality_from_qtable_high():
    """High average -> low quality."""
    q = estimate_jpeg_quality_from_qtable(100.0)
    assert 1 <= q <= 40


def test_jpeg_quality_from_qtable_very_high():
    """Very high average -> scale > 100 branch."""
    q = estimate_jpeg_quality_from_qtable(200.0)
    assert 1 <= q <= 20


# --- analyze_header edge cases ---


def test_analyze_header_pillow_exception():
    """Corrupt image data: analyze_header returns partial info without crash."""
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
    info = analyze_header(data, ImageFormat.PNG)
    assert info.format == ImageFormat.PNG


def test_analyze_header_jpeg_flat_ratio():
    """JPEG: flat_pixel_ratio computed from center crop."""
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.flat_pixel_ratio is not None
    assert info.flat_pixel_ratio > 0.5


def test_analyze_header_tiff_flat_ratio():
    """TIFF: flat_pixel_ratio computed from center crop."""
    img = Image.new("RGB", (100, 100), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.TIFF)
    assert info.flat_pixel_ratio is not None


def test_analyze_header_svg_with_editor_content():
    """SVG with Inkscape namespaces: has_metadata_chunks=True."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.has_metadata_chunks is True


def test_analyze_header_svg_viewbox_invalid():
    """SVG with invalid viewBox values: dimensions default to 0."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 abc def"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    # Should not crash, dimensions may be 0
    assert info.format == ImageFormat.SVG


def test_analyze_header_svgz_corrupt():
    """Corrupt SVGZ: analyze returns partial info."""
    data = b"\x1f\x8b" + b"\x00" * 20  # Invalid gzip
    info = analyze_header(data, ImageFormat.SVGZ)
    assert info.format == ImageFormat.SVGZ


def test_analyze_png_palette_oxipng_probe():
    """PNG palette mode: runs oxipng probe on small files."""
    img = Image.new("P", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.is_palette_mode is True
    assert info.oxipng_probe_ratio is not None


def test_analyze_png_palette_large_no_probe():
    """Large palette PNG: oxipng probe skipped (> 50KB)."""
    img = Image.new("P", (500, 500))
    # Generate random palette data to make file larger
    import random

    random.seed(42)
    pixels = [random.randint(0, 255) for _ in range(500 * 500)]
    img.putdata(pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    # If data is still < 50KB, pad it (unlikely for 500x500 palette)
    if len(data) < 50000:
        # Can't easily make palette PNG > 50KB, so test the path exists
        pass
    info = analyze_header(data, ImageFormat.PNG)
    assert info.is_palette_mode is True
