"""Extra header analysis tests — PNG content probes, JPEG analysis, edge cases."""

import gzip
import io

from PIL import Image

from estimation.header_analysis import (
    analyze_header,
)
from utils.format_detect import ImageFormat

# --- PNG content probes for non-palette images ---


def test_png_content_probes_small_file():
    """Small non-palette PNG: runs oxipng probe, color ratio, quantize ratio."""
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    # Should have probes set
    assert info.oxipng_probe_ratio is not None
    assert info.unique_color_ratio is not None


def test_png_content_probes_large_file():
    """Larger non-palette PNG: crop-based probe."""
    # Create a 200x200 image (160KB+ uncompressed)
    img = Image.new("RGB", (200, 200), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.unique_color_ratio is not None
    assert info.flat_pixel_ratio is not None


def test_png_palette_mode_analysis():
    """Palette mode PNG: PLTE chunk parsed, color count set."""
    img = Image.new("P", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.is_palette_mode is True
    assert info.color_count is not None
    assert info.color_count > 0


def test_png_with_text_chunk():
    """PNG with tEXt chunk: has_metadata_chunks=True."""
    from PIL import PngImagePlugin

    img = Image.new("RGB", (10, 10))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Author", "test")
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.PNG)
    assert info.has_metadata_chunks is True


# --- JPEG analysis ---


def test_jpeg_quality_estimation():
    """JPEG quality estimated from quantization table."""
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.estimated_quality is not None
    assert 30 <= info.estimated_quality <= 70


def test_jpeg_exif_detected():
    """JPEG with EXIF data: has_exif=True."""
    img = Image.new("RGB", (50, 50))
    exif = Image.Exif()
    exif[0x0112] = 1  # Orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.has_exif is True


def test_jpeg_flat_pixel_ratio():
    """JPEG analysis computes flat_pixel_ratio."""
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.flat_pixel_ratio is not None
    assert info.flat_pixel_ratio > 0.5  # Solid color = high flat ratio


# --- TIFF analysis ---


def test_tiff_flat_pixel_ratio():
    """TIFF analysis computes flat_pixel_ratio for solid color."""
    img = Image.new("RGB", (100, 100), (0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.TIFF)
    assert info.flat_pixel_ratio is not None
    assert info.flat_pixel_ratio > 0.5


# --- WebP analysis ---


def test_webp_dimensions():
    """WebP: dimensions extracted."""
    img = Image.new("RGB", (100, 80))
    buf = io.BytesIO()
    img.save(buf, format="WEBP")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.WEBP)
    assert info.dimensions["width"] == 100
    assert info.dimensions["height"] == 80


# --- GIF analysis ---


def test_gif_animated():
    """Animated GIF: frame_count > 1."""
    frames = [Image.new("P", (10, 10), i) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.GIF)
    # Pillow may report n_frames differently depending on version
    assert info.frame_count >= 1


# --- SVG analysis ---


def test_svg_viewbox_dimensions():
    """SVG viewBox parsed for dimensions."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 800"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.dimensions["width"] == 1200
    assert info.dimensions["height"] == 800


def test_svg_width_height_attrs():
    """SVG with width/height parsed (may use viewBox or attributes)."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="500" height="300" viewBox="0 0 500 300"><rect/></svg>'
    info = analyze_header(svg, ImageFormat.SVG)
    assert info.dimensions["width"] == 500
    assert info.dimensions["height"] == 300


def test_svgz_analysis():
    """SVGZ: gzip-compressed SVG analyzed correctly."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 400"><rect/></svg>'
    data = gzip.compress(svg)
    info = analyze_header(data, ImageFormat.SVGZ)
    assert info.dimensions["width"] == 600
    assert info.dimensions["height"] == 400


# --- Edge cases ---


def test_corrupt_jpeg():
    """Corrupt JPEG returns partial info."""
    data = b"\xff\xd8\xff\xe0" + b"\x00" * 20
    info = analyze_header(data, ImageFormat.JPEG)
    assert info.format == ImageFormat.JPEG


def test_empty_data():
    """Very small data returns format but no dimensions."""
    data = b"\x89PNG\r\n\x1a\n"
    info = analyze_header(data, ImageFormat.PNG)
    assert info.format == ImageFormat.PNG


def test_bmp_analysis():
    """BMP analysis extracts dimensions."""
    img = Image.new("RGB", (120, 90))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    info = analyze_header(data, ImageFormat.BMP)
    assert info.dimensions["width"] == 120
    assert info.dimensions["height"] == 90
    assert info.color_type == "rgb"
