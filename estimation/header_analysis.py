import io
import struct
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image

from utils.format_detect import ImageFormat


@dataclass
class HeaderInfo:
    """Parsed image header information."""

    format: ImageFormat
    dimensions: dict = field(default_factory=lambda: {"width": 0, "height": 0})
    color_type: Optional[str] = None  # "rgb", "rgba", "palette", "grayscale"
    bit_depth: Optional[int] = None
    has_icc_profile: bool = False
    has_exif: bool = False
    estimated_quality: Optional[int] = None  # JPEG only
    is_progressive: bool = False  # JPEG only
    is_palette_mode: bool = False  # PNG only
    color_count: Optional[int] = None  # PNG palette mode only
    has_metadata_chunks: bool = False  # PNG text chunks, SVG comments
    unique_color_ratio: Optional[float] = None  # PNG non-palette: unique colors / total pixels
    frame_count: int = 1  # 1 for static, >1 for animated
    file_size: int = 0


def analyze_header(data: bytes, fmt: ImageFormat) -> HeaderInfo:
    """Extract header information without full image decode.

    Uses Pillow in lazy mode (does not load pixel data) for most
    formats. PNG chunks are parsed directly for extra detail.
    """
    info = HeaderInfo(format=fmt, file_size=len(data))

    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return _analyze_svg(data, fmt, info)

    # Use Pillow lazy mode for raster formats
    try:
        img = Image.open(io.BytesIO(data))
        info.dimensions = {"width": img.width, "height": img.height}

        # Color type
        mode_map = {
            "RGB": "rgb",
            "RGBA": "rgba",
            "P": "palette",
            "L": "grayscale",
            "LA": "grayscale",
            "1": "grayscale",
            "CMYK": "cmyk",
        }
        info.color_type = mode_map.get(img.mode, img.mode.lower())

        # Bit depth (approximate from mode)
        info.bit_depth = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8}.get(
            img.mode, 8
        )

        # ICC profile
        info.has_icc_profile = "icc_profile" in img.info

        # EXIF
        try:
            exif = img.getexif()
            info.has_exif = len(exif) > 0
        except Exception:
            pass

        # Frame count (animation)
        try:
            info.frame_count = getattr(img, "n_frames", 1)
        except Exception:
            info.frame_count = 1

    except Exception:
        return info

    # Format-specific analysis
    if fmt in (ImageFormat.PNG, ImageFormat.APNG):
        _analyze_png_extra(data, info)
    elif fmt == ImageFormat.JPEG:
        _analyze_jpeg_extra(data, img, info)

    return info


def _analyze_png_extra(data: bytes, info: HeaderInfo) -> None:
    """PNG-specific: check palette mode, color count, color complexity, metadata chunks."""
    info.is_palette_mode = info.color_type == "palette"

    if info.is_palette_mode:
        # Count colors in PLTE chunk
        offset = 8  # Skip PNG signature
        while offset + 8 <= len(data):
            chunk_len = struct.unpack(">I", data[offset : offset + 4])[0]
            chunk_type = data[offset + 4 : offset + 8]
            if chunk_type == b"PLTE":
                info.color_count = chunk_len // 3  # 3 bytes per color
                break
            offset += 4 + 4 + chunk_len + 4
    else:
        # Non-palette: sample unique colors via 64x64 thumbnail
        info.unique_color_ratio = _sample_unique_color_ratio(data)

    # Check for text metadata chunks
    text_types = {b"tEXt", b"iTXt", b"zTXt"}
    offset = 8
    while offset + 8 <= len(data):
        chunk_len = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        if chunk_type in text_types:
            info.has_metadata_chunks = True
            break
        if chunk_type == b"IDAT":
            break
        offset += 4 + 4 + chunk_len + 4


def _sample_unique_color_ratio(data: bytes) -> Optional[float]:
    """Compute unique-color ratio from a 64x64 thumbnail.

    Returns unique_colors / total_pixels. Low values (~0.01) indicate
    flat graphics amenable to pngquant; high values (~0.8+) indicate
    photographic content where pngquant will fail (exit code 99).
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((64, 64))
        # Convert to RGB to normalize (drop alpha for color counting)
        rgb = img.convert("RGB")
        pixels = list(rgb.getdata())
        total = len(pixels)
        if total == 0:
            return None
        unique = len(set(pixels))
        return unique / total
    except Exception:
        return None


def estimate_jpeg_quality_from_qtable(avg_q: float) -> int:
    """Estimate JPEG quality from average quantization table value.

    Uses the inverse of the IJG (Independent JPEG Group) formula:
      For q >= 50: scale = 200 - 2*quality  → quality = (200 - scale) / 2
      For q < 50:  scale = 5000 / quality    → quality = 5000 / scale

    The average quantization value approximates scale/100 of the base table,
    so avg_q ≈ base_avg * scale / 100 where base_avg ≈ 25 for luminance.
    """
    if avg_q <= 0.5:
        return 100
    # Derive approximate scale factor (base luminance table avg ≈ 25)
    scale = (avg_q / 25.0) * 100.0
    if scale < 100:
        quality = int((200 - scale) / 2)
    else:
        quality = int(5000 / scale)
    return max(1, min(100, quality))


def _analyze_jpeg_extra(data: bytes, img: Image.Image, info: HeaderInfo) -> None:
    """JPEG-specific: estimate quality, check progressive."""
    try:
        qtables = img.quantization
        if qtables:
            table = qtables[0] if 0 in qtables else list(qtables.values())[0]
            avg_q = sum(table) / len(table)
            info.estimated_quality = estimate_jpeg_quality_from_qtable(avg_q)
    except Exception:
        info.estimated_quality = None

    # Check progressive
    info.is_progressive = img.info.get("progressive", False) or img.info.get(
        "progression", False
    )


def _analyze_svg(data: bytes, fmt: ImageFormat, info: HeaderInfo) -> HeaderInfo:
    """SVG-specific: analyze text content for optimization signals."""
    import gzip

    if fmt == ImageFormat.SVGZ:
        try:
            text = gzip.decompress(data).decode("utf-8", errors="replace")
        except Exception:
            return info
    else:
        text = data.decode("utf-8", errors="replace")

    # Estimate dimensions from viewBox or width/height attributes
    import re

    viewbox_match = re.search(r'viewBox="([^"]*)"', text)
    if viewbox_match:
        parts = viewbox_match.group(1).split()
        if len(parts) == 4:
            try:
                info.dimensions = {
                    "width": int(float(parts[2])),
                    "height": int(float(parts[3])),
                }
            except ValueError:
                pass

    # Check for metadata/comments
    has_comments = "<!--" in text
    has_metadata = "<metadata" in text.lower()
    has_editor = 'xmlns:inkscape' in text or 'xmlns:sodipodi' in text or 'adobe' in text.lower()
    info.has_metadata_chunks = has_comments or has_metadata or has_editor

    return info
