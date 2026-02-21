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
    png_quantize_ratio: Optional[float] = None  # PNG: quantized_size / original_size (thumbnail)
    oxipng_probe_ratio: Optional[float] = None  # PNG: oxipng_size / pillow_size (center crop)
    png_pngquant_probe_ratio: Optional[float] = (
        None  # PNG: pngquant+oxipng / original (actual file)
    )
    svg_bloat_ratio: Optional[float] = None  # SVG: removable bytes / total bytes
    flat_pixel_ratio: Optional[float] = (
        None  # Fraction of adjacent pixel pairs with diff < threshold
    )
    frame_count: int = 1  # 1 for static, >1 for animated
    file_size: int = 0
    raw_data: Optional[bytes] = None  # Raw file bytes for small files (< 10KB), used for probes


def analyze_header(data: bytes, fmt: ImageFormat) -> HeaderInfo:
    """Extract header information without full image decode.

    Uses Pillow in lazy mode (does not load pixel data) for most
    formats. PNG chunks are parsed directly for extra detail.
    """
    info = HeaderInfo(format=fmt, file_size=len(data))

    # Store raw data for small files so heuristics can run quality-dependent probes
    if len(data) < 12000:
        info.raw_data = data

    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return _analyze_svg(data, fmt, info)

    # Register optional format plugins so Pillow can open JXL/HEIC/AVIF
    if fmt == ImageFormat.JXL:
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            pass
    elif fmt == ImageFormat.HEIC:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
    elif fmt == ImageFormat.AVIF:
        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            pass

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
        info.bit_depth = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8}.get(img.mode, 8)

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

    # Content classification from center crop for formats that need it
    if fmt in (ImageFormat.JPEG, ImageFormat.TIFF, ImageFormat.BMP, ImageFormat.JXL):
        try:
            w = info.dimensions.get("width", 0)
            h = info.dimensions.get("height", 0)
            crop_size = min(64, w, h)
            if crop_size >= 8:
                cx, cy = w // 2, h // 2
                half = crop_size // 2
                crop_img = Image.open(io.BytesIO(data))
                crop = crop_img.crop((cx - half, cy - half, cx + half, cy + half))
                info.flat_pixel_ratio = _flat_pixel_ratio(crop.convert("RGB"))
        except Exception:
            pass

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
        # Run oxipng probe for small palette PNGs to measure lossless potential
        if len(data) < 50000:
            try:
                import oxipng

                optimized = oxipng.optimize_from_memory(data)
                info.oxipng_probe_ratio = len(optimized) / len(data)
            except Exception:
                pass
    else:
        # Non-palette: run content probes (color ratio, quantize, oxipng, flatness)
        _analyze_png_content(data, info)

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


def _analyze_png_content(data: bytes, info: HeaderInfo) -> None:
    """Compute content probes for non-palette PNG images.

    Sets on info: unique_color_ratio, png_quantize_ratio,
    flat_pixel_ratio, oxipng_probe_ratio, png_pngquant_probe_ratio.

    For small files (< 50KB), runs oxipng on the actual file for an exact
    lossless compression measurement. For small images (< 250K pixels),
    also runs pngquant + oxipng on the actual file to measure lossy potential.
    For larger files, uses a 64x64 center crop probe.
    Both paths compute flat_pixel_ratio from the center crop.
    """
    try:
        # For small files: run oxipng on actual data (fast, in-process).
        # This gives exact lossless savings measurement.
        if len(data) < 50000:
            try:
                import oxipng

                optimized = oxipng.optimize_from_memory(data)
                info.oxipng_probe_ratio = len(optimized) / len(data)
            except Exception:
                pass

        img = Image.open(io.BytesIO(data))
        orig_w, orig_h = img.size

        # Center crop at original resolution for pixel-level metrics.
        crop_size = min(64, orig_w, orig_h)
        if crop_size >= 8:
            cx, cy = orig_w // 2, orig_h // 2
            half = crop_size // 2
            crop = img.crop((cx - half, cy - half, cx + half, cy + half))
            rgb_crop = crop.convert("RGB")
            info.flat_pixel_ratio = _flat_pixel_ratio(rgb_crop)
            # Crop-based oxipng probe only if full-file probe wasn't done
            if info.oxipng_probe_ratio is None:
                info.oxipng_probe_ratio = _oxipng_probe(rgb_crop)

        # Pngquant probe: run pngquant + oxipng on actual file for small images.
        # Gives exact lossy optimization measurement (~25ms for small files).
        if len(data) < 50000 and orig_w * orig_h < 250000:
            info.png_pngquant_probe_ratio = _pngquant_probe(data)

        # Thumbnail for color ratio and quantize ratio
        img.thumbnail((64, 64))
        rgb = img.convert("RGB")
        pixels = list(rgb.getdata())
        total = len(pixels)
        if total == 0:
            return
        unique = len(set(pixels))
        info.unique_color_ratio = unique / total
        info.png_quantize_ratio = _quantize_probe(rgb)
    except Exception:
        pass


def _flat_pixel_ratio(rgb: Image.Image, threshold: int = 24) -> float:
    """Fraction of adjacent pixel pairs with L1 color distance below threshold.

    Measures local uniformity in an RGB thumbnail. High values (>0.85)
    indicate screenshot/UI content with large flat regions; low values
    (<0.50) indicate photographic or noisy content.

    Args:
        rgb: RGB mode Pillow image (typically 64x64 thumbnail).
        threshold: Sum-of-channels difference below which a pair is "flat".
    """
    pixels = list(rgb.getdata())
    w, h = rgb.size
    if w < 2 or h < 2:
        return 0.0

    flat = 0
    total = 0

    # Horizontal neighbors
    for y in range(h):
        off = y * w
        for x in range(w - 1):
            p1 = pixels[off + x]
            p2 = pixels[off + x + 1]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) < threshold:
                flat += 1
            total += 1

    # Vertical neighbors
    for y in range(h - 1):
        off = y * w
        off2 = (y + 1) * w
        for x in range(w):
            p1 = pixels[off + x]
            p2 = pixels[off2 + x]
            if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2]) < threshold:
                flat += 1
            total += 1

    return flat / total if total > 0 else 0.0


def _oxipng_probe(rgb_crop: Image.Image) -> Optional[float]:
    """Measure oxipng optimization potential on a center crop.

    Saves the crop as PNG via Pillow, runs oxipng in-process (pyoxipng),
    returns oxipng_size / pillow_size. Low ratios mean the content is
    highly compressible with optimized PNG encoding.
    """
    try:
        import oxipng

        buf = io.BytesIO()
        rgb_crop.save(buf, format="PNG")
        pillow_png = buf.getvalue()
        pillow_size = len(pillow_png)
        if pillow_size == 0:
            return None

        optimized = oxipng.optimize_from_memory(pillow_png)
        return len(optimized) / pillow_size
    except Exception:
        return None


def _pngquant_probe(data: bytes) -> Optional[float]:
    """Run pngquant + oxipng on actual file to measure lossy optimization.

    Uses permissive quality range (0-100) so pngquant always succeeds
    if the image can be quantized at all. The heuristics layer applies
    quality gating to account for stricter ranges at runtime.

    Returns pngquant_oxipng_size / original_file_size, or None on failure.
    Only called for small images (< 250K pixels) to keep latency low (~25ms).
    """
    import subprocess

    try:
        result = subprocess.run(
            ["pngquant", "--quality", "0-100", "-", "--output", "-"],
            input=data,
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        import oxipng

        optimized = oxipng.optimize_from_memory(result.stdout)
        return len(optimized) / len(data)
    except Exception:
        return None


def _quantize_probe(rgb: Image.Image) -> Optional[float]:
    """Measure how well a thumbnail quantizes to 256 colors.

    Saves the RGB thumbnail as PNG, quantizes to 256 colors, saves again,
    and returns quantized_size / original_size. Low ratios (~0.3-0.5) mean
    spatially coherent content that pngquant will compress well.
    """
    try:
        orig_buf = io.BytesIO()
        rgb.save(orig_buf, format="PNG")
        orig_size = orig_buf.tell()
        if orig_size == 0:
            return None

        quantized = rgb.quantize(colors=256)
        quant_buf = io.BytesIO()
        quantized.save(quant_buf, format="PNG")
        quant_size = quant_buf.tell()

        return quant_size / orig_size
    except Exception:
        return None


def estimate_jpeg_quality_from_qtable(avg_q: float) -> int:
    """Estimate JPEG quality from average quantization table value.

    Uses the inverse of the IJG (Independent JPEG Group) formula:
      For q >= 50: scale = 200 - 2*quality  → quality = (200 - scale) / 2
      For q < 50:  scale = 5000 / quality    → quality = 5000 / scale

    The average quantization value approximates scale/100 of the base table,
    so avg_q ≈ base_avg * scale / 100 where base_avg ≈ 57.625 for the
    standard IJG luminance table.
    """
    if avg_q <= 0.5:
        return 100
    # Derive approximate scale factor from IJG luminance base table avg
    scale = (avg_q / 57.625) * 100.0
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
    info.is_progressive = img.info.get("progressive", False) or img.info.get("progression", False)


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
    has_editor = "xmlns:inkscape" in text or "xmlns:sodipodi" in text or "adobe" in text.lower()
    info.has_metadata_chunks = has_comments or has_metadata or has_editor

    # Compute SVG bloat ratio: estimate removable bytes
    info.svg_bloat_ratio = _compute_svg_bloat_ratio(text)

    return info


def _compute_svg_bloat_ratio(text: str) -> float:
    """Estimate fraction of SVG text that is removable bloat."""
    import re

    total = len(text)
    if total == 0:
        return 0.0

    removable = 0

    # Comment bytes: <!-- ... -->
    for m in re.finditer(r"<!--[\s\S]*?-->", text):
        removable += len(m.group())

    # XML prolog: <?xml ...?>
    for m in re.finditer(r"<\?xml[^?]*\?>", text):
        removable += len(m.group())

    # Metadata elements: <metadata>...</metadata>
    for m in re.finditer(r"<metadata[\s\S]*?</metadata>", text, re.IGNORECASE):
        removable += len(m.group())

    # Editor namespace declarations and prefixed attributes
    for m in re.finditer(r'xmlns:(inkscape|sodipodi)="[^"]*"', text):
        removable += len(m.group())
    for m in re.finditer(r'(inkscape|sodipodi):[a-zA-Z-]+="[^"]*"', text):
        removable += len(m.group())
    # Adobe-specific namespace/attributes
    for m in re.finditer(r'xmlns:x="[^"]*adobe[^"]*"', text, re.IGNORECASE):
        removable += len(m.group())

    # Long IDs: savings from shortening (len(id) - 2 per long id)
    for m in re.finditer(r'id="([^"]+)"', text):
        id_val = m.group(1)
        if len(id_val) > 2:
            removable += len(id_val) - 2

    # Redundant attributes
    for pattern in [r'stroke="none"', r'stroke-width="0"', r'opacity="1"']:
        for m in re.finditer(pattern, text):
            removable += len(m.group())

    return min(removable / total, 1.0)
