"""Sample-based estimation engine.

Instead of heuristic prediction, this module compresses a downsized sample
of the image using the actual optimizers and extrapolates BPP (bits per pixel)
to the full image size.

For small images (<150K pixels), SVG, and animated formats, it compresses the
full file for an exact result.
"""

import asyncio
import io

from PIL import Image

from optimizers.router import optimize_image
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

SAMPLE_MAX_WIDTH = 300
EXACT_PIXEL_THRESHOLD = 150_000  # ~390x390 pixels


async def estimate(
    data: bytes,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate compression savings by compressing a sample.

    For small images, SVGs, and animated images: compresses the full file.
    For large raster images: downsamples to ~300px wide, compresses the
    sample, and extrapolates output BPP to the original pixel count.
    """
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(data)
    file_size = len(data)

    # SVG/SVGZ: no pixel data — compress the whole file
    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return await _estimate_exact(data, fmt, config, file_size)

    # Decode image for dimensions and animation detection
    img = await asyncio.to_thread(_open_image, data)
    width, height = img.size
    original_pixels = width * height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Animated images: compress full file (inter-frame redundancy matters)
    frame_count = getattr(img, "n_frames", 1)
    if frame_count > 1:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Small images: compress fully for exact result
    if original_pixels <= EXACT_PIXEL_THRESHOLD:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Large raster images: downsample + compress sample + extrapolate BPP
    return await _estimate_by_sample(
        data, img, fmt, config, file_size, width, height, color_type, bit_depth
    )


def _open_image(data: bytes) -> Image.Image:
    """Open image in Pillow (lazy decode — reads header only)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


async def _estimate_exact(
    data: bytes,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int = 0,
    height: int = 0,
    color_type: str | None = None,
    bit_depth: int | None = None,
) -> EstimateResponse:
    """Compress the full image with the actual optimizer. Returns exact result."""
    result = await optimize_image(data, config)
    already_optimized = result.method == "none"
    reduction = result.reduction_percent if not already_optimized else 0.0

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=result.optimized_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=already_optimized,
        confidence="high",
    )


async def _estimate_by_sample(
    data: bytes,
    img: Image.Image,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
) -> EstimateResponse:
    """Downsample to ~300px wide, compress sample, extrapolate BPP."""
    original_pixels = width * height

    # Proportional resize
    ratio = SAMPLE_MAX_WIDTH / width
    sample_width = SAMPLE_MAX_WIDTH
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

    # Create sample encoded with minimal compression
    sample_data = await asyncio.to_thread(_create_sample, img, sample_width, sample_height, fmt)

    # Compress sample with the actual optimizer
    result = await optimize_image(sample_data, config)

    # If optimizer says "already optimized", propagate that
    if result.method == "none":
        return EstimateResponse(
            original_size=file_size,
            original_format=fmt.value,
            dimensions={"width": width, "height": height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="high",
        )

    # Extrapolate output BPP to original pixel count
    sample_output_bpp = result.optimized_size * 8 / sample_pixels
    estimated_size = int(sample_output_bpp * original_pixels / 8)
    estimated_size = min(estimated_size, file_size)

    reduction = round((file_size - estimated_size) / file_size * 100, 1)
    reduction = max(0.0, reduction)

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=reduction == 0,
        confidence="high",
    )


def _create_sample(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    fmt: ImageFormat,
) -> bytes:
    """Resize image and encode with minimal compression.

    Minimal compression ensures the optimizer always has room to work,
    preventing false "already optimized" results on the sample.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    buf = io.BytesIO()

    if fmt == ImageFormat.JPEG:
        if sample.mode not in ("RGB", "L"):
            sample = sample.convert("RGB")
        sample.save(buf, format="JPEG", quality=100)
    elif fmt in (ImageFormat.PNG, ImageFormat.APNG):
        sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.WEBP:
        sample.save(buf, format="WEBP", lossless=True)
    elif fmt == ImageFormat.GIF:
        if sample.mode != "P":
            sample = sample.quantize(256)
        sample.save(buf, format="GIF")
    elif fmt == ImageFormat.TIFF:
        sample.save(buf, format="TIFF", compression="raw")
    elif fmt == ImageFormat.BMP:
        if sample.mode not in ("RGB", "L", "P"):
            sample = sample.convert("RGB")
        sample.save(buf, format="BMP")
    elif fmt == ImageFormat.AVIF:
        try:
            sample.save(buf, format="AVIF", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.HEIC:
        try:
            sample.save(buf, format="HEIF", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.JXL:
        try:
            sample.save(buf, format="JXL", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    else:
        sample.save(buf, format="PNG", compress_level=0)

    return buf.getvalue()


async def estimate_from_thumbnail(
    thumbnail_data: bytes,
    original_file_size: int,
    original_width: int,
    original_height: int,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate using a pre-downsized thumbnail (for large images).

    Used when the original image is >= 10MB and a CDN thumbnail is available.
    The thumbnail is compressed with the actual optimizer and BPP is
    extrapolated to the original dimensions.
    """
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(thumbnail_data)
    original_pixels = original_width * original_height

    # Decode thumbnail for pixel count
    img = await asyncio.to_thread(_open_image, thumbnail_data)
    thumb_width, thumb_height = img.size
    thumb_pixels = thumb_width * thumb_height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Compress thumbnail with actual optimizer
    result = await optimize_image(thumbnail_data, config)

    if result.method == "none":
        return EstimateResponse(
            original_size=original_file_size,
            original_format=fmt.value,
            dimensions={"width": original_width, "height": original_height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=original_file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="medium",
        )

    # Extrapolate BPP
    thumb_output_bpp = result.optimized_size * 8 / thumb_pixels
    estimated_size = int(thumb_output_bpp * original_pixels / 8)
    estimated_size = min(estimated_size, original_file_size)

    reduction = round((original_file_size - estimated_size) / original_file_size * 100, 1)
    reduction = max(0.0, reduction)

    return EstimateResponse(
        original_size=original_file_size,
        original_format=fmt.value,
        dimensions={"width": original_width, "height": original_height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=reduction == 0,
        confidence="medium",  # CDN thumbnail may have re-compression artifacts
    )


def _classify_potential(reduction: float) -> str:
    """Classify reduction percentage into potential category."""
    if reduction >= 30:
        return "high"
    elif reduction >= 10:
        return "medium"
    return "low"


def _get_color_type(img: Image.Image) -> str | None:
    """Map Pillow mode to color type string."""
    return {
        "RGB": "rgb",
        "RGBA": "rgba",
        "P": "palette",
        "L": "grayscale",
        "LA": "grayscale",
        "1": "grayscale",
    }.get(img.mode)


def _get_bit_depth(img: Image.Image) -> int | None:
    """Extract bit depth from Pillow image."""
    if img.mode == "1":
        return 1
    return img.info.get("bits") or 8
