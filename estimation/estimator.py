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
JPEG_SAMPLE_MAX_WIDTH = 1200  # JPEG needs larger samples for accurate BPP scaling
LOSSY_SAMPLE_MAX_WIDTH = 800  # HEIC/AVIF/JXL also need larger samples
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
    """Open image in Pillow and fully load pixel data into memory."""
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
    """Downsample, compress sample, extrapolate BPP to full image."""
    original_pixels = width * height

    # JPEG uses a larger sample (1200px) because JPEG BPP doesn't scale
    # linearly — small samples have proportionally more header overhead and
    # less DCT block coherence, inflating BPP relative to the full image.
    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH

    # Proportional resize
    ratio = max_width / width
    sample_width = max_width
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

    # JPEG: encode sample directly at target quality (bypasses optimizer pipeline
    # whose output-never-larger gate breaks on small samples with header overhead)
    if fmt == ImageFormat.JPEG:
        output_bpp, method = await asyncio.to_thread(
            _jpeg_sample_bpp, img, sample_width, sample_height, config
        )
        estimated_size = int(output_bpp * original_pixels / 8)
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
            method=method,
            already_optimized=reduction == 0,
            confidence="high",
        )

    # HEIC/AVIF/JXL: same pattern as JPEG — direct encode at target quality
    if fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL):
        bpp_fn = {
            ImageFormat.HEIC: _heic_sample_bpp,
            ImageFormat.AVIF: _avif_sample_bpp,
            ImageFormat.JXL: _jxl_sample_bpp,
        }[fmt]

        output_bpp, method = await asyncio.to_thread(
            bpp_fn, img, sample_width, sample_height, config
        )
        estimated_size = int(output_bpp * original_pixels / 8)
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
            method=method,
            already_optimized=reduction == 0,
            confidence="high",
        )

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


def _jpeg_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JPEG sample at target quality and return output BPP.

    Uses a larger sample (1200px) than other formats to ensure JPEG BPP
    scales accurately to the full image. Bypasses the optimizer pipeline
    whose output-never-larger gate causes false "already optimized" results.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "L"):
        sample = sample.convert("RGB")

    buf = io.BytesIO()
    save_kwargs = {
        "format": "JPEG",
        "quality": config.quality,
        "optimize": True,
    }
    if config.quality < 70:
        save_kwargs["progressive"] = True

    sample.save(buf, **save_kwargs)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "pillow_jpeg")


def _heic_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a HEIC sample at target quality and return output BPP."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA"):
        sample = sample.convert("RGB")

    heic_quality = max(30, min(90, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="HEIF", quality=heic_quality)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "heic-reencode")


def _avif_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode an AVIF sample at target quality and return output BPP."""
    import pillow_avif  # noqa: F401

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA"):
        sample = sample.convert("RGB")

    avif_quality = max(30, min(90, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="AVIF", quality=avif_quality, speed=6)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "avif-reencode")


def _jxl_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JXL sample at target quality and return output BPP."""
    try:
        import pillow_jxl  # noqa: F401
    except ImportError:
        import jxlpy  # noqa: F401

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA", "L"):
        sample = sample.convert("RGB")

    jxl_quality = max(30, min(95, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="JXL", quality=jxl_quality)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "jxl-reencode")


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
