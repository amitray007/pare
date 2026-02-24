"""Sample-based estimation engine.

Instead of heuristic prediction, this module compresses a downsized sample
of the image using the actual optimizers and extrapolates BPP (bits per pixel)
to the full image size.

For small images (<150K pixels), SVG, and animated formats, it compresses the
full file for an exact result.
"""

import asyncio

import pyvips

from optimizers.router import optimize_image
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

SAMPLE_MAX_WIDTH = 300
JPEG_SAMPLE_MAX_WIDTH = 1200
LOSSY_SAMPLE_MAX_WIDTH = 800
EXACT_PIXEL_THRESHOLD = 150_000


async def estimate(
    data: bytes,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate compression savings by compressing a sample."""
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(data)
    file_size = len(data)

    # SVG/SVGZ: no pixel data — compress the whole file
    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return await _estimate_exact(data, fmt, config, file_size)

    # Decode image for dimensions and animation detection
    img = await asyncio.to_thread(_open_image, data)
    width = img.width
    height = img.height
    original_pixels = width * height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Animated images: compress full file (inter-frame redundancy matters)
    n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
    if n_pages > 1:
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


def _open_image(data: bytes) -> pyvips.Image:
    """Open image with pyvips."""
    return pyvips.Image.new_from_buffer(data, "")


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
    img: pyvips.Image,
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

    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (
        ImageFormat.HEIC,
        ImageFormat.AVIF,
        ImageFormat.JXL,
        ImageFormat.WEBP,
        ImageFormat.PNG,
        ImageFormat.APNG,
    ):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH

    max_width = min(max_width, width)
    ratio = max_width / width
    sample_width = max_width
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

    _DIRECT_ENCODE_BPP_FNS = {
        ImageFormat.JPEG: _jpeg_sample_bpp,
        ImageFormat.HEIC: _heic_sample_bpp,
        ImageFormat.AVIF: _avif_sample_bpp,
        ImageFormat.JXL: _jxl_sample_bpp,
        ImageFormat.WEBP: _webp_sample_bpp,
        ImageFormat.PNG: _png_sample_bpp,
        ImageFormat.APNG: _png_sample_bpp,
    }

    bpp_fn = _DIRECT_ENCODE_BPP_FNS.get(fmt)
    if bpp_fn is not None:
        return await _bpp_to_estimate(
            bpp_fn,
            img,
            sample_width,
            sample_height,
            config,
            original_pixels,
            file_size,
            fmt,
            width,
            height,
            color_type,
            bit_depth,
        )

    # Generic fallback: create sample, run actual optimizer
    sample_data = await asyncio.to_thread(_create_sample, img, sample_width, sample_height, fmt)
    result = await optimize_image(sample_data, config)

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

    sample_output_bpp = result.optimized_size * 8 / sample_pixels
    estimated_size = min(int(sample_output_bpp * original_pixels / 8), file_size)
    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))

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


async def _bpp_to_estimate(
    bpp_fn,
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
    original_pixels: int,
    file_size: int,
    fmt: ImageFormat,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
) -> EstimateResponse:
    output_bpp, method = await asyncio.to_thread(bpp_fn, img, sample_width, sample_height, config)
    estimated_size = min(int(output_bpp * original_pixels / 8), file_size)
    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))

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


def _jpeg_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JPEG sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    save_kwargs = {
        "Q": config.quality,
        "optimize_coding": True,
        "strip": True,
    }

    buf = sample.jpegsave_buffer(**save_kwargs)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "jpegli")


def _heic_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a HEIC sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    heic_quality = max(30, min(90, config.quality + 10))

    buf = sample.heifsave_buffer(Q=heic_quality, compression="hevc", strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "heic-reencode")


def _avif_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode an AVIF sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    avif_quality = max(30, min(90, config.quality + 10))

    buf = sample.heifsave_buffer(Q=avif_quality, compression="av1", effort=4, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "avif-reencode")


def _jxl_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JXL sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    jxl_quality = max(30, min(95, config.quality + 10))

    buf = sample.jxlsave_buffer(Q=jxl_quality, effort=7, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "jxl-reencode")


def _webp_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a WebP sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    buf = sample.webpsave_buffer(Q=config.quality, effort=4, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "pyvips-webp")


def _png_sample_bpp(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a PNG sample and return output BPP."""
    import oxipng

    scale = sample_width / img.width
    sample = img.resize(scale)

    # Lossy path: quantize to palette (simulates pngquant via libimagequant)
    if config.png_lossy and config.quality < 70:
        max_colors = 64 if config.quality < 50 else 256
        png_data = sample.pngsave_buffer(
            palette=True, Q=config.quality, colours=max_colors, dither=1.0, strip=True
        )
        method = "pngquant + oxipng"
    else:
        png_data = sample.pngsave_buffer(compression=9, effort=10, strip=True)
        method = "oxipng"

    # oxipng post-processing
    oxipng_level = 4 if config.quality < 70 else 2
    optimized = oxipng.optimize_from_memory(png_data, level=oxipng_level)
    sample_pixels = sample_width * sample_height
    return (len(optimized) * 8 / sample_pixels, method)


def _create_sample(
    img: pyvips.Image,
    sample_width: int,
    sample_height: int,
    fmt: ImageFormat,
) -> bytes:
    """Resize image and encode with minimal compression."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    if fmt == ImageFormat.GIF:
        return sample.gifsave_buffer()
    elif fmt == ImageFormat.TIFF:
        return sample.tiffsave_buffer(compression="none")
    elif fmt == ImageFormat.BMP:
        from optimizers.bmp import encode_bmp_24

        return encode_bmp_24(sample)
    else:
        return sample.pngsave_buffer(compression=0)


async def estimate_from_thumbnail(
    thumbnail_data: bytes,
    original_file_size: int,
    original_width: int,
    original_height: int,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate using a pre-downsized thumbnail (for large images)."""
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(thumbnail_data)
    original_pixels = original_width * original_height

    img = await asyncio.to_thread(_open_image, thumbnail_data)
    thumb_width = img.width
    thumb_height = img.height
    thumb_pixels = thumb_width * thumb_height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

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

    thumb_output_bpp = result.optimized_size * 8 / thumb_pixels
    estimated_size = min(int(thumb_output_bpp * original_pixels / 8), original_file_size)
    reduction = max(0.0, round((original_file_size - estimated_size) / original_file_size * 100, 1))

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
        confidence="medium",
    )


def _classify_potential(reduction: float) -> str:
    if reduction >= 30:
        return "high"
    elif reduction >= 10:
        return "medium"
    return "low"


def _get_color_type(img: pyvips.Image) -> str | None:
    """Map pyvips interpretation to color type string."""
    interp = img.interpretation
    bands = img.bands
    mapping = {
        "srgb": "rgba" if bands == 4 else "rgb",
        "rgb": "rgba" if bands == 4 else "rgb",
        "b-w": "grayscale",
        "grey16": "grayscale",
    }
    return mapping.get(interp)


def _get_bit_depth(img: pyvips.Image) -> int | None:
    """Extract bit depth (per channel) from pyvips image."""
    fmt = img.format
    depth_map = {
        "uchar": 8,
        "char": 8,
        "ushort": 16,
        "short": 16,
        "uint": 32,
        "int": 32,
        "float": 32,
        "double": 64,
    }
    return depth_map.get(fmt, 8)
