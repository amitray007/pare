"""Sample-based estimation engine.

Instead of heuristic prediction, this module compresses a downsized sample
of the image using the actual optimizers and extrapolates BPP (bits per pixel)
to the full image size.

For small images (<150K pixels), SVG, and animated formats, it compresses the
full file for an exact result.
"""

import asyncio
import io
import logging
import math
import subprocess

from PIL import Image

from config import settings
from optimizers.router import optimize_image
from optimizers.utils import clamp_quality
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

logger = logging.getLogger("pare.estimation")

# Register optional Pillow format plugins so Image.open() can identify all formats.
# These are imported lazily by the optimizers; the estimator needs them registered
# before calling Image.open() to decode images for dimension/animation detection.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:
    pass
try:
    import pillow_avif  # noqa: F401 — auto-registers on import
except ImportError:
    pass

if settings.enable_jxl:
    try:
        import pillow_jxl  # noqa: F401 — auto-registers on import
    except ImportError:
        try:
            import jxlpy  # noqa: F401
        except ImportError:
            pass

SAMPLE_MAX_WIDTH = 800  # BMP/TIFF need 800px+ to capture full-resolution redundancy
JPEG_SAMPLE_MAX_WIDTH = 1200  # JPEG needs larger samples for accurate BPP scaling
LOSSY_SAMPLE_MAX_WIDTH = 800  # HEIC/AVIF/JXL also need larger samples
EXACT_PIXEL_THRESHOLD = 150_000  # ~390x390 pixels
EXACT_FILE_SIZE_THRESHOLD = (
    1_000_000  # 1MB — files below this compress quickly enough for exact mode
)

# Only these formats' optimizers implement max_reduction (binary search for quality).
# Other formats ignore max_reduction, so the estimator must not cap them either.
_MAX_REDUCTION_FORMATS = {ImageFormat.JPEG, ImageFormat.WEBP}

# These already-compressed formats benefit from exact mode at small file sizes because
# BPP extrapolation from decoded-and-resampled pixels is unreliable — the sample loses
# the original encoding state, producing artificially low BPP that doesn't match
# re-encoding the full original.
_EXACT_FILE_SIZE_FORMATS = {
    ImageFormat.JPEG,
    ImageFormat.WEBP,
    ImageFormat.HEIC,
    ImageFormat.AVIF,
}
if settings.enable_jxl:
    _EXACT_FILE_SIZE_FORMATS.add(ImageFormat.JXL)


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

    # PNG with low original BPP: LANCZOS downsampling introduces anti-aliasing
    # noise at color boundaries, inflating sample BPP for flat-color content
    # (screenshots, graphics, solid backgrounds).  Use exact mode instead —
    # the file is small in bytes even though it has many pixels.
    if fmt in (ImageFormat.PNG, ImageFormat.APNG):
        original_bpp = file_size * 8 / original_pixels
        if original_bpp < 2.0:
            return await _estimate_exact(
                data, fmt, config, file_size, width, height, color_type, bit_depth
            )

    # GIF: files are always small and gifsicle is fast — use exact mode.
    # (Animated GIFs are already handled above.)
    # The 300px generic-fallback sample is too small for gifsicle to work
    # effectively, producing 0% estimates on images that actually compress 15-35%.
    if fmt == ImageFormat.GIF:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Already-compressed formats (JPEG, WebP, HEIC, AVIF, JXL): for smaller files,
    # use exact mode to avoid LANCZOS smoothing artifacts that inflate sample BPP
    # on already-compressed sources and to capture lossless gains (jpegtran, metadata
    # strip) that sample-based estimation misses entirely.
    if fmt in _EXACT_FILE_SIZE_FORMATS and file_size <= EXACT_FILE_SIZE_THRESHOLD:
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

    return _build_estimate(
        file_size,
        fmt,
        width,
        height,
        color_type,
        bit_depth,
        result.optimized_size,
        reduction,
        result.method,
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

    # Proportional resize (never upscale — cap at original dimensions)
    max_width = min(max_width, width)
    ratio = max_width / width
    sample_width = max_width
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

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

    # Create sample encoded with minimal compression
    sample_data = await asyncio.to_thread(_create_sample, img, sample_width, sample_height, fmt)

    # Compress sample with the actual optimizer
    result = await optimize_image(sample_data, config)

    # If optimizer says "already optimized", propagate that
    if result.method == "none":
        return _build_estimate(
            file_size,
            fmt,
            width,
            height,
            color_type,
            bit_depth,
            file_size,
            0.0,
            "none",
        )

    # Extrapolate output BPP to original pixel count
    sample_output_bpp = result.optimized_size * 8 / sample_pixels
    estimated_size = int(sample_output_bpp * original_pixels / 8)
    estimated_size = min(estimated_size, file_size)

    reduction = round((file_size - estimated_size) / file_size * 100, 1)
    reduction = max(0.0, reduction)

    # No max_reduction cap here — this generic fallback path is only reached by
    # GIF/BMP/TIFF, whose optimizers do not implement max_reduction.

    return _build_estimate(
        file_size,
        fmt,
        width,
        height,
        color_type,
        bit_depth,
        estimated_size,
        reduction,
        result.method,
    )


async def _bpp_to_estimate(
    bpp_fn,
    img: Image.Image,
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
    """Encode a sample with bpp_fn and extrapolate BPP to full image size."""
    output_bpp, method = await asyncio.to_thread(bpp_fn, img, sample_width, sample_height, config)

    # TIFF lossless deflate/LZW: BPP scales sub-linearly with resolution because
    # larger images have more inter-pixel redundancy for the compressor to exploit.
    # The downsampled sample loses this redundancy, inflating BPP.  Apply a log-based
    # correction: at 3x downscale (2400→800) correction ≈ 0.72, at 1.5x ≈ 0.88.
    if fmt == ImageFormat.TIFF and method in ("tiff_adobe_deflate", "tiff_lzw"):
        downscale_ratio = width / sample_width
        if downscale_ratio > 1.0:
            output_bpp *= 1.0 / (1.0 + 0.35 * math.log(downscale_ratio))

    estimated_size = min(int(output_bpp * original_pixels / 8), file_size)
    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))

    # Honour max_reduction cap only for formats whose optimizers enforce it.
    if (
        fmt in _MAX_REDUCTION_FORMATS
        and config.max_reduction is not None
        and reduction > config.max_reduction
    ):
        reduction = round(config.max_reduction, 1)
        estimated_size = int(file_size * (1 - reduction / 100))

    # PNG lossless on photo content (high BPP): the sample resize smooths pixel
    # data, making oxipng achieve much better compression than on the full image.
    # In practice, lossless PNG barely compresses photos (< 5%).
    png_lossless = not (config.png_lossy and config.quality < 70)
    if (
        fmt in (ImageFormat.PNG, ImageFormat.APNG)
        and png_lossless
        and (file_size * 8 / original_pixels) > 10.0
        and reduction > 5.0
    ):
        reduction = 5.0
        estimated_size = int(file_size * (1 - reduction / 100))

    return _build_estimate(
        file_size,
        fmt,
        width,
        height,
        color_type,
        bit_depth,
        estimated_size,
        reduction,
        method,
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
    import pillow_heif  # noqa: F401 — registers HEIF plugin

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA"):
        sample = sample.convert("RGB")

    heic_quality = clamp_quality(config.quality)

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

    avif_quality = clamp_quality(config.quality)

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

    jxl_quality = clamp_quality(config.quality, hi=95)

    buf = io.BytesIO()
    sample.save(buf, format="JXL", quality=jxl_quality)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "jxl-reencode")


def _webp_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a WebP sample at target quality and return output BPP.

    Matches the WebP optimizer's Pillow path: lossy encode at target quality
    with method=4 for good compression.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA", "L"):
        sample = sample.convert("RGB")

    buf = io.BytesIO()
    sample.save(buf, format="WEBP", quality=config.quality, method=4)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "pillow")


def _png_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a PNG sample and return output BPP.

    For lossy mode (quality < 70 with png_lossy=True): runs actual pngquant
    on the sample (matching the optimizer pipeline), then oxipng.  Falls back
    to Pillow palette quantization if pngquant is not installed.
    For lossless mode: encodes with Pillow then runs oxipng.
    """
    import oxipng

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)

    # Lossy path: quantize to palette using actual pngquant (matching the optimizer)
    if config.png_lossy and config.quality < 70:
        max_colors = 64 if config.quality < 50 else 256
        speed = 3 if config.quality < 50 else 4

        # Encode sample to PNG for pngquant input
        if sample.mode not in ("RGB", "RGBA", "L", "P"):
            sample = sample.convert("RGBA")
        buf = io.BytesIO()
        sample.save(buf, format="PNG", compress_level=6)
        png_data = buf.getvalue()

        # Use actual pngquant for accurate palette quantization
        try:
            proc = subprocess.run(
                [
                    "pngquant",
                    str(max_colors),
                    "--quality",
                    f"1-{config.quality}",
                    "--speed",
                    str(speed),
                    "-",
                    "--output",
                    "-",
                ],
                input=png_data,
                capture_output=True,
                timeout=10,
            )
            if proc.returncode == 0 and proc.stdout:
                png_data = proc.stdout
            # exit code 99 = quality threshold not met; keep original png_data
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # pngquant not available — fall back to Pillow quantize
            if sample.mode == "RGBA":
                quantized = sample.quantize(max_colors)
            elif sample.mode != "P":
                quantized = sample.convert("RGB").quantize(max_colors)
            else:
                quantized = sample
            buf = io.BytesIO()
            quantized.save(buf, format="PNG", compress_level=6)
            png_data = buf.getvalue()

        method = "pngquant + oxipng"
    else:
        buf = io.BytesIO()
        sample.save(buf, format="PNG", compress_level=6)
        png_data = buf.getvalue()
        method = "oxipng"

    # oxipng matches what the actual optimizer uses
    oxipng_level = 4 if config.quality < 70 else 2
    optimized = oxipng.optimize_from_memory(png_data, level=oxipng_level)
    output_size = len(optimized)
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, method)


def _tiff_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a TIFF sample with multiple compression methods, pick smallest.

    Mirrors the TIFF optimizer's method selection:
    - All presets: tiff_adobe_deflate, tiff_lzw (lossless)
    - quality < 70: also tiff_jpeg (lossy JPEG-in-TIFF)
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)

    candidates: list[tuple[int, str]] = []

    # Lossless methods (all presets)
    for compression in ("tiff_adobe_deflate", "tiff_lzw"):
        buf = io.BytesIO()
        try:
            sample.save(buf, format="TIFF", compression=compression)
            candidates.append((buf.tell(), compression))
        except Exception as exc:
            logger.debug("TIFF sample encode with %s failed: %s", compression, exc)

    # Lossy JPEG-in-TIFF (quality < 70, RGB/L only — matches optimizer)
    if config.quality < 70 and sample.mode in ("RGB", "L"):
        buf = io.BytesIO()
        try:
            sample.save(buf, format="TIFF", compression="tiff_jpeg", quality=config.quality)
            candidates.append((buf.tell(), "tiff_jpeg"))
        except Exception as exc:
            logger.debug("TIFF sample encode with tiff_jpeg failed: %s", exc)

    if not candidates:
        buf = io.BytesIO()
        sample.save(buf, format="TIFF", compression="raw")
        candidates.append((buf.tell(), "tiff_raw"))

    best_size, best_method = min(candidates, key=lambda x: x[0])
    sample_pixels = sample_width * sample_height

    return (best_size * 8 / sample_pixels, best_method)


# Direct-encode path: encode sample at target quality, bypassing the
# optimizer pipeline whose output-never-larger gate breaks on small samples.
# Maps format -> BPP helper function.
_DIRECT_ENCODE_BPP_FNS = {
    ImageFormat.JPEG: _jpeg_sample_bpp,
    ImageFormat.HEIC: _heic_sample_bpp,
    ImageFormat.AVIF: _avif_sample_bpp,
    ImageFormat.WEBP: _webp_sample_bpp,
    ImageFormat.PNG: _png_sample_bpp,
    ImageFormat.APNG: _png_sample_bpp,
    ImageFormat.TIFF: _tiff_sample_bpp,
}

if settings.enable_jxl:
    _DIRECT_ENCODE_BPP_FNS[ImageFormat.JXL] = _jxl_sample_bpp


def _create_sample(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    fmt: ImageFormat,
) -> bytes:
    """Resize image and encode with minimal compression.

    Used only for formats without a direct-encode BPP helper (GIF, BMP, TIFF).
    Minimal compression ensures the optimizer always has room to work,
    preventing false "already optimized" results on the sample.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    buf = io.BytesIO()

    if fmt == ImageFormat.GIF:
        if sample.mode != "P":
            sample = sample.quantize(256)
        sample.save(buf, format="GIF")
    elif fmt == ImageFormat.TIFF:
        sample.save(buf, format="TIFF", compression="raw")
    elif fmt == ImageFormat.BMP:
        if sample.mode not in ("RGB", "L", "P"):
            sample = sample.convert("RGB")
        sample.save(buf, format="BMP")
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
        return _build_estimate(
            original_file_size,
            fmt,
            original_width,
            original_height,
            color_type,
            bit_depth,
            original_file_size,
            0.0,
            "none",
            confidence="medium",
        )

    # Extrapolate BPP
    thumb_output_bpp = result.optimized_size * 8 / thumb_pixels
    estimated_size = int(thumb_output_bpp * original_pixels / 8)
    estimated_size = min(estimated_size, original_file_size)

    reduction = round((original_file_size - estimated_size) / original_file_size * 100, 1)
    reduction = max(0.0, reduction)

    return _build_estimate(
        original_file_size,
        fmt,
        original_width,
        original_height,
        color_type,
        bit_depth,
        estimated_size,
        reduction,
        result.method,
        confidence="medium",
    )


def _classify_potential(reduction: float) -> str:
    """Classify reduction percentage into potential category."""
    if reduction >= 30:
        return "high"
    elif reduction >= 10:
        return "medium"
    return "low"


def _build_estimate(
    file_size: int,
    fmt: ImageFormat,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
    estimated_size: int,
    reduction: float,
    method: str,
    confidence: str = "high",
) -> EstimateResponse:
    """Build an EstimateResponse with standard field derivations."""
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
        confidence=confidence,
    )


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
    """Extract bit depth (per channel) from Pillow image."""
    mode_to_bits = {
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "LA": 8,
        "I": 32,
        "F": 32,
        "I;16": 16,
    }
    # Prefer explicit value from image metadata (e.g. PNG sBIT chunk)
    explicit = img.info.get("bits")
    if explicit:
        return explicit
    return mode_to_bits.get(img.mode, 8)
