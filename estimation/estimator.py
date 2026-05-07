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
from dataclasses import dataclass
from typing import Literal

from PIL import Image

from config import settings
from estimation.jpeg_header import JpegHeader, estimate_source_quality_lsm, parse_jpeg_header
from estimation.models import (
    Loaded,
    LoadedHeader,
    LoadedJpeg,
    LoadFailed,
    load_jpeg_header_model,
    load_png_header_model,
    load_png_model,
)
from estimation.png_features import extract_png_features
from estimation.png_header import PngHeader, parse_png_header
from optimizers.png import LARGE_MP_THRESHOLD
from optimizers.router import optimize_image
from optimizers.utils import clamp_quality
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

logger = logging.getLogger("pare.estimation")


# ---------------------------------------------------------------------------
# Fitted-BPP result union  (consensus #1 — result union, not exception)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FittedBpp:
    """Fitted model predicted a BPP value successfully."""

    bpp: float


@dataclass(frozen=True, slots=True)
class FittedFallback:
    """Fitted model could not predict; caller should fall back to sample path."""

    reason: Literal[
        "mode_unsupported_or_oob",
        "model_load_failed",
        "prediction_oob",
        "prediction_disagreement",
        "internal_error",
    ]


def _resolve_estimate_strategy(fmt: ImageFormat) -> str:
    """Return the estimation strategy for *fmt*.

    Reads ``settings.fitted_estimator_mode`` at call time so that
    ``monkeypatch.setattr("estimation.estimator.settings.fitted_estimator_mode", ...)``
    in tests takes effect without reloading the module (consensus #10).

    Strategies:
    - ``"png_header_only"`` — header-only path (replaces "fitted" for PNG).
    - ``"jpeg_header_only"`` — header-only path for JPEG.
    - ``"sample"`` — legacy direct-encode-sample / generic-fallback path.
    """
    if settings.fitted_estimator_mode != "active":
        return "sample"
    if fmt == ImageFormat.PNG:
        return "png_header_only"  # supersedes the old thumbnail-based "fitted" path
    if fmt == ImageFormat.JPEG:
        return "jpeg_header_only"
    return "sample"


def _png_fitted_bpp(
    img: "Image.Image",
    orig_w: int,
    orig_h: int,
    quality: int,
    orig_size: int = 0,
) -> FittedBpp | FittedFallback:
    """Apply the fitted PNG BPP model to *img*.

    Superseded by ``_png_header_only_bpp`` when ``fitted_estimator_mode='active'``.
    Kept for safety; will be removed in a follow-up if header-only is stable.

    Synchronous because feature extraction (PIL resize + scipy Sobel) is CPU-bound;
    callers in async context wrap this in ``asyncio.to_thread()`` per the project's
    async discipline (see ``CLAUDE.md``).

    Steps
    -----
    1. Extract features (returns None on unsupported mode or OOB pixel count).
    2. Load the PNG model (lru-cached; never raises).
    3. Standardise features.
    4. Apply piecewise-linear knot columns (log10_unique_colors, q50, q70).
    5. Compute predicted BPP via dot-product.
    6. Sanity-check the prediction (basic OOB + content-aware ratio gate).
    """
    try:
        return _png_fitted_bpp_inner(img, orig_w, orig_h, quality, orig_size)
    except Exception as exc:
        logger.warning("png fitted estimator internal error: %s", exc, exc_info=True)
        return FittedFallback(reason="internal_error")


def _png_fitted_bpp_inner(
    img: "Image.Image",
    orig_w: int,
    orig_h: int,
    quality: int,
    orig_size: int = 0,
) -> FittedBpp | FittedFallback:
    """Inner implementation — called by ``_png_fitted_bpp`` which wraps in try/except."""
    import math

    import numpy as np

    # 1. Feature extraction (pass orig_size for input_bpp)
    features = extract_png_features(img, orig_w, orig_h, quality, orig_size)
    if features is None:
        return FittedFallback(reason="mode_unsupported_or_oob")

    # 2. Load model
    match load_png_model():
        case Loaded(model=model):
            pass
        case LoadFailed():
            return FittedFallback(reason="model_load_failed")

    # 3. Build feature vector in model's declared order
    feature_values: dict[str, float] = {
        "has_alpha": float(features.has_alpha),
        "log10_unique_colors": features.log10_unique_colors,
        "mean_sobel": features.mean_sobel,
        "edge_density": features.edge_density,
        "quality": float(features.quality),
        "log10_orig_pixels": features.log10_orig_pixels,
        "input_bpp": features.input_bpp,
    }
    x_raw = np.array([feature_values[name] for name in model.features], dtype=np.float64)

    # 4. Standardise
    mean_ = np.array(model.scaler["mean"], dtype=np.float64)
    scale_ = np.array(model.scaler["scale"], dtype=np.float64)
    x_scaled = (x_raw - mean_) / scale_

    # 5. Piecewise-linear knots (raw feature values, not standardised)
    try:
        knot_lc_idx = model.features.index("log10_unique_colors")
        quality_idx = model.features.index("quality")
    except ValueError:
        return FittedFallback(reason="model_load_failed")

    log10_uc = x_raw[knot_lc_idx]
    knot_lc_val = max(0.0, log10_uc - model.knot_log10_unique_colors)

    quality_raw = x_raw[quality_idx]
    knot_q50_val = max(0.0, quality_raw - model.knot_q50)
    knot_q70_val = max(0.0, quality_raw - model.knot_q70)

    # 6. Predict: intercept + betas @ x_scaled + knot contributions
    betas = np.array(model.coefficients["betas"], dtype=np.float64)
    predicted_bpp = (
        model.coefficients["intercept"]
        + float(betas @ x_scaled)
        + model.coefficients["knot_beta"] * knot_lc_val
        + model.coefficients["knot_q50_beta"] * knot_q50_val
        + model.coefficients["knot_q70_beta"] * knot_q70_val
    )

    # 7. Post-prediction sanity — basic finite/range check
    if not math.isfinite(predicted_bpp) or predicted_bpp < 0.001 or predicted_bpp > 32.0:
        return FittedFallback(reason="prediction_oob")

    # 8. Content-aware ratio gate: output_bpp / input_bpp must be plausible.
    # Only applies when input_bpp is known (orig_size > 0).
    if features.input_bpp > 0.0:
        # Maximum plausible compression ratio by quality regime:
        # q < 50: pngquant caps max_colors=64, 20× max reduction → min ratio 0.05
        # 50 <= q < 70: max_colors=256, 10× max reduction → min ratio 0.10
        # q >= 70: lossless only, 2.5× max reduction → min ratio 0.40
        if quality < 50:
            min_ratio = 0.05
        elif quality < 70:
            min_ratio = 0.10
        else:
            min_ratio = 0.40
        max_ratio = 1.10  # output should not exceed input by more than 10%

        ratio = predicted_bpp / features.input_bpp
        if ratio < min_ratio or ratio > max_ratio:
            return FittedFallback(reason="prediction_disagreement")

    return FittedBpp(bpp=predicted_bpp)


# ---------------------------------------------------------------------------
# Header-only inference result union
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeaderOnlyBpp:
    """Header-only model predicted a BPP value successfully."""

    bpp: float


@dataclass(frozen=True, slots=True)
class HeaderOnlyFallback:
    """Header-only model could not predict; caller should fall back."""

    reason: Literal[
        "header_parse_error",
        "feature_oob",
        "prediction_oob",
        "model_load_failed",
        "internal_error",
        # JPEG-specific:
        "lossless_jpeg",
        "non_standard_components",
        "non_default_color_transform",
        "missing_chroma_table",
        "custom_quantization",
    ]


# Hard input-BPP caps — values above these indicate non-standard encodings.
_PNG_MAX_INPUT_BPP = 64.0  # e.g. 16-bit RGBA theoretical max = 64 bpp
_JPEG_MAX_INPUT_BPP = 24.0  # 8-bit YCbCr max


def _min_ratio_for_quality(quality: int) -> float:
    """Return the minimum plausible output/input BPP ratio for a given quality setting.

    Mirrors the ratio gate used in ``_png_fitted_bpp_inner``; reused by both
    PNG and JPEG header-only helpers.
    """
    if quality < 50:
        return 0.05
    elif quality < 70:
        return 0.10
    else:
        return 0.40


def _png_header_only_bpp(
    header: PngHeader,
    file_size: int,
    quality: int,
) -> HeaderOnlyBpp | HeaderOnlyFallback:
    """Predict PNG output BPP from header + file size alone. No image decode.

    Features: [has_alpha, quality, log10_orig_pixels, input_bpp]
    Model: linear with q50 and q70 piecewise-linear knots.

    Wraps the body in ``try/except`` returning HeaderOnlyFallback("internal_error")
    so the caller never sees an exception.
    """
    try:
        return _png_header_only_bpp_inner(header, file_size, quality)
    except Exception as exc:
        logger.warning("header-only png internal error: %s", exc, exc_info=True)
        return HeaderOnlyFallback(reason="internal_error")


def _png_header_only_bpp_inner(
    header: PngHeader,
    file_size: int,
    quality: int,
) -> HeaderOnlyBpp | HeaderOnlyFallback:
    """Inner implementation — called by ``_png_header_only_bpp`` which wraps in try/except."""
    import numpy as np

    original_pixels = header.width * header.height

    # Pixel cap: reject images that would exceed the memory budget
    if original_pixels > settings.max_image_pixels:
        return HeaderOnlyFallback(reason="feature_oob")

    input_bpp = file_size * 8 / original_pixels if original_pixels > 0 else 0.0

    # Sanity: reject implausibly high input BPP
    if input_bpp <= 0.0 or input_bpp > _PNG_MAX_INPUT_BPP:
        return HeaderOnlyFallback(reason="feature_oob")

    # Load model (lru-cached, never raises)
    match load_png_header_model():
        case LoadedHeader(model=model):
            pass
        case LoadFailed():
            return HeaderOnlyFallback(reason="model_load_failed")

    has_alpha = float(header.has_alpha)
    log10_orig_pixels = math.log10(original_pixels) if original_pixels > 0 else 0.0

    # Build feature vector in model's declared order: [has_alpha, quality, log10_orig_pixels, input_bpp]
    x_raw = np.array([has_alpha, float(quality), log10_orig_pixels, input_bpp], dtype=np.float64)

    # Standardise
    mean_ = np.array(model.scaler["mean"], dtype=np.float64)
    scale_ = np.array(model.scaler["scale"], dtype=np.float64)
    x_scaled = (x_raw - mean_) / scale_

    # Piecewise-linear quality knots (on raw quality value)
    knot_q50_val = max(0.0, float(quality) - model.knot_q50)
    knot_q70_val = max(0.0, float(quality) - model.knot_q70)

    # Predict: intercept + betas @ x_scaled + knot contributions
    betas = np.array(model.coefficients["betas"], dtype=np.float64)
    predicted_bpp = (
        model.coefficients["intercept"]
        + float(betas @ x_scaled)
        + model.coefficients["knot_q50_beta"] * knot_q50_val
        + model.coefficients["knot_q70_beta"] * knot_q70_val
    )

    # Post-prediction clip
    if not math.isfinite(predicted_bpp) or predicted_bpp < 0.001 or predicted_bpp > 32.0:
        return HeaderOnlyFallback(reason="prediction_oob")

    # Content-aware ratio gate
    min_ratio = _min_ratio_for_quality(quality)
    ratio = predicted_bpp / input_bpp
    if ratio < min_ratio or ratio > 1.10:
        return HeaderOnlyFallback(reason="prediction_oob")

    return HeaderOnlyBpp(bpp=predicted_bpp)


def _jpeg_header_only_bpp(
    header: JpegHeader,
    file_size: int,
    quality: int,
    progressive_pref: bool,
) -> HeaderOnlyBpp | HeaderOnlyFallback:
    """Predict JPEG output BPP from header + file size alone. No image decode.

    Features (13): target_quality, source_quality, nse, subsampling_444,
    subsampling_422, subsampling_420, progressive, log10_orig_pixels, input_bpp,
    mean_dqt_luma, std_dqt_luma, mean_dqt_chroma, std_dqt_chroma.

    Calls ``estimate_source_quality_lsm`` to derive source_quality and NSE.
    NSE < 0.85 → "custom_quantization" fallback.
    Wraps in try/except returning HeaderOnlyFallback("internal_error").
    """
    try:
        return _jpeg_header_only_bpp_inner(header, file_size, quality, progressive_pref)
    except Exception as exc:
        logger.warning("header-only jpeg internal error: %s", exc, exc_info=True)
        return HeaderOnlyFallback(reason="internal_error")


def _jpeg_header_only_bpp_inner(
    header: JpegHeader,
    file_size: int,
    quality: int,
    progressive_pref: bool,
) -> HeaderOnlyBpp | HeaderOnlyFallback:
    """Inner implementation — called by ``_jpeg_header_only_bpp`` which wraps in try/except."""
    import numpy as np

    # Route parser-flagged non-modelable conditions
    if header.fallback_reason is not None:
        reason = header.fallback_reason
        valid_reasons = {
            "lossless_jpeg",
            "non_standard_components",
            "non_default_color_transform",
            "missing_chroma_table",
        }
        if reason in valid_reasons:
            return HeaderOnlyFallback(reason=reason)  # type: ignore[arg-type]
        return HeaderOnlyFallback(reason="header_parse_error")

    original_pixels = header.width * header.height

    # Pixel cap: reject images that would exceed the memory budget
    if original_pixels > settings.max_image_pixels:
        return HeaderOnlyFallback(reason="feature_oob")

    input_bpp = file_size * 8 / original_pixels if original_pixels > 0 else 0.0

    if input_bpp <= 0.0 or input_bpp > _JPEG_MAX_INPUT_BPP:
        return HeaderOnlyFallback(reason="feature_oob")

    # LSM source-quality estimation
    if not header.dqt_luma or len(header.dqt_luma) != 64:
        return HeaderOnlyFallback(reason="header_parse_error")

    source_quality, nse = estimate_source_quality_lsm(header.dqt_luma, header.dqt_chroma)
    if nse < 0.85:
        return HeaderOnlyFallback(reason="custom_quantization")

    # Load model
    match load_jpeg_header_model():
        case LoadedJpeg(model=model):
            pass
        case LoadFailed():
            return HeaderOnlyFallback(reason="model_load_failed")

    log10_orig_pixels = math.log10(original_pixels) if original_pixels > 0 else 0.0

    # Subsampling one-hot
    sub = header.subsampling
    subsampling_444 = float(sub == "4:4:4")
    subsampling_422 = float(sub == "4:2:2")
    subsampling_420 = float(sub == "4:2:0")

    # DQT stats
    dqt_luma_arr = header.dqt_luma
    mean_dqt_luma = sum(dqt_luma_arr) / len(dqt_luma_arr)
    variance_luma = sum((v - mean_dqt_luma) ** 2 for v in dqt_luma_arr) / len(dqt_luma_arr)
    std_dqt_luma = math.sqrt(variance_luma)

    if header.dqt_chroma and len(header.dqt_chroma) == 64:
        dqt_chroma_arr = header.dqt_chroma
        mean_dqt_chroma = sum(dqt_chroma_arr) / len(dqt_chroma_arr)
        variance_chroma = sum((v - mean_dqt_chroma) ** 2 for v in dqt_chroma_arr) / len(
            dqt_chroma_arr
        )
        std_dqt_chroma = math.sqrt(variance_chroma)
    else:
        mean_dqt_chroma = 0.0
        std_dqt_chroma = 0.0

    # Build feature vector in model's declared order (13 features)
    x_raw = np.array(
        [
            float(quality),  # target_quality
            float(source_quality),  # source_quality
            nse,  # nse
            subsampling_444,
            subsampling_422,
            subsampling_420,
            float(progressive_pref),  # progressive (target preference)
            log10_orig_pixels,
            input_bpp,
            mean_dqt_luma,
            std_dqt_luma,
            mean_dqt_chroma,
            std_dqt_chroma,
        ],
        dtype=np.float64,
    )

    # Standardise
    mean_ = np.array(model.scaler["mean"], dtype=np.float64)
    scale_ = np.array(model.scaler["scale"], dtype=np.float64)
    x_scaled = (x_raw - mean_) / scale_

    # Linear prediction (no knots for JPEG header model)
    betas = np.array(model.coefficients["betas"], dtype=np.float64)
    predicted_bpp = model.coefficients["intercept"] + float(betas @ x_scaled)

    # Post-prediction clip
    if not math.isfinite(predicted_bpp) or predicted_bpp < 0.001 or predicted_bpp > 32.0:
        return HeaderOnlyFallback(reason="prediction_oob")

    # Content-aware ratio gate
    min_ratio = _min_ratio_for_quality(quality)
    ratio = predicted_bpp / input_bpp
    if ratio < min_ratio or ratio > 1.10:
        return HeaderOnlyFallback(reason="prediction_oob")

    return HeaderOnlyBpp(bpp=predicted_bpp)


# ---------------------------------------------------------------------------
# estimate_from_header_bytes — header-only result builder (URL + multipart)
# ---------------------------------------------------------------------------


async def estimate_from_header_bytes(
    data: bytes,
    total_size: int,
    fmt: ImageFormat,
    config: OptimizationConfig,
) -> EstimateResponse | None:
    """Run the header-only inference path and return an EstimateResponse.

    Used by the router for both URL-mode (Range-fetch) and multipart short-circuit.
    Returns ``None`` on any failure — caller falls through to full download / full
    estimation.  Never raises.

    Does NOT acquire the estimate semaphore (caller already did).
    Does NOT validate file size against max_file_size_bytes (total_size from
    Content-Range is trusted; worst case is a slightly-off prediction).
    """
    try:
        if fmt == ImageFormat.PNG:
            return await _estimate_from_png_header(data, total_size, config)
        elif fmt == ImageFormat.JPEG:
            return await _estimate_from_jpeg_header(data, total_size, config)
    except Exception as exc:
        logger.warning("estimate_from_header_bytes unexpected error: %s", exc, exc_info=True)
    return None


async def _estimate_from_png_header(
    data: bytes,
    total_size: int,
    config: OptimizationConfig,
) -> EstimateResponse | None:
    """PNG header-only EstimateResponse builder. Returns None on any failure."""
    header = parse_png_header(data)
    if header is None:
        return None

    result = await asyncio.to_thread(_png_header_only_bpp, header, total_size, config.quality)
    match result:
        case HeaderOnlyBpp(bpp=bpp):
            original_pixels = header.width * header.height
            estimated_size = min(int(bpp * original_pixels / 8), total_size)
            reduction = max(0.0, round((total_size - estimated_size) / total_size * 100, 1))
            has_alpha = header.has_alpha
            color_type = "rgba" if has_alpha else "rgb"
            return _build_estimate(
                file_size=total_size,
                fmt=ImageFormat.PNG,
                width=header.width,
                height=header.height,
                color_type=color_type,
                bit_depth=header.bit_depth,
                estimated_size=estimated_size,
                reduction=reduction,
                method="png_header_only",
                confidence="medium",
                path="png_header_only",
                fallback_reason=None,
            )
        case HeaderOnlyFallback():
            return None


async def _estimate_from_jpeg_header(
    data: bytes,
    total_size: int,
    config: OptimizationConfig,
) -> EstimateResponse | None:
    """JPEG header-only EstimateResponse builder. Returns None on any failure."""
    header = parse_jpeg_header(data)
    if header is None:
        return None

    if header.fallback_reason is not None:
        return None

    result = await asyncio.to_thread(
        _jpeg_header_only_bpp, header, total_size, config.quality, config.progressive_jpeg
    )
    match result:
        case HeaderOnlyBpp(bpp=bpp):
            original_pixels = header.width * header.height
            estimated_size = min(int(bpp * original_pixels / 8), total_size)
            reduction = max(0.0, round((total_size - estimated_size) / total_size * 100, 1))
            color_type = "rgb" if header.components == 3 else "grayscale"
            return _build_estimate(
                file_size=total_size,
                fmt=ImageFormat.JPEG,
                width=header.width,
                height=header.height,
                color_type=color_type,
                bit_depth=header.bit_depth,
                estimated_size=estimated_size,
                reduction=reduction,
                method="jpeg_header_only",
                confidence="medium",
                path="jpeg_header_only",
                fallback_reason=None,
            )
        case HeaderOnlyFallback():
            return None


# Register optional Pillow format plugins so Image.open() can identify all formats.
# These are imported lazily by the optimizers; the estimator needs them registered
# before calling Image.open() to decode images for dimension/animation detection.
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass  # pragma: no cover
try:
    import pillow_avif  # noqa: F401 — auto-registers on import
except ImportError:  # pragma: no cover
    pass  # pragma: no cover

if settings.enable_jxl:  # pragma: no cover
    try:  # pragma: no cover
        import pillow_jxl  # noqa: F401 — auto-registers on import  # pragma: no cover
    except ImportError:  # pragma: no cover
        try:  # pragma: no cover
            import jxlpy.JXLImagePlugin  # noqa: F401  # pragma: no cover
        except ImportError:  # pragma: no cover
            pass  # pragma: no cover

SAMPLE_MAX_WIDTH = 800  # BMP/TIFF need 800px+ to capture full-resolution redundancy
_PNG_SAMPLE_TIMEOUT = 10  # seconds — pngquant subprocess timeout for sample encoding
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
if settings.enable_jxl:  # pragma: no cover
    _EXACT_FILE_SIZE_FORMATS.add(ImageFormat.JXL)  # pragma: no cover


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

    if fmt == ImageFormat.JXL and not settings.enable_jxl:
        from exceptions import UnsupportedFormatError

        raise UnsupportedFormatError("JXL support is disabled")

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
    """Open image lazily — reads headers only, no pixel decompression.

    Pillow's Image.open() reads format/size/mode from headers without
    decompressing pixel data. Call img.load() later only when pixel
    access is needed (e.g., resize for sample-based estimation).
    """
    return Image.open(io.BytesIO(data))


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
        path="exact",
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
    # Load pixel data now — needed for img.resize() in sample creation.
    # Deferred from _open_image() to avoid loading pixels for exact-mode paths.
    await asyncio.to_thread(img.load)

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

    # --- Header-only estimator path (PNG/JPEG, mode=active) ---
    strategy = _resolve_estimate_strategy(fmt)

    if strategy == "png_header_only" and fmt == ImageFormat.PNG:
        # Parse header from raw bytes (data already loaded by caller)
        try:
            png_header = parse_png_header(data)
        except Exception as exc:
            logger.warning("png header-only: parse_png_header raised: %s", exc, exc_info=True)
            png_header = None
        if png_header is None:
            fallback_reason_val = "header_parse_error"
            logger.info("png header-only: parse failed — using direct_encode_sample")
        else:
            ho_result = await asyncio.to_thread(
                _png_header_only_bpp, png_header, file_size, config.quality
            )
            match ho_result:
                case HeaderOnlyBpp(bpp=bpp):
                    estimated_size = min(int(bpp * original_pixels / 8), file_size)
                    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))
                    return _build_estimate(
                        file_size,
                        fmt,
                        width,
                        height,
                        color_type,
                        bit_depth,
                        estimated_size,
                        reduction,
                        "png_header_only",
                        confidence="medium",
                        path="png_header_only",
                        fallback_reason=None,
                    )
                case HeaderOnlyFallback(reason=reason):
                    fallback_reason_val = reason
                    logger.info(
                        "png header-only fell back: %s — using direct_encode_sample", reason
                    )

        # Fall through to direct_encode_sample with fallback_reason populated
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
                fallback_reason=fallback_reason_val,
            )

    elif strategy == "jpeg_header_only" and fmt == ImageFormat.JPEG:
        try:
            jpeg_header = parse_jpeg_header(data)
        except Exception as exc:
            logger.warning("jpeg header-only: parse_jpeg_header raised: %s", exc, exc_info=True)
            jpeg_header = None
        if jpeg_header is None:
            fallback_reason_val = "header_parse_error"
            logger.info("jpeg header-only: parse failed — using direct_encode_sample")
        elif jpeg_header.fallback_reason is not None:
            fallback_reason_val = jpeg_header.fallback_reason
            logger.info(
                "jpeg header-only: parser flagged %s — using direct_encode_sample",
                fallback_reason_val,
            )
        else:
            ho_result = await asyncio.to_thread(
                _jpeg_header_only_bpp,
                jpeg_header,
                file_size,
                config.quality,
                config.progressive_jpeg,
            )
            match ho_result:
                case HeaderOnlyBpp(bpp=bpp):
                    estimated_size = min(int(bpp * original_pixels / 8), file_size)
                    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))
                    return _build_estimate(
                        file_size,
                        fmt,
                        width,
                        height,
                        color_type,
                        bit_depth,
                        estimated_size,
                        reduction,
                        "jpeg_header_only",
                        confidence="medium",
                        path="jpeg_header_only",
                        fallback_reason=None,
                    )
                case HeaderOnlyFallback(reason=reason):
                    fallback_reason_val = reason
                    logger.info(
                        "jpeg header-only fell back: %s — using direct_encode_sample", reason
                    )

        # Fall through to direct_encode_sample
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
                fallback_reason=fallback_reason_val,
            )

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
            path="generic_fallback_sample",
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
        path="generic_fallback_sample",
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
    fallback_reason: str | None = None,
) -> EstimateResponse:
    """Encode a sample with bpp_fn and extrapolate BPP to full image size."""
    # PNG sample BPP needs original dimensions to mirror the optimizer's dimension-aware
    # oxipng level cap (the sample itself is small, but the level decision must reflect
    # the original image's pixel count so estimation matches the real run).
    if fmt in (ImageFormat.PNG, ImageFormat.APNG):
        output_bpp, method = await asyncio.to_thread(
            bpp_fn, img, sample_width, sample_height, config, width, height
        )
    else:
        output_bpp, method = await asyncio.to_thread(
            bpp_fn, img, sample_width, sample_height, config
        )

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
        path="direct_encode_sample",
        fallback_reason=fallback_reason,
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
        import jxlpy.JXLImagePlugin  # noqa: F401

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

    Matches the WebP optimizer's Pillow path: lossy encode at target quality.
    Method mirrors WebpOptimizer._webp_method — 4 for HIGH (quality < 50),
    3 for MEDIUM/LOW (quality >= 50).
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA", "L"):
        sample = sample.convert("RGB")

    method = 4 if config.quality < 50 else 3
    buf = io.BytesIO()
    sample.save(buf, format="WEBP", quality=config.quality, method=method)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, f"pillow-m{method}")


def _png_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
    orig_width: int = 0,
    orig_height: int = 0,
) -> tuple[float, str]:
    """Encode a PNG sample and return output BPP.

    For lossy mode (quality < 70 with png_lossy=True): runs actual pngquant
    on the sample (matching the optimizer pipeline), then oxipng.  Falls back
    to Pillow palette quantization if pngquant is not installed.
    For lossless mode: encodes with Pillow then runs oxipng.

    orig_width/orig_height: original image dimensions (before downsampling).
    Used to mirror the optimizer's dimension-aware oxipng level cap — the level
    decision must reflect the original image's pixel count so estimation matches
    what the real optimizer will produce, not the sample's much-smaller area.
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
        # Sync intentionally — this helper runs inside asyncio.to_thread (see _bpp_to_estimate)
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
                timeout=_PNG_SAMPLE_TIMEOUT,
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

    # Mirror optimizer's dimension-aware level cap (operates on the original image,
    # not the sample, so estimation matches the actual run).
    oxipng_level = (
        2 if (orig_width * orig_height > LARGE_MP_THRESHOLD or config.quality >= 70) else 4
    )
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

if settings.enable_jxl:  # pragma: no cover
    _DIRECT_ENCODE_BPP_FNS[ImageFormat.JXL] = _jxl_sample_bpp  # pragma: no cover


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
            path="direct_encode_sample",
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
        path="direct_encode_sample",
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
    path: str | None = None,
    fallback_reason: str | None = None,
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
        path=path,
        fallback_reason=fallback_reason,
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
