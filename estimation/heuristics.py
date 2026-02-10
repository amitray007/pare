from dataclasses import dataclass

from estimation.header_analysis import HeaderInfo
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


@dataclass
class Prediction:
    """Estimation prediction result."""

    estimated_size: int
    reduction_percent: float
    potential: str  # "high", "medium", "low"
    method: str
    already_optimized: bool
    confidence: str  # "high", "medium", "low"


def predict_reduction(
    info: HeaderInfo, fmt: ImageFormat, config: OptimizationConfig
) -> Prediction:
    """Predict compression reduction based on format-specific heuristics."""
    dispatch = {
        ImageFormat.PNG: _predict_png,
        ImageFormat.APNG: _predict_apng,
        ImageFormat.JPEG: _predict_jpeg,
        ImageFormat.WEBP: _predict_webp,
        ImageFormat.GIF: _predict_gif,
        ImageFormat.SVG: _predict_svg,
        ImageFormat.SVGZ: _predict_svgz,
        ImageFormat.AVIF: _predict_metadata_only,
        ImageFormat.HEIC: _predict_metadata_only,
        ImageFormat.TIFF: _predict_passthrough,
        ImageFormat.BMP: _predict_passthrough,
        ImageFormat.PSD: _predict_passthrough,
    }
    predictor = dispatch.get(fmt, _predict_passthrough)
    return predictor(info, config)


def _predict_png(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """PNG heuristics — mirrors optimizer logic.

    config.png_lossy=True  → predict pngquant + oxipng (lossy quantization)
    config.png_lossy=False → predict oxipng-only (lossless recompression, 2-8%)
    """
    if not config.png_lossy:
        # Lossless only: oxipng recompression
        reduction = 5.0
        if info.has_metadata_chunks and config.strip_metadata:
            reduction += 3.0
        return Prediction(
            estimated_size=int(info.file_size * (1 - reduction / 100)),
            reduction_percent=round(reduction, 1),
            potential="low",
            method="oxipng",
            already_optimized=reduction < 3.0,
            confidence="medium",
        )

    # Lossy path: pngquant + oxipng
    if info.is_palette_mode:
        if info.color_count and info.color_count < 16:
            reduction = 15.0
            potential = "low"
        else:
            reduction = 40.0
            potential = "medium"
        already_optimized = not info.has_metadata_chunks
        method = "pngquant + oxipng"
        confidence = "medium"
    else:
        reduction, potential, method, confidence = _predict_png_by_complexity(
            info.unique_color_ratio
        )
        already_optimized = False

    if info.has_metadata_chunks and config.strip_metadata:
        reduction += 3.0

    estimated_size = int(info.file_size * (1 - reduction / 100))
    return Prediction(
        estimated_size=estimated_size,
        reduction_percent=round(reduction, 1),
        potential=potential,
        method=method,
        already_optimized=already_optimized,
        confidence=confidence,
    )


def _predict_png_by_complexity(
    color_ratio: float | None,
) -> tuple[float, str, str, str]:
    """Predict PNG reduction based on unique-color ratio.

    Low ratio = flat graphics (pngquant succeeds).
    High ratio = photographic content (pngquant exit 99, oxipng-only).

    Returns (reduction%, potential, method, confidence).
    """
    if color_ratio is None:
        return 20.0, "medium", "pngquant + oxipng", "low"

    if color_ratio < 0.005:
        return 85.0, "high", "pngquant + oxipng", "high"
    elif color_ratio < 0.05:
        return 55.0, "high", "pngquant + oxipng", "medium"
    elif color_ratio < 0.20:
        return 55.0, "high", "pngquant + oxipng", "medium"
    elif color_ratio < 0.50:
        return 30.0, "medium", "pngquant + oxipng", "low"
    else:
        return 3.0, "low", "oxipng", "medium"


def _predict_apng(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """APNG — lossless only (oxipng), limited savings."""
    reduction = 5.0 if info.has_metadata_chunks else 2.0
    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="low",
        method="oxipng",
        already_optimized=reduction < 3.0,
        confidence="low",
    )


def _predict_jpeg(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """JPEG heuristics — mirrors optimizer (try mozjpeg + jpegtran, pick smallest).

    The optimizer always tries mozjpeg at config.quality and jpegtran,
    then picks the smallest output. The estimator mirrors this decision:
    - If source quality <= target: jpegtran wins (lossless Huffman)
    - If source quality > target: mozjpeg wins with quality headroom
    """
    source_q = info.estimated_quality or 85
    target_q = config.quality
    delta = source_q - target_q

    if delta <= 0:
        # Source already at or below target — mozjpeg at higher quality
        # produces larger output, so jpegtran wins (Huffman optimization)
        reduction = 8.0
        method = "jpegtran"
        potential = "low"
    else:
        # mozjpeg re-encode wins with quality headroom
        # Piecewise linear calibrated from actual mozjpeg benchmark results
        if delta <= 5:
            base = 8 + 4.0 * delta
        elif delta <= 15:
            base = 28 + 2.5 * (delta - 5)
        elif delta <= 30:
            base = 53 + 1.8 * (delta - 15)
        else:
            base = 80 + 0.5 * (delta - 30)
        # Higher source quality = more compressible data
        source_factor = max(-5, min(10, (source_q - 85) * 0.5))
        reduction = min(92, base + source_factor)
        method = "mozjpeg"
        potential = "high" if reduction >= 40 else "medium"

    if info.has_exif and config.strip_metadata:
        reduction += 2.0

    if config.progressive_jpeg:
        reduction += 1.0

    if info.is_progressive:
        reduction *= 0.95

    already_optimized = delta <= 0 and not info.has_exif
    estimated_size = int(info.file_size * (1 - reduction / 100))

    return Prediction(
        estimated_size=estimated_size,
        reduction_percent=round(reduction, 1),
        potential=potential,
        method=method,
        already_optimized=already_optimized,
        confidence="medium",
    )


def _bpp_to_quality(bpp: float) -> int:
    """Map bits-per-pixel to estimated WebP quality.

    Piecewise linear calibrated from benchmark data (photographic content):
        bpp ~2.1 → q60, bpp ~3.0 → q80, bpp ~5.2 → q95
    """
    if bpp <= 0.1:
        return 20
    elif bpp <= 2.1:
        return int(max(20, 60 - (2.1 - bpp) * 20))
    elif bpp <= 3.0:
        return int(60 + (bpp - 2.1) / 0.9 * 20)
    elif bpp <= 5.2:
        return int(80 + (bpp - 3.0) / 2.2 * 15)
    else:
        return int(min(98, 95 + (bpp - 5.2) * 1.5))


def _predict_webp(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """WebP heuristics — mirrors optimizer (Pillow re-encode at config.quality).

    Estimates source quality from bits-per-pixel, then computes
    quality delta to predict reduction. Uses a calibrated piecewise
    model with source-quality scaling (higher source = more compressible).
    """
    w = info.dimensions.get("width", 1)
    h = info.dimensions.get("height", 1)
    pixels = max(w * h, 1)
    bpp = (info.file_size * 8) / pixels
    est_source_q = _bpp_to_quality(bpp)
    delta = est_source_q - config.quality

    if delta <= 0:
        reduction = 3.0
        potential = "low"
    else:
        # Piecewise linear base curve (calibrated for q95 source):
        # Steep initial slope — high-quality WebP has huge redundancy
        # that drops quickly, then diminishing returns at larger deltas.
        if delta <= 10:
            base = 3 + 3.5 * delta
        elif delta <= 25:
            base = 38 + 1.5 * (delta - 10)
        elif delta <= 45:
            base = 60.5 + 0.6 * (delta - 25)
        else:
            base = min(78, 72.5 + 0.2 * (delta - 45))

        # Source quality scaling: lower source quality images are already
        # compact and yield less reduction per delta point.
        # Calibrated: q95→1.0, q80→0.76, q60→0.45
        q_mult = max(0.3, min(1.1, 0.45 + (est_source_q - 60) * 0.0157))
        reduction = min(78, base * q_mult)
        potential = "high" if reduction >= 40 else "medium"

    estimated_size = int(info.file_size * (1 - reduction / 100))
    return Prediction(
        estimated_size=estimated_size,
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="pillow",
        already_optimized=delta <= 0,
        confidence="medium",
    )


def _predict_gif(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """GIF heuristics — calibrated from benchmarks."""
    if info.frame_count > 1:
        reduction = 15.0
        potential = "medium"
    else:
        reduction = 10.0
        potential = "low"

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="gifsicle",
        already_optimized=False,
        confidence="medium",
    )


def _predict_svg(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """SVG heuristics — metadata and editor bloat drive savings."""
    if info.has_metadata_chunks:
        reduction = 30.0
        potential = "high"
        already_optimized = False
    else:
        reduction = 8.0
        potential = "low"
        already_optimized = True

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="scour",
        already_optimized=already_optimized,
        confidence="medium",
    )


def _predict_svgz(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """SVGZ heuristics — already gzip-compressed, limited savings."""
    reduction = 8.0 if info.has_metadata_chunks else 5.0

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="low",
        method="scour",
        already_optimized=not info.has_metadata_chunks,
        confidence="medium",
    )


def _predict_metadata_only(
    info: HeaderInfo, config: OptimizationConfig
) -> Prediction:
    """AVIF/HEIC — metadata stripping only, minimal savings."""
    has_metadata = info.has_exif or info.has_icc_profile
    reduction = 5.0 if has_metadata else 0.0

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="low",
        method="metadata-strip",
        already_optimized=not has_metadata,
        confidence="low",
    )


def _predict_passthrough(
    info: HeaderInfo, config: OptimizationConfig
) -> Prediction:
    """TIFF/BMP/PSD — limited optimization potential."""
    return Prediction(
        estimated_size=info.file_size,
        reduction_percent=0.0,
        potential="low",
        method=f"pillow-{info.format.value}",
        already_optimized=True,
        confidence="low",
    )
