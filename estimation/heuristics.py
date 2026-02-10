from dataclasses import dataclass
from math import exp as _exp

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
        elif info.file_size < 2000:
            reduction = 30.0
            potential = "medium"
        else:
            reduction = 40.0
            potential = "medium"
        already_optimized = not info.has_metadata_chunks
        method = "pngquant + oxipng"
        confidence = "medium"
    else:
        reduction, potential, method, confidence = _predict_png_by_complexity(
            info, config
        )
        already_optimized = False

    if info.has_metadata_chunks and config.strip_metadata:
        reduction += 3.0

    # Tiny-file cap: PNG headers (signature + IHDR + IEND) are ~60 bytes
    # of fixed overhead that can't be removed. Cap reduction accordingly.
    if info.file_size < 500:
        min_png_size = 67  # signature(8) + IHDR(25) + IEND(12) + minimal IDAT(22)
        max_reduction = max(0.0, (1 - min_png_size / info.file_size) * 100)
        reduction = min(reduction, max_reduction)

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
    info: HeaderInfo,
    config: OptimizationConfig,
) -> tuple[float, str, str, str]:
    """Predict PNG reduction using oxipng probe + content heuristics.

    For small files (< 50KB), oxipng_probe_ratio is from the actual file
    (exact lossless measurement). For larger files, it's from a 64x64 crop
    (content signal only, not direct ratio predictor).

    Two-path model matching optimizer:
    1. Lossless: oxipng only -> from probe or heuristic
    2. Lossy: pngquant + oxipng -> from quantize probe, gated by content type
    Picks the better path (max reduction).

    Returns (reduction%, potential, method, confidence).
    """
    opr = info.oxipng_probe_ratio
    qpr = info.png_quantize_ratio
    lpr = info.png_lossy_proxy_ratio
    fpr = info.flat_pixel_ratio
    cr = info.unique_color_ratio
    is_full_file_probe = info.file_size < 50000

    if opr is None and cr is None:
        return 20.0, "medium", "pngquant + oxipng", "low"

    is_flat = fpr is not None and fpr > 0.75
    is_photo = cr is not None and cr > 0.50 and fpr is not None and fpr < 0.50

    # --- Lossless path ---
    if opr is not None and is_full_file_probe:
        # Exact measurement from actual file
        lossless_reduction = (1.0 - opr) * 100.0
    elif is_photo:
        lossless_reduction = 3.0
    elif opr is not None:
        # Crop probe: discount for size-dependent scaling mismatch
        lossless_reduction = (1.0 - opr) * 100.0 * 0.6
    else:
        lossless_reduction = 5.0

    # --- Lossy path ---
    lossy_reduction = 0.0

    if lpr is not None and is_full_file_probe:
        # Direct lossy proxy measurement (quantize + oxipng on actual image).
        # This is the most accurate predictor — use it directly, gated by
        # content type and quality to account for pngquant exit-99 cases.
        lossy_proxy_reduction = (1.0 - lpr) * 100.0

        if is_flat:
            # Flat content: pngquant almost always succeeds (palette-friendly).
            # Gradients have qpr > 2.0 and lossy proxy is typically worse
            # than lossless, so max() below handles them naturally.
            lossy_reduction = lossy_proxy_reduction
        elif is_photo:
            # Photos: pngquant only succeeds at aggressive quality settings.
            if config.quality <= 50:
                lossy_reduction = lossy_proxy_reduction
        else:
            # Graphics/artwork: pngquant works at most quality levels
            lossy_reduction = lossy_proxy_reduction

    elif is_full_file_probe and qpr is not None:
        # Small file with thumbnail probes but no lossy proxy (> 250K pixels).
        # Fall back to thumbnail-based estimation with content gating.
        if is_flat:
            # Flat content without proxy: can't predict pngquant bonus reliably.
            # Use lossless only (max() below handles this naturally).
            lossy_reduction = 0.0
        elif is_photo:
            if config.quality <= 50 and qpr < 0.60:
                lossy_reduction = (1.0 - qpr) * 100.0
        elif qpr < 0.70:
            lossy_reduction = (1.0 - qpr) * 100.0

    elif not is_full_file_probe:
        # Large files: heuristic for lossy path
        if is_flat or is_photo:
            lossy_reduction = 0.0
        elif cr is not None and cr < 0.005:
            lossy_reduction = 90.0
        elif cr is not None and cr < 0.20:
            lossy_reduction = 55.0
        elif qpr is not None and qpr < 0.50:
            lossy_reduction = 55.0

    # Pick the better path (matches optimizer: picks smallest output)
    if lossy_reduction > lossless_reduction:
        reduction = lossy_reduction
        method = "pngquant + oxipng"
    else:
        reduction = lossless_reduction
        method = "oxipng"

    reduction = max(0.0, min(95.0, reduction))

    if lpr is not None and is_full_file_probe:
        confidence = "high"
    elif opr is not None and is_full_file_probe:
        confidence = "high"
    elif opr is not None:
        confidence = "medium"
    else:
        confidence = "low"

    potential = "high" if reduction >= 40 else ("medium" if reduction >= 15 else "low")

    return reduction, potential, method, confidence


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
    then picks the smallest output. This model predicts both methods and
    picks the one with the larger reduction, matching optimizer behavior.

    Calibrated from benchmark data across source qualities 40-98 and
    target qualities 40/60/80.
    """
    source_q = info.estimated_quality or 85
    target_q = config.quality
    delta = source_q - target_q

    # Jpegtran prediction: lossless Huffman optimization.
    # Base reduction scales with (100-sq) since lower quality = more zeros.
    # High-quality sources (q>90) have an exponential bonus because near-1
    # quantization values create massive Huffman redundancy.
    jpegtran_reduction = 6.75 + 0.194 * (100 - source_q)
    if source_q > 90:
        jpegtran_reduction += 0.668 * _exp(0.293 * (source_q - 90))

    # Mozjpeg prediction: trellis quantization + optimized Huffman.
    # For delta>0: encoder bonus is ~28% (calibrated with piecewise linear
    # delta curve). For delta<=0: bonus depends on source quality — high-q
    # sources have less room for trellis improvement.
    if delta > 0:
        encoder_bonus = 28.0
        sq_factor = 1.0 + (source_q - 75) * 0.008
        s1 = 1.1 + (source_q - 75) * 0.015
        if delta <= 8:
            extra = s1 * delta
        elif delta <= 20:
            base_8 = s1 * 8
            extra = base_8 + 2.8 * sq_factor * (delta - 8)
        elif delta <= 40:
            base_8 = s1 * 8
            base_20 = base_8 + 2.5 * sq_factor * 12
            extra = base_20 + 0.65 * sq_factor * (delta - 20)
        else:
            base_8 = s1 * 8
            base_20 = base_8 + 2.5 * sq_factor * 12
            base_40 = base_20 + 0.65 * sq_factor * 20
            extra = base_40 + 0.2 * (delta - 40)
        mozjpeg_reduction = min(93.0, encoder_bonus + extra)
    elif delta >= -3:
        # delta=0 or small negative (within quality estimation ±3 rounding).
        # delta=-1 is almost always IJG rounding (off-by-one), so treat as
        # full bonus. Only taper at -2/-3 where real negative delta is possible.
        # Encoder bonus depends on source quality at delta≈0:
        # ~28% for sq<=75, dropping to ~8% at sq>=90.
        encoder_bonus = max(8.0, 28.0 - 1.67 * max(0, source_q - 78))
        taper = 1.0 + min(0, delta + 1) / 5.0  # -1→1.0, -2→0.8, -3→0.6
        mozjpeg_reduction = encoder_bonus * taper
    else:
        # Large negative delta: target quality is well above source,
        # mozjpeg at higher quality produces larger file.
        mozjpeg_reduction = 0.0

    # Optimizer picks the method with the largest size reduction
    if mozjpeg_reduction >= jpegtran_reduction:
        reduction = mozjpeg_reduction
        method = "mozjpeg"
    else:
        reduction = jpegtran_reduction
        method = "jpegtran"

    # Screenshot/flat content adjustment: screenshots have fundamentally
    # different compression characteristics — better at small deltas
    # (flat regions → many zero DCT coefficients) but saturate earlier
    # at large deltas (~78% ceiling vs ~93% for photos). Blend the
    # photo-calibrated prediction toward the empirical screenshot mean.
    if delta > 0 and info.flat_pixel_ratio is not None and info.flat_pixel_ratio > 0.75:
        screenshot_mean = 69.0
        reduction = reduction * 0.4 + screenshot_mean * 0.6

    if info.has_exif and config.strip_metadata:
        reduction += 2.0

    if config.progressive_jpeg:
        reduction += 1.0

    if info.is_progressive:
        reduction *= 0.95

    # Tiny-file adjustment: JPEG headers (markers, quantization + Huffman
    # tables) are fixed overhead. For very small files (<2KB), the overhead
    # is proportionally larger than the 700B baseline because coding
    # efficiency drops and minimum table sizes dominate.
    if info.file_size < 5000:
        overhead = 700 + max(0, 2000 - info.file_size) * 0.3
        overhead_ratio = overhead / info.file_size
        max_reduction = (1 - overhead_ratio) * 100
        reduction = min(reduction, max(0, max_reduction))

    potential = "high" if reduction >= 40 else ("medium" if reduction >= 15 else "low")
    already_optimized = delta < 0 and not info.has_exif
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
    quality delta to predict reduction. Uses interpolation between
    calibrated reference curves at q60, q80, and q95 source qualities.
    """
    w = info.dimensions.get("width", 1)
    h = info.dimensions.get("height", 1)
    pixels = max(w * h, 1)
    bpp = (info.file_size * 8) / pixels
    est_source_q = _bpp_to_quality(bpp)
    delta = est_source_q - config.quality

    if delta < 0:
        reduction = 0.0
        potential = "low"
    elif delta == 0:
        reduction = 5.0
        potential = "low"
    else:
        reduction = _webp_interpolated_reduction(est_source_q, delta)
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


def _webp_interpolated_reduction(est_source_q: int, delta: int) -> float:
    """Interpolate WebP reduction from calibrated reference curves.

    Three reference curves at q60, q80, q95 are derived from benchmark
    data. For intermediate source qualities, linearly interpolate.
    """

    def _curve_60(d: int) -> float:
        return min(50.0, 7.0 + 0.92 * d)

    def _curve_80(d: int) -> float:
        if d <= 20:
            return 5.5 + 1.33 * d
        elif d <= 40:
            return 32.0 + 1.1 * (d - 20)
        else:
            return min(75.0, 54.0 + 0.4 * (d - 40))

    def _curve_95(d: int) -> float:
        if d <= 15:
            return 5.0 + 2.77 * d
        elif d <= 35:
            return 46.5 + 0.825 * (d - 15)
        elif d <= 55:
            return 63.0 + 0.475 * (d - 35)
        else:
            return min(78.0, 72.5 + 0.2 * (d - 55))

    if est_source_q <= 60:
        return _curve_60(delta)
    elif est_source_q <= 80:
        t = (est_source_q - 60) / 20.0
        return _curve_60(delta) * (1 - t) + _curve_80(delta) * t
    elif est_source_q <= 95:
        t = (est_source_q - 80) / 15.0
        return _curve_80(delta) * (1 - t) + _curve_95(delta) * t
    else:
        return min(78.0, _curve_95(delta) * 1.03)


def _predict_gif(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """GIF heuristics — bytes-per-pixel and size-aware model.

    Uses bytes-per-pixel to distinguish gradient/photographic content
    (high bpp, low gifsicle savings) from flat graphics (low bpp,
    high savings). File size further adjusts: smaller files have less
    redundancy for gifsicle to exploit.
    """
    if info.frame_count > 1:
        reduction = 15.0
        potential = "medium"
    else:
        w = info.dimensions.get("width", 1)
        h = info.dimensions.get("height", 1)
        pixels = max(w * h, 1)
        bpp = info.file_size / pixels

        if info.file_size < 1000:
            reduction = 10.0
        elif bpp >= 0.10:
            reduction = 2.0
        elif bpp >= 0.03:
            reduction = 10.0 if info.file_size < 2500 else 14.0
        else:
            reduction = 12.0 if info.file_size < 2500 else 15.0

        potential = "medium" if reduction >= 10 else "low"

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="gifsicle",
        already_optimized=False,
        confidence="medium",
    )


def _predict_svg(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """SVG heuristics — bytes-based continuous model.

    Estimates absolute bytes saved from scour structural optimization
    (base) plus bloat removal, then converts to percentage.
    """
    ratio = info.svg_bloat_ratio

    if ratio is not None:
        base_bytes = 28.0
        bloat_bytes = info.file_size * ratio * 0.98
        total_saved = base_bytes + bloat_bytes
        reduction = max(3.0, min(60.0, (total_saved / max(1, info.file_size)) * 100))
    else:
        if info.has_metadata_chunks:
            reduction = 30.0
        else:
            reduction = 8.0

    potential = "high" if reduction >= 30 else ("medium" if reduction >= 10 else "low")
    already_optimized = reduction <= 5.0

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="scour",
        already_optimized=already_optimized,
        confidence="medium",
    )


def _predict_svgz(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """SVGZ heuristics — bytes-based continuous model (reduced for gzip)."""
    ratio = info.svg_bloat_ratio

    if ratio is not None:
        base_bytes = 5.0
        bloat_bytes = info.file_size * ratio * 0.38
        total_saved = base_bytes + bloat_bytes
        reduction = max(2.0, min(30.0, (total_saved / max(1, info.file_size)) * 100))
    else:
        reduction = 8.0 if info.has_metadata_chunks else 5.0

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="low",
        method="scour",
        already_optimized=reduction <= 3.0,
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
