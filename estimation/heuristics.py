from dataclasses import dataclass

from estimation.header_analysis import HeaderInfo
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


def predict_reduction(info: HeaderInfo, fmt: ImageFormat) -> Prediction:
    """Predict compression reduction based on format-specific heuristics."""
    dispatch = {
        ImageFormat.PNG: _predict_png,
        ImageFormat.APNG: _predict_apng,
        ImageFormat.JPEG: _predict_jpeg,
        ImageFormat.WEBP: _predict_webp,
        ImageFormat.GIF: _predict_gif,
        ImageFormat.SVG: _predict_svg,
        ImageFormat.SVGZ: _predict_svg,
        ImageFormat.AVIF: _predict_metadata_only,
        ImageFormat.HEIC: _predict_metadata_only,
        ImageFormat.TIFF: _predict_passthrough,
        ImageFormat.BMP: _predict_passthrough,
        ImageFormat.PSD: _predict_passthrough,
    }
    predictor = dispatch.get(fmt, _predict_passthrough)
    return predictor(info)


def _predict_png(info: HeaderInfo) -> Prediction:
    """PNG heuristics.

    - Palette mode: limited room (~5-10%)
    - Non-palette, low color complexity: pngquant very effective (~55-80%)
    - Non-palette, high color complexity (photos): pngquant fails, oxipng-only (~3-5%)
    - Has metadata chunks: metadata removal adds a few %
    """
    if info.is_palette_mode:
        # Already indexed — re-quantization has moderate effect
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

    if info.has_metadata_chunks:
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
        # Could not sample — conservative fallback
        return 20.0, "medium", "pngquant + oxipng", "low"

    if color_ratio < 0.005:
        # Truly solid / near-solid (1-2 colors): pngquant crushes these
        return 85.0, "high", "pngquant + oxipng", "high"
    elif color_ratio < 0.05:
        # Very low complexity (simple screenshots, icons): high but variable
        return 55.0, "high", "pngquant + oxipng", "medium"
    elif color_ratio < 0.20:
        # Graphics / screenshots with some color variation
        return 55.0, "high", "pngquant + oxipng", "medium"
    elif color_ratio < 0.50:
        # Mixed content — pngquant may or may not succeed
        return 30.0, "medium", "pngquant + oxipng", "low"
    else:
        # Photo / gradient / noise — pngquant will fail (exit 99)
        return 3.0, "low", "oxipng", "medium"


def _predict_apng(info: HeaderInfo) -> Prediction:
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


def _predict_jpeg(info: HeaderInfo) -> Prediction:
    """JPEG heuristics.

    - Quality > 90: big savings (~50-65%)
    - Quality 70-90: moderate savings (~25-40%)
    - Quality < 70: minimal savings (~5-15%), lossless only
    """
    q = info.estimated_quality or 80

    if q > 90:
        reduction = 55.0
        potential = "high"
        method = "mozjpeg"
    elif q > 70:
        reduction = 30.0
        potential = "medium"
        method = "mozjpeg"
    else:
        reduction = 10.0
        potential = "low"
        method = "jpegtran"

    if info.is_progressive:
        reduction *= 0.9  # Slightly less room

    if info.has_exif:
        reduction += 2.0

    already_optimized = q <= 70 and not info.has_exif
    estimated_size = int(info.file_size * (1 - reduction / 100))

    return Prediction(
        estimated_size=estimated_size,
        reduction_percent=round(reduction, 1),
        potential=potential,
        method=method,
        already_optimized=already_optimized,
        confidence="medium",  # Thumbnail will upgrade to "high"
    )


def _predict_webp(info: HeaderInfo) -> Prediction:
    """WebP heuristics — moderate savings expected."""
    reduction = 35.0
    if info.file_size < 10_000:
        reduction = 15.0  # Small files compress less
    elif info.file_size > 500_000:
        reduction = 40.0

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="medium",
        method="pillow",
        already_optimized=False,
        confidence="medium",
    )


def _predict_gif(info: HeaderInfo) -> Prediction:
    """GIF heuristics — frame-dependent."""
    if info.frame_count > 1:
        reduction = 25.0
        potential = "medium"
    else:
        reduction = 15.0
        potential = "low"

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential=potential,
        method="gifsicle",
        already_optimized=False,
        confidence="medium",
    )


def _predict_svg(info: HeaderInfo) -> Prediction:
    """SVG heuristics — depends on metadata and editor bloat."""
    if info.has_metadata_chunks:
        reduction = 45.0
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


def _predict_metadata_only(info: HeaderInfo) -> Prediction:
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


def _predict_passthrough(info: HeaderInfo) -> Prediction:
    """TIFF/BMP/PSD — limited optimization potential."""
    return Prediction(
        estimated_size=info.file_size,
        reduction_percent=0.0,
        potential="low",
        method=f"pillow-{info.format.value}",
        already_optimized=True,
        confidence="low",
    )
