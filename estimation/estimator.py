import io

from PIL import Image

from estimation.header_analysis import HeaderInfo, analyze_header
from estimation.heuristics import Prediction, predict_reduction
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format


async def estimate(data: bytes, config: OptimizationConfig | None = None) -> EstimateResponse:
    """Estimate compression savings without full compression.

    Three-layer estimation:
    1. Header analysis — format, dimensions, bit depth, color type (~1ms)
    2. Format heuristics — predicted reduction based on signals (~1ms)
    3. Thumbnail compression — JPEG/WebP only (~15-30ms)

    Target latency: ~20-50ms.
    """
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(data)
    header_info = analyze_header(data, fmt)
    prediction = predict_reduction(header_info, fmt, config)

    # Layer 3: thumbnail compression for JPEG only.
    # WebP skipped: thumbnail measures decoded-pixel compressibility, not
    # re-compression of an already-compressed file, causing overestimates.
    if fmt == ImageFormat.JPEG and prediction.method == "jpegtran":
        thumbnail_ratio = await _thumbnail_compress(data, fmt, config.quality)
        if thumbnail_ratio is not None:
            prediction = _combine_with_thumbnail(
                prediction, thumbnail_ratio, header_info
            )

    return EstimateResponse(
        original_size=len(data),
        original_format=fmt.value,
        dimensions=header_info.dimensions,
        color_type=header_info.color_type,
        bit_depth=header_info.bit_depth,
        estimated_optimized_size=prediction.estimated_size,
        estimated_reduction_percent=prediction.reduction_percent,
        optimization_potential=prediction.potential,
        method=prediction.method,
        already_optimized=prediction.already_optimized,
        confidence=prediction.confidence,
    )


async def _thumbnail_compress(
    data: bytes, fmt: ImageFormat, quality: int
) -> float | None:
    """Resize to 64x64, compress with actual tool, return compression ratio.

    Only used for JPEG and WebP where thumbnail compression
    ratio scales roughly linearly with resolution.

    Returns:
        Compression ratio (e.g., 0.4 means 60% reduction), or None on failure.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.thumbnail((64, 64))

        # Save at q100 to get baseline
        original_buf = io.BytesIO()
        thumb_format = "JPEG" if fmt == ImageFormat.JPEG else "WEBP"
        img.save(original_buf, format=thumb_format, quality=100)
        original_size = original_buf.tell()

        # Save at target quality from config
        compressed_buf = io.BytesIO()
        img.save(compressed_buf, format=thumb_format, quality=quality)
        compressed_size = compressed_buf.tell()

        if original_size == 0:
            return None

        return compressed_size / original_size
    except Exception:
        return None


def _combine_with_thumbnail(
    prediction: Prediction,
    thumbnail_ratio: float,
    info: HeaderInfo,
) -> Prediction:
    """Adjust prediction using thumbnail compression ratio.

    If thumbnail and heuristics agree → high confidence.
    If they diverge → average them, medium confidence.
    """
    thumbnail_reduction = round((1 - thumbnail_ratio) * 100, 1)

    heuristic_reduction = prediction.reduction_percent

    # Average the two estimates
    combined_reduction = round((heuristic_reduction + thumbnail_reduction) / 2, 1)

    # Confidence: if they agree within 15%, high confidence
    if abs(heuristic_reduction - thumbnail_reduction) < 15:
        confidence = "high"
    else:
        confidence = "medium"

    estimated_size = int(info.file_size * (1 - combined_reduction / 100))

    return Prediction(
        estimated_size=estimated_size,
        reduction_percent=combined_reduction,
        potential=prediction.potential,
        method=prediction.method,
        already_optimized=prediction.already_optimized,
        confidence=confidence,
    )
