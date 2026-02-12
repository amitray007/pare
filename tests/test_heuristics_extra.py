"""Extra heuristics tests — PNG, JPEG, APNG, WebP prediction paths."""


import pytest

from estimation.header_analysis import HeaderInfo
from estimation.heuristics import (
    _bpp_to_quality,
    _predict_apng,
    _predict_jpeg,
    _predict_png,
    _webp_interpolated_reduction,
    predict_reduction,
)
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


def _make_info(fmt=ImageFormat.PNG, width=800, height=600, file_size=100000, **kwargs):
    info = HeaderInfo(
        format=fmt,
        dimensions={"width": width, "height": height},
        file_size=file_size,
    )
    for k, v in kwargs.items():
        setattr(info, k, v)
    return info


# --- PNG lossless path ---


def test_png_lossless_with_oxipng_probe_small():
    """Lossless path + oxipng probe on small file (full file)."""
    info = _make_info(file_size=40000, oxipng_probe_ratio=0.70)
    config = OptimizationConfig(quality=80, png_lossy=False)
    result = _predict_png(info, config)
    assert result.method == "oxipng"
    assert result.reduction_percent == pytest.approx(30.0, abs=1.0)


def test_png_lossless_with_oxipng_probe_large():
    """Lossless path + oxipng probe on large file (crop-based)."""
    info = _make_info(file_size=200000, oxipng_probe_ratio=0.70)
    config = OptimizationConfig(quality=80, png_lossy=False)
    result = _predict_png(info, config)
    assert result.method == "oxipng"
    # Crop probe gets 0.6x discount
    assert result.reduction_percent == pytest.approx(18.0, abs=1.0)


def test_png_lossless_no_probe():
    """Lossless path with no probe -> default 5%."""
    info = _make_info(file_size=200000)
    config = OptimizationConfig(quality=80, png_lossy=False)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(5.0, abs=1.0)


def test_png_lossless_with_metadata():
    """Lossless path + metadata stripping adds 3%."""
    info = _make_info(file_size=200000, has_metadata_chunks=True)
    config = OptimizationConfig(quality=80, png_lossy=False, strip_metadata=True)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(8.0, abs=1.0)


# --- PNG lossy path ---


def test_png_lossy_palette_mode_few_colors():
    """Palette mode with few colors -> 15%."""
    info = _make_info(file_size=5000, is_palette_mode=True, color_count=8)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(15.0, abs=2.0)


def test_png_lossy_palette_mode_small_file():
    """Palette mode, small file -> 30%."""
    info = _make_info(file_size=1500, is_palette_mode=True, color_count=128)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(30.0, abs=2.0)


def test_png_lossy_palette_mode_normal():
    """Palette mode, normal file -> 40%."""
    info = _make_info(file_size=10000, is_palette_mode=True, color_count=200)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(40.0, abs=5.0)


def test_png_lossy_no_probes_no_color_ratio():
    """No probe data at all -> fallback 20%."""
    info = _make_info(file_size=100000)
    # Clear all probe-related fields
    info.oxipng_probe_ratio = None
    info.unique_color_ratio = None
    info.png_quantize_ratio = None
    info.png_pngquant_probe_ratio = None
    info.flat_pixel_ratio = None
    info.is_palette_mode = False
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent == pytest.approx(20.0, abs=1.0)
    assert result.confidence == "low"


def test_png_lossy_photo_content_large():
    """Photo content (high color ratio, low flat ratio) on large file."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.7,
        flat_pixel_ratio=0.2,
        oxipng_probe_ratio=0.97,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 50


def test_png_lossy_flat_content():
    """Flat content (high flat ratio) -> lossless wins."""
    info = _make_info(
        file_size=30000,
        unique_color_ratio=0.01,
        flat_pixel_ratio=0.95,
        oxipng_probe_ratio=0.50,
        png_pngquant_probe_ratio=0.60,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_lossy_tiny_file_cap():
    """Very small file gets tiny-file cap."""
    info = _make_info(file_size=200, is_palette_mode=True, color_count=128)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    # Cap based on minimum PNG overhead
    assert result.reduction_percent >= 0


def test_png_lossy_quality_50_bonus():
    """quality < 50 adds 64-color bonus for photo content."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.7,
        flat_pixel_ratio=0.2,
        oxipng_probe_ratio=0.97,
        is_palette_mode=False,
    )
    config_40 = OptimizationConfig(quality=40, png_lossy=True)
    config_60 = OptimizationConfig(quality=60, png_lossy=True)
    result_40 = _predict_png(info, config_40)
    result_60 = _predict_png(info, config_60)
    assert result_40.reduction_percent >= result_60.reduction_percent


def test_png_with_pngquant_probe():
    """Small file with pngquant probe data."""
    info = _make_info(
        file_size=5000,
        unique_color_ratio=0.3,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.85,
        png_pngquant_probe_ratio=0.40,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.confidence == "high"
    assert result.reduction_percent > 30


def test_png_lossy_with_quantize_ratio():
    """Small file with quantize ratio but no pngquant probe."""
    info = _make_info(
        file_size=30000,
        unique_color_ratio=0.3,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.85,
        png_quantize_ratio=0.50,
        png_pngquant_probe_ratio=None,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_lossy_low_color_ratio_large():
    """Large file with very low unique color ratio."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.003,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.95,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 50


# --- APNG ---


def test_apng_with_metadata():
    """APNG with metadata -> 5%."""
    info = _make_info(ImageFormat.APNG, file_size=50000, has_metadata_chunks=True)
    result = _predict_apng(info, OptimizationConfig())
    assert result.reduction_percent == 5.0


def test_apng_no_metadata():
    """APNG without metadata -> 2%."""
    info = _make_info(ImageFormat.APNG, file_size=50000, has_metadata_chunks=False)
    result = _predict_apng(info, OptimizationConfig())
    assert result.reduction_percent == 2.0
    assert result.already_optimized


# --- JPEG ---


def test_jpeg_positive_delta():
    """Source quality > target quality -> mozjpeg wins."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85)
    result = _predict_jpeg(info, OptimizationConfig(quality=60))
    assert result.method == "mozjpeg"
    assert result.reduction_percent > 20


def test_jpeg_negative_delta():
    """Target quality > source quality -> large negative delta, 0% mozjpeg."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=50)
    result = _predict_jpeg(info, OptimizationConfig(quality=80))
    # Jpegtran should win since mozjpeg produces larger at higher quality
    assert result.reduction_percent > 0


def test_jpeg_zero_delta():
    """delta=0 -> encoder bonus only."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=80)
    result = _predict_jpeg(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_jpeg_small_negative_delta():
    """delta=-2 -> tapered encoder bonus."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=78)
    result = _predict_jpeg(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_jpeg_high_source_quality():
    """Very high source quality (>90) -> exponential jpegtran bonus."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=98)
    result = _predict_jpeg(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 30


def test_jpeg_with_exif_strip():
    """EXIF + strip_metadata -> +2%."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85, has_exif=True)
    result_strip = _predict_jpeg(info, OptimizationConfig(quality=60, strip_metadata=True))
    result_no_strip = _predict_jpeg(info, OptimizationConfig(quality=60, strip_metadata=False))
    assert result_strip.reduction_percent > result_no_strip.reduction_percent


def test_jpeg_progressive_bonus():
    """progressive_jpeg -> +1%."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85)
    result_prog = _predict_jpeg(info, OptimizationConfig(quality=60, progressive_jpeg=True))
    result_no = _predict_jpeg(info, OptimizationConfig(quality=60, progressive_jpeg=False))
    assert result_prog.reduction_percent > result_no.reduction_percent


def test_jpeg_is_progressive_penalty():
    """Already progressive source -> 0.95x reduction."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85, is_progressive=True)
    result = _predict_jpeg(info, OptimizationConfig(quality=60))
    info2 = _make_info(
        ImageFormat.JPEG, file_size=50000, estimated_quality=85, is_progressive=False
    )
    result2 = _predict_jpeg(info2, OptimizationConfig(quality=60))
    assert result.reduction_percent < result2.reduction_percent


def test_jpeg_screenshot_content():
    """Flat pixel ratio > 0.75 -> screenshot adjustment."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85, flat_pixel_ratio=0.9)
    result = _predict_jpeg(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 0


def test_jpeg_large_positive_delta():
    """Very large delta (>40) -> high reduction."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=95)
    result = _predict_jpeg(info, OptimizationConfig(quality=30))
    assert result.reduction_percent > 50


def test_jpeg_already_optimized():
    """Already optimized: negative delta, no EXIF."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=60, has_exif=False)
    result = _predict_jpeg(info, OptimizationConfig(quality=80))
    assert result.already_optimized


def test_jpeg_no_estimated_quality():
    """No estimated quality -> default to 85."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=None)
    result = _predict_jpeg(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 0


def test_jpeg_delta_in_8_20_range():
    """Delta between 8 and 20 uses second piecewise segment."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85)
    result = _predict_jpeg(info, OptimizationConfig(quality=70))  # delta=15
    assert result.reduction_percent > 20


def test_jpeg_delta_in_20_40_range():
    """Delta between 20 and 40 uses third piecewise segment."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=90)
    result = _predict_jpeg(info, OptimizationConfig(quality=60))  # delta=30
    assert result.reduction_percent > 30


# --- WebP ---


def test_bpp_to_quality_low():
    assert _bpp_to_quality(0.05) == 20


def test_bpp_to_quality_mid():
    q = _bpp_to_quality(2.5)
    assert 60 <= q <= 80


def test_bpp_to_quality_high():
    q = _bpp_to_quality(5.5)
    assert q >= 95


def test_webp_interpolated_below_60():
    """Source quality <= 60 -> curve_60."""
    result = _webp_interpolated_reduction(50, 10)
    assert result > 0


def test_webp_interpolated_60_80():
    """Source quality 60-80 -> interpolated."""
    result = _webp_interpolated_reduction(70, 10)
    assert result > 0


def test_webp_interpolated_80_95():
    """Source quality 80-95 -> interpolated."""
    result = _webp_interpolated_reduction(88, 10)
    assert result > 0


def test_webp_interpolated_above_95():
    """Source quality > 95 -> curve_95 * 1.03."""
    result = _webp_interpolated_reduction(98, 10)
    assert result > 0


def test_webp_curve_80_large_delta():
    """WebP curve_80 at large delta (>40)."""
    result = _webp_interpolated_reduction(80, 50)
    assert result > 50


def test_webp_curve_95_large_delta():
    """WebP curve_95 at large delta (>55)."""
    result = _webp_interpolated_reduction(95, 60)
    assert result > 60


# --- max_reduction cap (JPEG path) ---


def test_max_reduction_jpeg_cap():
    """JPEG with max_reduction: cap mozjpeg, keep jpegtran if lower."""
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=95)
    config = OptimizationConfig(quality=60, max_reduction=10.0)
    result = predict_reduction(info, ImageFormat.JPEG, config)
    assert result.reduction_percent <= 15.0  # some tolerance for jpegtran


def test_max_reduction_webp_cap():
    """WebP with max_reduction: caps to max_reduction."""
    info = _make_info(ImageFormat.WEBP, width=100, height=100, file_size=50000)
    config = OptimizationConfig(quality=60, max_reduction=5.0)
    result = predict_reduction(info, ImageFormat.WEBP, config)
    assert result.reduction_percent <= 5.0
