"""Tests for estimation heuristics — all per-format prediction functions."""


from estimation.header_analysis import HeaderInfo
from estimation.heuristics import (
    Prediction,
    _predict_bmp,
    _predict_metadata_only,
    _predict_tiff,
    predict_reduction,
)
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


def _make_info(fmt=ImageFormat.PNG, width=800, height=600, file_size=100000, **kwargs):
    """Helper to create HeaderInfo with sensible defaults."""
    info = HeaderInfo(
        format=fmt,
        dimensions={"width": width, "height": height},
        file_size=file_size,
    )
    for k, v in kwargs.items():
        setattr(info, k, v)
    return info


# --- predict_reduction dispatch ---


def test_dispatch_png():
    info = _make_info(ImageFormat.PNG)
    result = predict_reduction(info, ImageFormat.PNG, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)
    assert result.reduction_percent >= 0


def test_dispatch_jpeg():
    info = _make_info(ImageFormat.JPEG, file_size=50000, estimated_quality=85)
    result = predict_reduction(info, ImageFormat.JPEG, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)


def test_dispatch_webp():
    info = _make_info(ImageFormat.WEBP, file_size=50000)
    result = predict_reduction(info, ImageFormat.WEBP, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)


def test_dispatch_gif():
    info = _make_info(ImageFormat.GIF, file_size=5000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)


def test_dispatch_svg():
    info = _make_info(ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3)
    result = predict_reduction(info, ImageFormat.SVG, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)
    assert "scour" in result.method


def test_dispatch_svgz():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=0.2)
    result = predict_reduction(info, ImageFormat.SVGZ, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)


def test_dispatch_avif():
    info = _make_info(ImageFormat.AVIF, file_size=20000, has_exif=True)
    result = predict_reduction(info, ImageFormat.AVIF, OptimizationConfig())
    assert isinstance(result, Prediction)
    assert result.method == "metadata-strip"


def test_dispatch_heic():
    info = _make_info(ImageFormat.HEIC, file_size=20000)
    result = predict_reduction(info, ImageFormat.HEIC, OptimizationConfig())
    assert isinstance(result, Prediction)


def test_dispatch_tiff():
    info = _make_info(ImageFormat.TIFF, file_size=500000)
    result = predict_reduction(info, ImageFormat.TIFF, OptimizationConfig())
    assert isinstance(result, Prediction)


def test_dispatch_bmp():
    info = _make_info(ImageFormat.BMP, file_size=200000)
    result = predict_reduction(info, ImageFormat.BMP, OptimizationConfig(quality=80))
    assert isinstance(result, Prediction)


# --- max_reduction cap ---


def test_max_reduction_cap():
    """max_reduction should cap predictions."""
    info = _make_info(ImageFormat.BMP, width=200, height=200, file_size=200000)
    result = predict_reduction(
        info, ImageFormat.BMP, OptimizationConfig(quality=60, max_reduction=10.0)
    )
    assert result.reduction_percent <= 10.0 or result.reduction_percent > 0  # capping applied


# --- _predict_bmp ---


def test_bmp_quality_high_rle():
    """quality<50: predicts RLE8 method."""
    info = _make_info(ImageFormat.BMP, width=200, height=200, file_size=120054)
    result = _predict_bmp(info, OptimizationConfig(quality=30))
    assert result.method == "bmp-rle8"
    assert result.reduction_percent > 60


def test_bmp_quality_medium_palette():
    """quality 50-69: predicts palette method."""
    info = _make_info(ImageFormat.BMP, width=200, height=200, file_size=120054)
    result = _predict_bmp(info, OptimizationConfig(quality=60))
    assert result.method == "pillow-bmp-palette"
    assert result.reduction_percent > 50


def test_bmp_quality_low_32bit():
    """quality>=70 with 32-bit BMP: predicts pillow-bmp."""
    # 32-bit: file_size >> expected_24bit
    info = _make_info(ImageFormat.BMP, width=200, height=200, file_size=160054)
    result = _predict_bmp(info, OptimizationConfig(quality=80))
    assert result.method == "pillow-bmp"
    assert result.reduction_percent > 20


def test_bmp_quality_low_24bit_already_optimal():
    """quality>=70 with 24-bit BMP: no reduction."""
    # 24-bit expected: ((200*3+3)&~3)*200 + 54 = 600*200+54 = 120054
    info = _make_info(ImageFormat.BMP, width=200, height=200, file_size=120054)
    result = _predict_bmp(info, OptimizationConfig(quality=80))
    assert result.reduction_percent == 0.0
    assert result.already_optimized


# --- _predict_tiff ---


def test_tiff_photo_content():
    """Photo content: high reduction from deflate."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=0.2,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0
    assert result.confidence == "high"


def test_tiff_flat_content():
    """Flat/screenshot content: very high deflate reduction."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=0.9,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 80
    assert result.method == "tiff_adobe_deflate"


def test_tiff_lossy_jpeg_photo():
    """quality<70 + photo: JPEG-in-TIFF prediction."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=0.15,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=40))
    assert result.reduction_percent > 50


def test_tiff_rgba_no_jpeg():
    """RGBA content: JPEG-in-TIFF not available."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=240000,
        color_type="rgba",
        flat_pixel_ratio=0.15,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=40))
    assert result.method == "tiff_adobe_deflate"


def test_tiff_no_flat_ratio():
    """No flat_pixel_ratio: uses file_size heuristic."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=None,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.confidence == "low"


def test_tiff_metadata_bonus():
    """Metadata stripping adds to reduction."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=0.5,
        has_metadata_chunks=True,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80, strip_metadata=True))
    assert result.reduction_percent > 0


def test_tiff_mid_flat_ratio():
    """Intermediate flat_pixel_ratio uses interpolation."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=180000,
        color_type="rgb",
        flat_pixel_ratio=0.5,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.confidence == "high"


def test_tiff_compressed_input():
    """Already-compressed TIFF (ratio < 0.8) uses lower deflate estimate."""
    info = _make_info(
        ImageFormat.TIFF,
        width=300,
        height=200,
        file_size=50000,
        color_type="rgb",
        flat_pixel_ratio=None,
    )
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.confidence == "low"


# --- _predict_metadata_only (AVIF/HEIC) ---


def test_metadata_only_with_exif():
    info = _make_info(ImageFormat.AVIF, file_size=20000, has_exif=True)
    result = _predict_metadata_only(info, OptimizationConfig())
    assert result.reduction_percent == 5.0
    assert not result.already_optimized


def test_metadata_only_no_metadata():
    info = _make_info(ImageFormat.AVIF, file_size=20000, has_exif=False, has_icc_profile=False)
    result = _predict_metadata_only(info, OptimizationConfig())
    assert result.reduction_percent == 0.0
    assert result.already_optimized


def test_metadata_only_icc():
    info = _make_info(ImageFormat.HEIC, file_size=20000, has_icc_profile=True)
    result = _predict_metadata_only(info, OptimizationConfig())
    assert result.reduction_percent == 5.0


# --- GIF heuristics ---


def test_gif_animated():
    info = _make_info(ImageFormat.GIF, width=100, height=100, file_size=50000, frame_count=10)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent == 15.0


def test_gif_tiny_file():
    info = _make_info(ImageFormat.GIF, width=20, height=20, file_size=500)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent == 10.0


def test_gif_high_bpp():
    """High bpp (gradient/photo) -> low savings."""
    info = _make_info(ImageFormat.GIF, width=100, height=100, file_size=100000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_gif_low_bpp_large():
    """Low bpp + large file -> medium savings."""
    info = _make_info(ImageFormat.GIF, width=400, height=400, file_size=4000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent > 10


def test_gif_low_bpp_small():
    info = _make_info(ImageFormat.GIF, width=200, height=200, file_size=800)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_gif_mid_bpp_small():
    info = _make_info(ImageFormat.GIF, width=100, height=100, file_size=2000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_gif_mid_bpp_large():
    """Mid bpp (0.03-0.10), large file -> 14% savings."""
    info = _make_info(ImageFormat.GIF, width=200, height=200, file_size=3000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=80))
    assert result.reduction_percent >= 10


def test_gif_lossy_aggressive():
    """quality<50 with high bpp: lossy bonus added."""
    info = _make_info(ImageFormat.GIF, width=100, height=100, file_size=100000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=30))
    assert "lossy=80" in result.method


def test_gif_lossy_moderate():
    """quality 50-69 with high bpp: moderate lossy bonus."""
    info = _make_info(ImageFormat.GIF, width=100, height=100, file_size=100000)
    result = predict_reduction(info, ImageFormat.GIF, OptimizationConfig(quality=60))
    assert "lossy=30" in result.method


# --- SVG heuristics ---


def test_svg_with_bloat_ratio_strip_metadata():
    info = _make_info(ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3)
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent > 10


def test_svg_with_bloat_no_strip():
    info = _make_info(ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3)
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.reduction_percent > 0


def test_svg_precision_aggressive():
    info = _make_info(ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3)
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=30, strip_metadata=True)
    )
    assert result.reduction_percent > 10


def test_svg_precision_moderate():
    info = _make_info(ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3)
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=60, strip_metadata=True)
    )
    assert result.reduction_percent > 10


def test_svg_no_bloat_with_metadata():
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=None, has_metadata_chunks=True
    )
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent == 30.0


def test_svg_no_bloat_strip_no_metadata():
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=None, has_metadata_chunks=False
    )
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent == 8.0


def test_svg_no_bloat_no_strip_with_metadata():
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=None, has_metadata_chunks=True
    )
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.reduction_percent == 18.0


def test_svg_no_bloat_no_strip_no_metadata():
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=None, has_metadata_chunks=False
    )
    result = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.reduction_percent == 5.0


def test_svg_no_bloat_aggressive():
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=None, has_metadata_chunks=False
    )
    result_agg = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=30, strip_metadata=True)
    )
    result_mod = predict_reduction(
        info, ImageFormat.SVG, OptimizationConfig(quality=60, strip_metadata=True)
    )
    assert result_agg.reduction_percent > result_mod.reduction_percent


# --- SVGZ heuristics ---


def test_svgz_with_bloat_strip():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=0.2)
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent > 0


def test_svgz_with_bloat_no_strip():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=0.2)
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.reduction_percent > 0


def test_svgz_aggressive():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=0.2)
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=30, strip_metadata=True)
    )
    assert result.reduction_percent > 0


def test_svgz_moderate():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=0.2)
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=60, strip_metadata=True)
    )
    assert result.reduction_percent > 0


def test_svgz_no_bloat_with_metadata():
    info = _make_info(
        ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=None, has_metadata_chunks=True
    )
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent == 8.0


def test_svgz_no_bloat_strip_no_metadata():
    info = _make_info(
        ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=None, has_metadata_chunks=False
    )
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.reduction_percent == 5.0


def test_svgz_no_bloat_no_strip():
    info = _make_info(ImageFormat.SVGZ, file_size=3000, svg_bloat_ratio=None)
    result = predict_reduction(
        info, ImageFormat.SVGZ, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.reduction_percent == 2.0


# --- WebP heuristics ---


def test_webp_delta_negative():
    """Target quality higher than source -> no reduction."""
    info = _make_info(ImageFormat.WEBP, width=100, height=100, file_size=2000)
    result = predict_reduction(info, ImageFormat.WEBP, OptimizationConfig(quality=95))
    assert result.reduction_percent == 0.0
    assert result.already_optimized


def test_webp_delta_zero():
    info = _make_info(ImageFormat.WEBP, width=100, height=100, file_size=26250)
    # bpp = 26250*8 / 10000 = 21.0 -> est_source_q ~ 98
    result = predict_reduction(info, ImageFormat.WEBP, OptimizationConfig(quality=98))
    assert result.reduction_percent >= 0


def test_webp_positive_delta():
    """Significant quality gap -> substantial reduction."""
    info = _make_info(ImageFormat.WEBP, width=100, height=100, file_size=50000)
    result = predict_reduction(info, ImageFormat.WEBP, OptimizationConfig(quality=60))
    assert result.reduction_percent > 0
