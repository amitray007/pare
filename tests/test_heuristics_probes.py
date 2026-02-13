"""Tests for heuristics probe paths — JPEG probe, PNG lossy probe, PNG by-complexity."""

import io
from unittest.mock import MagicMock, patch

from PIL import Image

from estimation.header_analysis import HeaderInfo
from estimation.heuristics import (
    _predict_avif,
    _predict_bmp,
    _predict_gif,
    _predict_heic,
    _predict_jpeg,
    _predict_png,
    _predict_svg,
    _predict_svgz,
    _predict_tiff,
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


# --- JPEG probe path ---


def test_jpeg_probe_small_file():
    """Small JPEG file (< 12KB) with raw_data triggers probe."""
    # Create a small JPEG
    img = Image.new("RGB", (32, 32), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    info = _make_info(
        ImageFormat.JPEG,
        width=32,
        height=32,
        file_size=len(data),
        estimated_quality=85,
        raw_data=data,
    )

    # Mock subprocess to simulate cjpeg/jpegtran
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = data[: int(len(data) * 0.6)]

    with patch("estimation.heuristics.subprocess.run", return_value=mock_result):
        result = _predict_jpeg(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 0
    assert result.confidence == "high"


def test_jpeg_probe_failure():
    """JPEG probe fails -> falls back to heuristic."""
    img = Image.new("RGB", (32, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    info = _make_info(
        ImageFormat.JPEG,
        width=32,
        height=32,
        file_size=len(data),
        estimated_quality=85,
        raw_data=data,
    )

    with patch("estimation.heuristics.subprocess.run", side_effect=Exception("no cjpeg")):
        result = _predict_jpeg(info, OptimizationConfig(quality=60))
    # Should still return a result from heuristic path
    assert result.reduction_percent > 0


# --- PNG lossy probe path ---


def test_png_lossy_probe_small_file():
    """Small PNG (< 12KB) with raw_data and quality < 70 triggers lossy probe."""
    img = Image.new("RGB", (16, 16), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    info = _make_info(
        ImageFormat.PNG,
        width=16,
        height=16,
        file_size=len(data),
        raw_data=data,
    )

    mock_pngquant = MagicMock()
    mock_pngquant.returncode = 0
    mock_pngquant.stdout = data[: int(len(data) * 0.5)]

    def mock_subprocess_run(cmd, **kwargs):
        return mock_pngquant

    mock_oxipng = MagicMock()
    mock_oxipng.optimize_from_memory = lambda d, level=2: d[: int(len(d) * 0.9)]

    with patch("estimation.heuristics.subprocess.run", side_effect=mock_subprocess_run):
        with patch.dict("sys.modules", {"oxipng": mock_oxipng}):
            result = _predict_png(info, OptimizationConfig(quality=60, png_lossy=True))
    assert result.reduction_percent > 0


def test_png_lossy_probe_pngquant_fails():
    """PNG lossy probe: pngquant returns non-zero -> lossless only."""
    img = Image.new("RGB", (16, 16))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    info = _make_info(
        ImageFormat.PNG,
        width=16,
        height=16,
        file_size=len(data),
        raw_data=data,
    )

    mock_result = MagicMock()
    mock_result.returncode = 99
    mock_result.stdout = b""

    mock_oxipng = MagicMock()
    mock_oxipng.optimize_from_memory = lambda d, level=2: d[: int(len(d) * 0.9)]

    with patch("estimation.heuristics.subprocess.run", return_value=mock_result):
        with patch.dict("sys.modules", {"oxipng": mock_oxipng}):
            result = _predict_png(info, OptimizationConfig(quality=60, png_lossy=True))
    assert result.reduction_percent >= 0


# --- PNG by-complexity paths ---


def test_png_complexity_large_flat_content():
    """Large file, flat content: lossy reduction = 0 (lossless wins)."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.01,
        flat_pixel_ratio=0.9,
        oxipng_probe_ratio=0.50,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_complexity_large_low_color_ratio():
    """Large file, very low unique color ratio (< 0.005) -> 90% lossy."""
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


def test_png_complexity_large_mid_color_ratio():
    """Large file, mid color ratio (0.005-0.20) -> 55% lossy."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.10,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.90,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_complexity_small_quantize_ratio():
    """Small file, no pngquant probe but has quantize ratio."""
    info = _make_info(
        file_size=30000,
        unique_color_ratio=0.3,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.85,
        png_quantize_ratio=0.40,
        png_pngquant_probe_ratio=None,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_complexity_small_flat_with_quantize():
    """Small flat file with quantize ratio -> lossy = 0."""
    info = _make_info(
        file_size=30000,
        unique_color_ratio=0.01,
        flat_pixel_ratio=0.9,
        oxipng_probe_ratio=0.85,
        png_quantize_ratio=0.80,
        png_pngquant_probe_ratio=None,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


def test_png_complexity_large_with_quantize():
    """Large file with quantize probe -> uses heuristic."""
    info = _make_info(
        file_size=200000,
        unique_color_ratio=0.3,
        flat_pixel_ratio=0.3,
        oxipng_probe_ratio=0.85,
        png_quantize_ratio=0.40,
        is_palette_mode=False,
    )
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = _predict_png(info, config)
    assert result.reduction_percent > 0


# --- GIF prediction ---


def test_gif_very_small():
    """Very small GIF -> 0% reduction."""
    info = _make_info(ImageFormat.GIF, file_size=500, width=10, height=10)
    result = _predict_gif(info, OptimizationConfig())
    assert result.reduction_percent >= 0


def test_gif_large_low_bpp():
    """Large GIF with low bpp (< 0.03) -> ~15% reduction."""
    info = _make_info(ImageFormat.GIF, file_size=5000, width=500, height=500)
    result = _predict_gif(info, OptimizationConfig(quality=60))
    assert result.reduction_percent >= 15


def test_gif_high_bpp_with_lossy():
    """GIF with high bpp but lossy quality -> small reduction + lossy bonus."""
    # bpp = 5000 / (200*200) = 0.125 >= 0.10 -> base 2.0, bpp >= 0.05 + q<50 -> +8
    info = _make_info(ImageFormat.GIF, file_size=5000, width=200, height=200)
    result = _predict_gif(info, OptimizationConfig(quality=40))
    assert result.reduction_percent >= 10


# --- SVG / SVGZ prediction ---


def test_svg_prediction_with_bloat():
    """SVG with high bloat ratio."""
    info = _make_info(
        ImageFormat.SVG, file_size=5000, svg_bloat_ratio=0.3, has_metadata_chunks=True
    )
    result = _predict_svg(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 10


def test_svgz_prediction():
    """SVGZ prediction: limited savings."""
    info = _make_info(ImageFormat.SVGZ, file_size=3000)
    result = _predict_svgz(info, OptimizationConfig())
    assert result.reduction_percent >= 0


# --- TIFF prediction ---


def test_tiff_lossy_quality():
    """TIFF with quality < 70 and photo content: uses lossy JPEG-in-TIFF prediction."""
    # raw_size = 800*600*3 = 1440000. file_size close to raw for uncompressed TIFF.
    info = _make_info(ImageFormat.TIFF, file_size=1440000, flat_pixel_ratio=0.2)
    result = _predict_tiff(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 0


def test_tiff_lossless_quality():
    """TIFF with quality >= 70: lossless deflate."""
    # file_size = raw_size for uncompressed TIFF
    info = _make_info(ImageFormat.TIFF, file_size=1440000, flat_pixel_ratio=0.2)
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


def test_tiff_flat_content():
    """TIFF with flat content (screenshot): high deflate compression."""
    info = _make_info(ImageFormat.TIFF, file_size=1440000, flat_pixel_ratio=0.9)
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 90


def test_tiff_no_flat_ratio():
    """TIFF without flat_pixel_ratio: uses size-based heuristic."""
    info = _make_info(ImageFormat.TIFF, file_size=1440000, flat_pixel_ratio=None)
    result = _predict_tiff(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 0


# --- BMP prediction ---


def test_bmp_palette_quality():
    """BMP with quality < 70: palette prediction."""
    # 24-bit BMP: row_bytes_24 = (800*3+3)&~3 = 2400. file_size = 2400*600+54 = 1440054
    # 8-bit palette: expected_8bit = 54 + 1024 + ((800+3)&~3)*600 = 54+1024+481200 = 482278
    # reduction = (1 - 482278/1440054) * 100 ≈ 66.5%
    info = _make_info(ImageFormat.BMP, file_size=1440054, color_type="rgb")
    result = _predict_bmp(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 50


def test_bmp_rle8_quality():
    """BMP with quality < 50: RLE8 prediction."""
    info = _make_info(ImageFormat.BMP, file_size=1440054, color_type="rgb", flat_pixel_ratio=0.9)
    result = _predict_bmp(info, OptimizationConfig(quality=40))
    assert result.reduction_percent > 50
    assert result.method == "bmp-rle8"


def test_bmp_lossless_32bit():
    """BMP with quality >= 70 and 32-bit source: 32->24 bit reduction."""
    # 32-bit BMP: file_size > expected_24bit * 1.1
    # expected_24bit = (800*3+3)&~3 * 600 + 54 = 1440054
    # 32-bit: file_size ≈ 800*4*600 + 54 = 1920054
    info = _make_info(ImageFormat.BMP, file_size=1920054, color_type="rgba", bit_depth=8)
    result = _predict_bmp(info, OptimizationConfig(quality=80))
    assert result.reduction_percent > 20
    assert result.method == "pillow-bmp"


def test_bmp_already_24bit():
    """BMP already 24-bit with quality >= 70: 0% reduction."""
    info = _make_info(ImageFormat.BMP, file_size=1440054, color_type="rgb")
    result = _predict_bmp(info, OptimizationConfig(quality=80))
    assert result.reduction_percent == 0.0
    assert result.already_optimized


# --- AVIF / HEIC prediction ---


def test_avif_prediction():
    # 800x600, 50KB → bpp=0.104, very low → no savings at q=70
    info = _make_info(ImageFormat.AVIF, file_size=50000)
    result = _predict_avif(info, OptimizationConfig(quality=60))
    assert result.reduction_percent == 0.0
    assert result.method == "none"


def test_avif_prediction_high_bpp():
    # 300x200, 52KB → bpp=0.867, high quality → significant savings
    info = _make_info(ImageFormat.AVIF, width=300, height=200, file_size=52000)
    result = _predict_avif(info, OptimizationConfig(quality=60))
    assert result.reduction_percent > 50
    assert result.method == "avif-reencode"


def test_heic_prediction():
    # 800x600, 50KB → bpp=0.104, very low → no savings at q=90
    info = _make_info(ImageFormat.HEIC, file_size=50000)
    result = _predict_heic(info, OptimizationConfig(quality=80))
    assert result.reduction_percent == 0.0
    assert result.method == "none"


# --- max_reduction cap JPEG screenshot path ---


def test_max_reduction_jpeg_screenshot():
    """JPEG screenshot (high flat ratio) with max_reduction: jpegtran wins."""
    info = _make_info(
        ImageFormat.JPEG,
        file_size=50000,
        estimated_quality=95,
        flat_pixel_ratio=0.9,
        width=1920,
        height=1080,
    )
    config = OptimizationConfig(quality=60, max_reduction=5.0)
    result = predict_reduction(info, ImageFormat.JPEG, config)
    # jpegtran should give high reduction for screenshots
    assert result.reduction_percent > 5


def test_max_reduction_jpeg_high_quality():
    """JPEG with high source quality (>90) and max_reduction: exponential jpegtran bonus."""
    info = _make_info(
        ImageFormat.JPEG,
        file_size=50000,
        estimated_quality=98,
    )
    config = OptimizationConfig(quality=80, max_reduction=5.0)
    result = predict_reduction(info, ImageFormat.JPEG, config)
    assert result.reduction_percent > 5
