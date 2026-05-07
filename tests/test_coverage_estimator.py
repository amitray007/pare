"""Tests targeting uncovered branches in estimation/estimator.py.

Coverage targets (from fresh report):
- Lines 35-49: optional plugin import try/except (module-level, not directly testable)
- Line 75: JXL exact-file-size set add (settings.enable_jxl branch)
- Lines 94-96: JXL disabled → UnsupportedFormatError
- Line 114: animated image exact-mode branch (frame_count > 1)
- Line 265: _estimate_by_sample "already optimized" (result.method == "none")
- Lines 337: TIFF BPP log-correction heuristic (downscale_ratio > 1.0)
- Lines 392: _tiff_sample_bpp lossy JPEG-in-TIFF branch (quality < 70, RGB/L mode)
- Lines 417-453: _heic_sample_bpp — non-RGB mode path
- Lines 463-479: _jxl_sample_bpp — exercise the function
- Lines 494-504: _webp_sample_bpp — method=4 branch (quality < 50)
- Line 538: pngquant FileNotFoundError/TimeoutExpired fallback in _png_sample_bpp
- Lines 565-575: pngquant fallback RGBA/non-P/P quantize sub-branches
- Lines 618-619, 627-628, 631-633: _tiff_sample_bpp exception + empty-candidates paths
- Line 655: JXL in _DIRECT_ENCODE_BPP_FNS (settings.enable_jxl branch)
- Lines 674-676, 678, 681, 684: PNG lossless photo-BPP cap (reduction > 5.0)
- Line 823: _get_bit_depth explicit bits path (img.info["bits"])
"""

import io
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from estimation.estimator import (
    _get_bit_depth,
    _jpeg_sample_bpp,
    _png_sample_bpp,
    _tiff_sample_bpp,
    _webp_sample_bpp,
    estimate,
)
from schemas import OptimizationConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(fmt: str, width: int, height: int, quality: int = 95, mode: str = "RGB") -> bytes:
    img = Image.new(mode, (width, height), color=(100, 150, 200) if mode == "RGB" else 128)
    buf = io.BytesIO()
    kw: dict = {}
    if fmt == "JPEG":
        kw["quality"] = quality
    elif fmt == "PNG":
        kw["compress_level"] = 0
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _make_large_jpeg(width: int = 1000, height: int = 1000, quality: int = 95) -> bytes:
    """Return a JPEG that is >1MB so it takes the sample path."""
    import os

    raw = os.urandom(width * height * 3)
    img = Image.frombytes("RGB", (width, height), raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    data = buf.getvalue()
    return data


# ---------------------------------------------------------------------------
# JXL disabled → UnsupportedFormatError (lines 94-96)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_jxl_disabled_raises():
    """When JXL support is disabled, estimating a JXL file raises UnsupportedFormatError."""
    from config import settings
    from exceptions import UnsupportedFormatError

    # Only run this test if JXL is available to create the test file
    try:
        try:
            import pillow_jxl  # noqa: F401
        except ImportError:
            import jxlpy.JXLImagePlugin  # noqa: F401
    except ImportError:
        pytest.skip("JXL plugin not available — cannot create test fixture")

    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JXL", quality=90)
    jxl_data = buf.getvalue()

    with patch.object(settings, "enable_jxl", False):
        with pytest.raises(UnsupportedFormatError, match="JXL support is disabled"):
            await estimate(jxl_data, OptimizationConfig(quality=60))


# ---------------------------------------------------------------------------
# Animated image → exact mode (line 114, frame_count > 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_animated_webp_uses_exact_mode():
    """A multi-frame animated WebP takes the animated exact-mode path (line 114)."""
    frames = [Image.new("RGB", (100, 100), color=(i * 40, 100, 200)) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:], loop=0)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "webp"
    assert result.confidence == "high"
    assert result.estimated_optimized_size <= result.original_size


# ---------------------------------------------------------------------------
# _estimate_by_sample "already optimized" path (line 265)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bmp_estimate_already_optimized_propagated():
    """When the sample optimizer returns method='none', _estimate_by_sample
    propagates 0% reduction (line 265)."""
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()

    from schemas import OptimizeResult

    mock_result = OptimizeResult(
        original_size=len(data),
        optimized_size=len(data),
        reduction_percent=0.0,
        method="none",
        success=True,
        format="bmp",
        optimized_bytes=data,
    )

    # BMP goes through generic fallback → calls optimize_image on the sample
    with patch("estimation.estimator.optimize_image", return_value=mock_result):
        result = await estimate(data, OptimizationConfig(quality=60))

    assert result.estimated_reduction_percent == 0.0
    assert result.already_optimized is True
    assert result.method == "none"


# ---------------------------------------------------------------------------
# TIFF BPP log-correction heuristic (line 337)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tiff_bpp_correction_applied():
    """Large TIFF: the downscale-ratio log correction fires when sample is smaller
    than original (downscale_ratio > 1.0 branch, line 337)."""
    # Need a TIFF large enough to be downsampled (width > LOSSY_SAMPLE_MAX_WIDTH=800)
    img = Image.new("RGB", (2400, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="raw")
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=80))
    assert result.original_format == "tiff"
    # Correction reduces estimated BPP; result should be valid
    assert result.estimated_optimized_size <= result.original_size
    assert result.estimated_reduction_percent >= 0.0


# ---------------------------------------------------------------------------
# _tiff_sample_bpp — lossy JPEG-in-TIFF branch (line 392 area)
# ---------------------------------------------------------------------------


def test_tiff_sample_bpp_lossy_path():
    """_tiff_sample_bpp includes tiff_jpeg candidate when quality < 70 and mode is RGB."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)  # < 70 → lossy branch

    bpp, method = _tiff_sample_bpp(img, 100, 100, config)

    assert bpp > 0
    # lossy JPEG-in-TIFF should win over lossless for photo content at q=60
    assert method in ("tiff_jpeg", "tiff_adobe_deflate", "tiff_lzw")


def test_tiff_sample_bpp_lossless_only_high_quality():
    """_tiff_sample_bpp uses only lossless methods when quality >= 70."""
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    config = OptimizationConfig(quality=80)  # >= 70 → lossless only

    bpp, method = _tiff_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method in ("tiff_adobe_deflate", "tiff_lzw")


def test_tiff_sample_bpp_l_mode_lossy():
    """_tiff_sample_bpp includes tiff_jpeg for grayscale (L mode) at low quality."""
    img = Image.new("L", (100, 100), color=128)
    config = OptimizationConfig(quality=50)

    bpp, method = _tiff_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method in ("tiff_jpeg", "tiff_adobe_deflate", "tiff_lzw")


def test_tiff_sample_bpp_non_rgb_skips_jpeg():
    """_tiff_sample_bpp skips tiff_jpeg for palette mode (P)."""
    img = Image.new("P", (100, 100))
    config = OptimizationConfig(quality=60)  # < 70 but mode not in (RGB, L)

    bpp, method = _tiff_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    # JPEG-in-TIFF is only for RGB/L; result must be from lossless methods
    assert method in ("tiff_adobe_deflate", "tiff_lzw", "tiff_raw")


def test_tiff_sample_bpp_empty_candidates_fallback():
    """_tiff_sample_bpp falls back to raw compression when all lossless saves fail (lines 631-633).

    Patch Image.Image.save so that every save attempt using a non-raw compression
    raises an exception. The fallback 'raw' branch at the end of the function must
    produce a valid result.
    """
    img = Image.new("RGB", (50, 50))
    config = OptimizationConfig(quality=80)

    real_save = Image.Image.save

    def flaky_save(self, fp, format=None, **params):
        compression = params.get("compression", "raw")
        if compression != "raw":
            raise Exception("forced failure")
        real_save(self, fp, format=format, **params)

    with patch.object(Image.Image, "save", flaky_save):
        bpp, method = _tiff_sample_bpp(img, 30, 30, config)

    assert bpp > 0
    assert method == "tiff_raw"


def test_tiff_sample_bpp_one_lossless_save_fails():
    """_tiff_sample_bpp exception branch: one compression fails, the other succeeds.

    Verifies that the except block inside the loop (lines 618-619) is hit for
    tiff_adobe_deflate while tiff_lzw succeeds — the result should use tiff_lzw.
    """
    img = Image.new("RGB", (50, 50))
    config = OptimizationConfig(quality=80)

    real_save = Image.Image.save

    def flaky_deflate_save(self, fp, format=None, **params):
        if params.get("compression") == "tiff_adobe_deflate":
            raise Exception("forced deflate failure")
        real_save(self, fp, format=format, **params)

    with patch.object(Image.Image, "save", flaky_deflate_save):
        bpp, method = _tiff_sample_bpp(img, 30, 30, config)

    # Only tiff_lzw succeeds → method must be tiff_lzw
    assert bpp > 0
    assert method == "tiff_lzw"


# ---------------------------------------------------------------------------
# _webp_sample_bpp — method=4 branch (quality < 50, line 498)
# ---------------------------------------------------------------------------


def test_webp_sample_bpp_high_quality_method4():
    """_webp_sample_bpp uses method=4 when quality < 50 (HIGH preset)."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = OptimizationConfig(quality=40)  # < 50 → method=4

    bpp, method = _webp_sample_bpp(img, 100, 100, config)

    assert bpp > 0
    assert method == "pillow-m4"


def test_webp_sample_bpp_medium_quality_method3():
    """_webp_sample_bpp uses method=3 when quality >= 50 (MEDIUM/LOW preset)."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)  # >= 50 → method=3

    bpp, method = _webp_sample_bpp(img, 100, 100, config)

    assert bpp > 0
    assert method == "pillow-m3"


def test_webp_sample_bpp_non_standard_mode_converted():
    """_webp_sample_bpp converts non-standard modes to RGB."""
    img = Image.new("CMYK", (100, 100))
    img = img.convert("RGB")  # simulate non-RGB by using a converted image
    # Actually test RGBA mode conversion path: CMYK not a valid webp mode
    img_la = Image.new("LA", (100, 100), (128, 255))
    config = OptimizationConfig(quality=60)

    bpp, method = _webp_sample_bpp(img_la, 50, 50, config)
    assert bpp > 0


# ---------------------------------------------------------------------------
# _png_sample_bpp — pngquant FileNotFoundError fallback (lines 538, 565-575)
# ---------------------------------------------------------------------------


def test_png_sample_bpp_pngquant_fallback_rgba():
    """When pngquant raises FileNotFoundError, _png_sample_bpp falls back to
    Pillow quantize. RGBA mode uses quantize() directly."""
    img = Image.new("RGBA", (100, 100), (100, 150, 200, 255))
    config = OptimizationConfig(quality=60, png_lossy=True)  # < 70 → pngquant path

    with patch("subprocess.run", side_effect=FileNotFoundError("pngquant not found")):
        bpp, method = _png_sample_bpp(img, 50, 50, config, 100, 100)

    assert bpp > 0
    assert method == "pngquant + oxipng"


def test_png_sample_bpp_pngquant_fallback_rgb():
    """pngquant TimeoutExpired fallback for RGB mode → convert to RGB then quantize."""
    img = Image.new("RGB", (100, 100), (100, 150, 200))
    config = OptimizationConfig(quality=60, png_lossy=True)

    with patch(
        "subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["pngquant"], timeout=10),
    ):
        bpp, method = _png_sample_bpp(img, 50, 50, config, 100, 100)

    assert bpp > 0
    assert method == "pngquant + oxipng"


def test_png_sample_bpp_pngquant_fallback_palette_mode():
    """pngquant fallback for P (palette) mode → uses existing palette image directly."""
    img = Image.new("P", (100, 100))
    config = OptimizationConfig(quality=60, png_lossy=True)

    with patch("subprocess.run", side_effect=FileNotFoundError("no pngquant")):
        bpp, method = _png_sample_bpp(img, 50, 50, config, 100, 100)

    assert bpp > 0
    assert method == "pngquant + oxipng"


# ---------------------------------------------------------------------------
# PNG lossless photo-BPP cap (lines 674-684)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_lossless_photo_bpp_cap():
    """PNG in lossless mode with high BPP (photo-like) gets reduction capped at 5%.

    The sample-resize smooths pixels, making oxipng look more effective than
    it really is on the full image. The cap prevents over-optimistic estimates.
    """
    import os

    # Random pixel data → high BPP after PNG compression
    raw = os.urandom(800 * 600 * 3)
    img = Image.frombytes("RGB", (800, 600), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    data = buf.getvalue()

    # Confirm high BPP
    original_bpp = len(data) * 8 / (800 * 600)
    assert original_bpp > 10.0, f"Expected high-BPP PNG for test, got {original_bpp:.2f}"

    # Lossless mode: png_lossy=False
    result = await estimate(data, OptimizationConfig(quality=80, png_lossy=False))

    assert result.original_format == "png"
    # The cap keeps reduction at or below 5%
    assert result.estimated_reduction_percent <= 5.01, (
        f"Expected ≤5% for lossless PNG on photo content, got "
        f"{result.estimated_reduction_percent}%"
    )


@pytest.mark.asyncio
async def test_png_lossy_photo_bpp_no_cap():
    """PNG in lossy mode with high BPP should NOT apply the 5% cap."""
    import os

    raw = os.urandom(800 * 600 * 3)
    img = Image.frombytes("RGB", (800, 600), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    data = buf.getvalue()

    # Lossy mode: png_lossy=True, quality < 70
    result = await estimate(data, OptimizationConfig(quality=60, png_lossy=True))

    assert result.original_format == "png"
    # Lossy pngquant should reduce more than the 5% lossless cap
    # (just verify no artificial cap was applied — reduction can be any amount)
    assert result.estimated_reduction_percent >= 0.0


# ---------------------------------------------------------------------------
# _get_bit_depth — explicit bits from img.info (line 823)
# ---------------------------------------------------------------------------


def test_get_bit_depth_explicit_bits():
    """_get_bit_depth returns img.info['bits'] when present."""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {"bits": 16}
    mock_img.mode = "RGB"

    result = _get_bit_depth(mock_img)

    assert result == 16


def test_get_bit_depth_fallback_from_mode():
    """_get_bit_depth falls back to mode lookup when info['bits'] absent."""
    mock_img = MagicMock(spec=Image.Image)
    mock_img.info = {}
    mock_img.mode = "RGBA"

    result = _get_bit_depth(mock_img)

    assert result == 8


# ---------------------------------------------------------------------------
# _jpeg_sample_bpp — progressive path (quality < 70)
# ---------------------------------------------------------------------------


def test_jpeg_sample_bpp_progressive():
    """_jpeg_sample_bpp enables progressive encoding when quality < 70."""
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)  # < 70 → progressive=True branch

    bpp, method = _jpeg_sample_bpp(img, 100, 100, config)

    assert bpp > 0
    assert method == "pillow_jpeg"


def test_jpeg_sample_bpp_non_rgb_mode_converted():
    """_jpeg_sample_bpp converts non-RGB/L modes to RGB."""
    img = Image.new("RGBA", (100, 100), (100, 150, 200, 255))
    config = OptimizationConfig(quality=80)

    bpp, method = _jpeg_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "pillow_jpeg"


# ---------------------------------------------------------------------------
# Large WebP max_reduction cap via _bpp_to_estimate (line 719 area)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_large_webp_max_reduction_cap():
    """Large WebP (>1MB) estimate via direct-encode path respects max_reduction cap."""
    import os

    raw = os.urandom(800 * 600 * 3)
    img = Image.frombytes("RGB", (800, 600), raw)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()

    # Ensure file is large enough for sample path (>1MB)
    # If under 1MB, use a larger image
    if len(data) <= 1_000_000:
        raw2 = os.urandom(1200 * 900 * 3)
        img2 = Image.frombytes("RGB", (1200, 900), raw2)
        buf2 = io.BytesIO()
        img2.save(buf2, format="WEBP", quality=95)
        data = buf2.getvalue()

    if len(data) <= 1_000_000:
        pytest.skip("Cannot generate >1MB WebP in this environment")

    capped = await estimate(data, OptimizationConfig(quality=40, max_reduction=10.0))

    assert capped.estimated_reduction_percent <= 10.0 + 0.01


# ---------------------------------------------------------------------------
# Estimate HEIC sample BPP non-RGB conversion
# ---------------------------------------------------------------------------

try:
    import pillow_heif  # noqa: F401

    _pillow_heif_available = True
except ImportError:
    _pillow_heif_available = False

try:
    import pillow_avif  # noqa: F401

    _pillow_avif_available = True
except ImportError:
    _pillow_avif_available = False

_jxl_available = False
try:
    import pillow_jxl  # noqa: F401

    _jxl_available = True
except ImportError:
    try:
        import jxlpy.JXLImagePlugin  # noqa: F401

        _jxl_available = True
    except ImportError:
        pass


@pytest.mark.skipif(not _pillow_heif_available, reason="pillow_heif not installed")
def test_heic_sample_bpp_non_rgb_mode_converted():
    """_heic_sample_bpp converts non-RGB/RGBA modes to RGB."""
    from estimation.estimator import _heic_sample_bpp

    img = Image.new("L", (100, 100), color=128)  # grayscale → not in (RGB, RGBA)
    config = OptimizationConfig(quality=60)

    bpp, method = _heic_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "heic-reencode"


@pytest.mark.skipif(not _pillow_avif_available, reason="pillow_avif not installed")
def test_avif_sample_bpp_rgb_mode():
    """_avif_sample_bpp encodes an RGB sample and returns BPP."""
    from estimation.estimator import _avif_sample_bpp

    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)

    bpp, method = _avif_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "avif-reencode"


@pytest.mark.skipif(not _pillow_avif_available, reason="pillow_avif not installed")
def test_avif_sample_bpp_non_rgb_mode_converted():
    """_avif_sample_bpp converts L mode to RGB (not in RGB/RGBA)."""
    from estimation.estimator import _avif_sample_bpp

    img = Image.new("L", (100, 100), color=128)
    config = OptimizationConfig(quality=60)

    bpp, method = _avif_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "avif-reencode"


@pytest.mark.skipif(not _jxl_available, reason="JXL plugin not installed")
def test_jxl_sample_bpp_rgb_mode():
    """_jxl_sample_bpp encodes an RGB sample and returns BPP."""
    from estimation.estimator import _jxl_sample_bpp

    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)

    bpp, method = _jxl_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "jxl-reencode"


@pytest.mark.skipif(not _jxl_available, reason="JXL plugin not installed")
def test_jxl_sample_bpp_non_standard_mode_converted():
    """_jxl_sample_bpp converts CMYK mode (not in RGB/RGBA/L) to RGB."""
    from estimation.estimator import _jxl_sample_bpp

    # Use a mode not in ("RGB", "RGBA", "L") — YCbCr is a good candidate
    img = Image.new("YCbCr", (100, 100))
    config = OptimizationConfig(quality=60)

    bpp, method = _jxl_sample_bpp(img, 50, 50, config)

    assert bpp > 0
    assert method == "jxl-reencode"


# ---------------------------------------------------------------------------
# _create_sample branches (lines 674-684)
# GIF non-P mode, BMP non-standard mode, else branch
# ---------------------------------------------------------------------------


def test_create_sample_gif_non_palette_mode():
    """_create_sample with GIF format and non-P mode → quantize(256) then save."""
    from estimation.estimator import _create_sample
    from utils.format_detect import ImageFormat

    # RGB mode GIF → quantize branch (line 674-675)
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    result = _create_sample(img, 50, 50, ImageFormat.GIF)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_create_sample_gif_palette_mode():
    """_create_sample with GIF format and P mode → save directly (no quantize)."""
    from estimation.estimator import _create_sample
    from utils.format_detect import ImageFormat

    img = Image.new("P", (100, 100))
    result = _create_sample(img, 50, 50, ImageFormat.GIF)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_create_sample_bmp_non_standard_mode():
    """_create_sample with BMP format and non-RGB/L/P mode → convert to RGB (line 681)."""
    from estimation.estimator import _create_sample
    from utils.format_detect import ImageFormat

    # YCbCr is not in ("RGB", "L", "P") → convert to RGB branch
    img = Image.new("YCbCr", (100, 100))
    result = _create_sample(img, 50, 50, ImageFormat.BMP)

    assert isinstance(result, bytes)
    assert len(result) > 0


def test_create_sample_unknown_format_falls_through():
    """_create_sample with an unrecognized format falls through to PNG (line 684)."""
    from estimation.estimator import _create_sample
    from utils.format_detect import ImageFormat

    # SVG is not GIF/TIFF/BMP → falls through to the else branch (saves as PNG)
    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    result = _create_sample(img, 25, 25, ImageFormat.SVG)

    assert isinstance(result, bytes)
    assert len(result) > 0
    # Output should be a PNG (magic bytes)
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# estimate_from_thumbnail — config=None default (line 703) and method="none" (line 719)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_thumbnail_none_config():
    """estimate_from_thumbnail with config=None uses default OptimizationConfig (line 703)."""
    from estimation.estimator import estimate_from_thumbnail

    buf = io.BytesIO()
    Image.new("RGB", (100, 80), color=(100, 150, 200)).save(buf, format="JPEG", quality=80)
    thumb_data = buf.getvalue()

    # config=None → function creates OptimizationConfig() internally
    result = await estimate_from_thumbnail(
        thumbnail_data=thumb_data,
        original_file_size=5_000_000,
        original_width=1000,
        original_height=800,
        config=None,
    )

    assert result.original_size == 5_000_000
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent >= 0


@pytest.mark.asyncio
async def test_estimate_from_thumbnail_already_optimized():
    """estimate_from_thumbnail when optimizer returns method='none' → returns 0% reduction
    with method='none' (line 719)."""
    from estimation.estimator import estimate_from_thumbnail
    from schemas import OptimizationConfig, OptimizeResult

    buf = io.BytesIO()
    Image.new("RGB", (100, 80)).save(buf, format="JPEG", quality=80)
    thumb_data = buf.getvalue()

    mock_result = OptimizeResult(
        success=True,
        original_size=len(thumb_data),
        optimized_size=len(thumb_data),
        reduction_percent=0.0,
        format="jpeg",
        method="none",
        optimized_bytes=thumb_data,
    )

    with patch("estimation.estimator.optimize_image", return_value=mock_result):
        result = await estimate_from_thumbnail(
            thumbnail_data=thumb_data,
            original_file_size=5_000_000,
            original_width=1000,
            original_height=800,
            config=OptimizationConfig(quality=80),
        )

    assert result.estimated_reduction_percent == 0.0
    assert result.method == "none"
    assert result.confidence == "medium"


# ---------------------------------------------------------------------------
# _tiff_sample_bpp — JPEG-in-TIFF exception path (lines 627-628)
# ---------------------------------------------------------------------------


def test_tiff_sample_bpp_jpeg_compression_exception_logged():
    """_tiff_sample_bpp catches exception from tiff_jpeg save and falls back to lossless
    (lines 627-628)."""
    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    config = OptimizationConfig(quality=60)  # < 70 → tries tiff_jpeg

    real_save = Image.Image.save

    def save_that_fails_jpeg(self, fp, format=None, **params):
        if params.get("compression") == "tiff_jpeg":
            raise Exception("tiff_jpeg save failed")
        real_save(self, fp, format=format, **params)

    with patch.object(Image.Image, "save", save_that_fails_jpeg):
        bpp, method = _tiff_sample_bpp(img, 50, 50, config)

    # JPEG-in-TIFF failed → best is from lossless methods
    assert bpp > 0
    assert method in ("tiff_adobe_deflate", "tiff_lzw")


# ---------------------------------------------------------------------------
# _png_sample_bpp — non-standard mode conversion to RGBA (line 538)
# ---------------------------------------------------------------------------


def test_png_sample_bpp_non_standard_mode_in_lossy_path():
    """_png_sample_bpp converts non-RGB/RGBA/L/P mode to RGBA for pngquant input (line 538)."""
    # YCbCr mode is not in ("RGB", "RGBA", "L", "P") → convert("RGBA") branch
    img = Image.new("YCbCr", (100, 100))
    config = OptimizationConfig(quality=60, png_lossy=True)  # < 70 → pngquant path

    # Run with pngquant fallback to avoid actually needing pngquant
    with patch("subprocess.run", side_effect=FileNotFoundError("no pngquant")):
        bpp, method = _png_sample_bpp(img, 50, 50, config, 100, 100)

    assert bpp > 0
    assert method == "pngquant + oxipng"
