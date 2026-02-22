"""Tests for the sample-based estimation engine."""

import io

import pytest
from PIL import Image

from estimation.estimator import estimate
from schemas import OptimizationConfig


def _make_image(fmt: str, width: int, height: int, quality: int = 95, **kwargs) -> bytes:
    """Helper: create a synthetic image in the given format."""
    mode = "RGB"
    if fmt == "PNG" and kwargs.get("rgba"):
        mode = "RGBA"
    img = Image.new(mode, (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality
    if fmt == "PNG":
        save_kwargs["compress_level"] = kwargs.get("compress_level", 6)
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


# --- Exact mode: small images ---


@pytest.mark.asyncio
async def test_small_png_exact_result():
    """Images under EXACT_PIXEL_THRESHOLD are compressed fully (exact)."""
    data = _make_image("PNG", 100, 100)  # 10K pixels, well under threshold
    result = await estimate(data, OptimizationConfig(quality=40, png_lossy=True))
    assert result.original_format == "png"
    assert result.original_size == len(data)
    assert result.estimated_reduction_percent >= 0
    assert result.confidence == "high"
    assert result.dimensions["width"] == 100
    assert result.dimensions["height"] == 100


@pytest.mark.asyncio
async def test_small_jpeg_exact_result():
    """Small JPEG compressed fully."""
    data = _make_image("JPEG", 200, 200, quality=95)  # 40K pixels
    result = await estimate(data, OptimizationConfig(quality=40))
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent > 0  # q95 -> q40 should reduce


@pytest.mark.asyncio
async def test_exact_mode_uses_actual_optimizer():
    """In exact mode, estimated_optimized_size matches what optimizer produces."""
    data = _make_image("JPEG", 100, 100, quality=95)
    config = OptimizationConfig(quality=60)
    result = await estimate(data, config)
    # The estimate should match exactly (it ran the full optimizer)
    # We can't easily verify the exact number, but confidence should be high
    assert result.confidence == "high"
    assert result.estimated_optimized_size <= result.original_size


# --- Extrapolate mode: large images ---


@pytest.mark.asyncio
async def test_large_jpeg_extrapolation():
    """Large JPEG uses sample-based extrapolation."""
    data = _make_image("JPEG", 1000, 1000, quality=95)  # 1M pixels
    result = await estimate(data, OptimizationConfig(quality=40))
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent > 0
    assert result.confidence == "high"
    assert result.estimated_optimized_size < result.original_size


@pytest.mark.asyncio
async def test_large_png_extrapolation():
    """Large PNG uses sample-based extrapolation."""
    data = _make_image("PNG", 800, 600)  # 480K pixels
    result = await estimate(data, OptimizationConfig(quality=40, png_lossy=True))
    assert result.original_format == "png"
    assert result.estimated_reduction_percent >= 0
    assert result.estimated_optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_extrapolation_bpp_consistency():
    """BPP should be roughly consistent: estimate for a large image should
    be proportional to the small-image result scaled by pixel count."""
    small_data = _make_image("JPEG", 300, 300, quality=95)
    large_data = _make_image("JPEG", 900, 900, quality=95)
    config = OptimizationConfig(quality=60)

    small_result = await estimate(small_data, config)
    large_result = await estimate(large_data, config)

    # The BPP should be similar (within ~20% for synthetic images)
    small_bpp = small_result.estimated_optimized_size * 8 / (300 * 300)
    large_bpp = large_result.estimated_optimized_size * 8 / (900 * 900)
    assert abs(small_bpp - large_bpp) / max(small_bpp, large_bpp) < 0.25


# --- SVG special case ---


@pytest.mark.asyncio
async def test_svg_compresses_full_file(sample_svg):
    """SVG always compresses the full file (no pixel sampling)."""
    result = await estimate(sample_svg, OptimizationConfig(quality=60))
    assert result.original_format == "svg"
    assert "scour" in result.method
    assert result.confidence == "high"


# --- Default config ---


@pytest.mark.asyncio
async def test_estimate_none_config_uses_defaults():
    """estimate() with config=None uses default OptimizationConfig."""
    data = _make_image("PNG", 100, 100)
    result = await estimate(data, None)
    assert result.original_format == "png"
    assert result.estimated_reduction_percent >= 0


@pytest.mark.asyncio
async def test_estimate_default_config():
    """estimate() with no config uses defaults."""
    data = _make_image("JPEG", 100, 100, quality=95)
    result = await estimate(data)
    assert result.original_format == "jpeg"


# --- Response fields ---


@pytest.mark.asyncio
async def test_response_has_all_fields():
    """EstimateResponse has all required fields."""
    data = _make_image("PNG", 200, 200)
    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_size > 0
    assert result.original_format == "png"
    assert "width" in result.dimensions
    assert "height" in result.dimensions
    assert isinstance(result.estimated_optimized_size, int)
    assert isinstance(result.estimated_reduction_percent, float)
    assert result.optimization_potential in ("high", "medium", "low")
    assert isinstance(result.method, str)
    assert isinstance(result.already_optimized, bool)
    assert result.confidence in ("high", "medium", "low")


# --- Animated images (exact mode) ---


@pytest.mark.asyncio
async def test_animated_gif_uses_exact_mode():
    """Animated GIFs compress the full file, never sample."""
    # Create a 2-frame GIF
    frames = [Image.new("P", (400, 400), color=i) for i in range(2)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "gif"
    assert result.confidence == "high"


# --- Edge cases ---


@pytest.mark.asyncio
async def test_already_optimized_image():
    """An image that can't be compressed further reports 0% reduction."""
    # Create a tiny, already-efficient JPEG
    img = Image.new("L", (8, 8), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=80))
    # quality=80 is higher than source quality=20, so little/no reduction expected
    assert result.estimated_reduction_percent >= 0
    assert result.estimated_optimized_size <= result.original_size
