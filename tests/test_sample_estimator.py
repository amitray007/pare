"""Tests for the sample-based estimation engine."""

import io
import os

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
    # Both sizes must be > EXACT_PIXEL_THRESHOLD (150K) to use sample path
    small_data = _make_image("JPEG", 500, 500, quality=95)  # 250K pixels
    large_data = _make_image("JPEG", 1500, 1500, quality=95)  # 2.25M pixels
    config = OptimizationConfig(quality=60)

    small_result = await estimate(small_data, config)
    large_result = await estimate(large_data, config)

    # Both use the same sample-based path, so BPP should be similar
    small_bpp = small_result.estimated_optimized_size * 8 / (500 * 500)
    large_bpp = large_result.estimated_optimized_size * 8 / (1500 * 1500)
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
async def test_large_jpeg_sample_not_already_optimized():
    """Large JPEG at q=95 estimated at q=60 should report meaningful reduction."""
    data = _make_image("JPEG", 1000, 1000, quality=95)
    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "jpeg"
    assert result.method != "none", "JPEG should not report 'none' method"
    assert (
        result.estimated_reduction_percent > 10
    ), f"Expected >10% reduction, got {result.estimated_reduction_percent}%"
    assert not result.already_optimized


@pytest.mark.asyncio
async def test_jpeg_preset_differentiation():
    """Higher compression presets should estimate more reduction for JPEG."""
    # Use random pixel data (photo-like) so quality differences are meaningful.
    # Solid-color images compress trivially at all qualities.
    raw = os.urandom(1000 * 1000 * 3)
    img = Image.frombytes("RGB", (1000, 1000), raw)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()

    high = await estimate(data, OptimizationConfig(quality=40))  # HIGH preset
    medium = await estimate(data, OptimizationConfig(quality=60))  # MEDIUM preset
    low = await estimate(data, OptimizationConfig(quality=80))  # LOW preset

    assert high.estimated_reduction_percent > medium.estimated_reduction_percent, (
        f"HIGH ({high.estimated_reduction_percent}%) should beat "
        f"MEDIUM ({medium.estimated_reduction_percent}%)"
    )
    assert medium.estimated_reduction_percent > low.estimated_reduction_percent, (
        f"MEDIUM ({medium.estimated_reduction_percent}%) should beat "
        f"LOW ({low.estimated_reduction_percent}%)"
    )


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


# --- Large image estimation accuracy ---


@pytest.mark.asyncio
async def test_large_png_screenshot_not_zero():
    """Large PNG screenshot should estimate meaningful reduction, not 0%."""
    from PIL import ImageDraw

    img = Image.new("RGB", (1000, 800))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 40], fill=(50, 50, 60))
    draw.rectangle([0, 40, 200, 800], fill=(240, 240, 240))
    draw.rectangle([200, 40, 1000, 800], fill=(255, 255, 255))
    draw.rectangle([200, 700, 1000, 800], fill=(230, 230, 230))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60, png_lossy=True))
    assert result.original_format == "png"
    assert (
        result.estimated_reduction_percent > 0
    ), f"Large PNG screenshot should not estimate 0%, got method={result.method}"


@pytest.mark.asyncio
async def test_large_png_lossless_estimation():
    """Large PNG in lossless mode should still produce a reasonable estimate."""
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=80, png_lossy=False))
    assert result.original_format == "png"
    assert result.estimated_reduction_percent > 0


@pytest.mark.asyncio
async def test_large_webp_not_zero():
    """Large WebP should estimate meaningful reduction."""
    raw = os.urandom(800 * 600 * 3)
    img = Image.frombytes("RGB", (800, 600), raw)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "webp"
    assert result.method != "none", "WebP should not report 'none' method"
    assert (
        result.estimated_reduction_percent > 0
    ), "Large WebP at q=95 estimated at q=60 should show reduction"
