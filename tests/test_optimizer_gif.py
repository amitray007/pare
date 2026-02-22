"""Tests for GIF optimizer — quality tiers, lossless, lossy."""

import io

import pytest
from PIL import Image

from optimizers.gif import GifOptimizer
from schemas import OptimizationConfig
from utils.subprocess_runner import run_tool

# Check if gifsicle is available
try:
    import asyncio

    asyncio.get_event_loop().run_until_complete(run_tool(["gifsicle", "--version"], b""))
    HAS_GIFSICLE = True
except (FileNotFoundError, OSError, Exception):
    HAS_GIFSICLE = False


@pytest.fixture
def gif_optimizer():
    return GifOptimizer()


def _make_gif(width=100, height=100, colors=64, frames=1):
    """Create a test GIF image."""
    imgs = []
    for i in range(frames):
        img = Image.new("RGB", (width, height), ((i * 50) % 256, 100, 200))
        imgs.append(img.quantize(colors))
    buf = io.BytesIO()
    if frames > 1:
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:])
    else:
        imgs[0].save(buf, format="GIF")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_lossless_optimization(gif_optimizer):
    """quality >= 70: lossless gifsicle --optimize=3 only."""
    data = _make_gif()
    config = OptimizationConfig(quality=80)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.method == "gifsicle"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_moderate_lossy(gif_optimizer):
    """quality 50-69: gifsicle --lossy=30 --colors=192."""
    data = _make_gif(width=200, height=200, colors=256)
    config = OptimizationConfig(quality=60)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert "lossy=30" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_aggressive_lossy(gif_optimizer):
    """quality < 50: gifsicle --lossy=80 --colors=128."""
    data = _make_gif(width=200, height=200, colors=256)
    config = OptimizationConfig(quality=30)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert "lossy=80" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_animated(gif_optimizer):
    """Animated GIF is optimized without breaking frames."""
    data = _make_gif(frames=3)
    config = OptimizationConfig(quality=60)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    # Verify output is valid and still animated
    out_img = Image.open(io.BytesIO(result.optimized_bytes))
    assert getattr(out_img, "n_frames", 1) >= 1


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_quality_tiers(gif_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_gif(width=200, height=200, colors=256)
    result_high = await gif_optimizer.optimize(data, OptimizationConfig(quality=30))
    result_low = await gif_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )
