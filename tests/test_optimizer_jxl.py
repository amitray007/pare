"""Tests for JXL optimizer — basic optimization, metadata strip, quality tiers."""

import io

import pytest

try:
    import jxlpy  # noqa: F401

    HAS_JXLPY = True
except ImportError:
    HAS_JXLPY = False

pytestmark = pytest.mark.skipif(not HAS_JXLPY, reason="jxlpy not installed")

from PIL import Image

from optimizers.jxl import JxlOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def jxl_optimizer():
    return JxlOptimizer()


def _make_jxl(quality=90, size=(100, 100)):
    img = Image.new("RGB", size, (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JXL", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_jxl_basic_optimization(jxl_optimizer):
    """JXL optimizer produces valid output not larger than input."""
    data = _make_jxl(quality=95)
    config = OptimizationConfig(quality=60)
    result = await jxl_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "jxl"


@pytest.mark.asyncio
async def test_jxl_metadata_strip(jxl_optimizer):
    """JXL metadata strip path runs without error."""
    data = _make_jxl(quality=90)
    config = OptimizationConfig(quality=80, strip_metadata=True)
    result = await jxl_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_jxl_quality_tiers(jxl_optimizer):
    """Higher quality (lower aggressiveness) produces larger or equal output."""
    data = _make_jxl(quality=95, size=(200, 200))
    result_high = await jxl_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_low = await jxl_optimizer.optimize(data, OptimizationConfig(quality=80))
    # Aggressive quality should produce smaller or equal output
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )


@pytest.mark.asyncio
async def test_jxl_already_optimized(jxl_optimizer):
    """Low-quality JXL at conservative settings returns original."""
    data = _make_jxl(quality=30, size=(64, 64))
    config = OptimizationConfig(quality=80)
    result = await jxl_optimizer.optimize(data, config)
    assert result.success
    # Output should not be larger than input (guarantee)
    assert result.optimized_size <= result.original_size
