"""Tests for AVIF optimizer — re-encoding, metadata strip, quality tiers."""

import io
from unittest.mock import patch

import pytest

try:
    import pillow_avif  # noqa: F401
    from PIL import Image as _Im

    _buf = io.BytesIO()
    _Im.new("RGB", (1, 1)).save(_buf, format="AVIF")
    HAS_AVIF = True
except (ImportError, Exception):
    HAS_AVIF = False

from PIL import Image

from optimizers.avif import AvifOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def avif_optimizer():
    return AvifOptimizer()


def _make_avif(quality=90, size=(100, 100)):
    img = Image.new("RGB", size, (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_basic_optimization(avif_optimizer):
    """AVIF optimizer produces valid output not larger than input."""
    data = _make_avif(quality=95)
    config = OptimizationConfig(quality=60)
    result = await avif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "avif"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_metadata_strip(avif_optimizer):
    """AVIF metadata strip path runs without error."""
    data = _make_avif(quality=90)
    config = OptimizationConfig(quality=80, strip_metadata=True)
    result = await avif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_quality_tiers(avif_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_avif(quality=95, size=(200, 200))
    result_high = await avif_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_low = await avif_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )


@pytest.mark.asyncio
async def test_avif_both_fail():
    """Both methods fail: returns method='none'."""
    opt = AvifOptimizer()
    data = b"\x00" * 100
    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"
