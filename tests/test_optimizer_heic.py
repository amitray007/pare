"""Tests for HEIC optimizer — re-encoding, metadata strip, quality tiers."""

import io
from unittest.mock import patch

import pytest

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    from PIL import Image as _Im

    _buf = io.BytesIO()
    _Im.new("RGB", (1, 1)).save(_buf, format="HEIF")
    HAS_HEIC = True
except (ImportError, Exception):
    HAS_HEIC = False

from PIL import Image

from optimizers.heic import HeicOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def heic_optimizer():
    return HeicOptimizer()


def _make_heic(quality=90, size=(100, 100)):
    img = Image.new("RGB", size, (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="HEIF", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_basic_optimization(heic_optimizer):
    """HEIC optimizer produces valid output not larger than input."""
    data = _make_heic(quality=95)
    config = OptimizationConfig(quality=60)
    result = await heic_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "heic"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_metadata_strip(heic_optimizer):
    """HEIC metadata strip path runs without error."""
    data = _make_heic(quality=90)
    config = OptimizationConfig(quality=80, strip_metadata=True)
    result = await heic_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_quality_tiers(heic_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_heic(quality=95, size=(200, 200))
    result_high = await heic_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_low = await heic_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )


@pytest.mark.asyncio
async def test_heic_both_fail():
    """Both methods fail: returns method='none'."""
    opt = HeicOptimizer()
    data = b"\x00" * 100
    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"
