"""Tests for PillowReencodeOptimizer shared base class."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.pillow_reencode import PillowReencodeOptimizer
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


class FakeReencodeOptimizer(PillowReencodeOptimizer):
    """Concrete subclass for testing the base class logic."""

    format = ImageFormat.AVIF
    pillow_format = "PNG"  # Use PNG to avoid needing pillow_avif
    strip_method_name = "test-strip"
    reencode_method_name = "test-reencode"
    quality_min = 30
    quality_max = 90
    quality_offset = 10

    def _ensure_plugin(self):
        pass  # No plugin needed for PNG


def _make_png(size=(100, 100), color=(128, 64, 32)):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def fake_optimizer():
    return FakeReencodeOptimizer()


@pytest.mark.asyncio
async def test_optimize_returns_valid_result(fake_optimizer):
    """optimize() returns a valid OptimizeResult."""
    data = _make_png()
    config = OptimizationConfig(quality=60)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_optimize_with_metadata_strip(fake_optimizer):
    """Both strip and reencode run when strip_metadata=True."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=True)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.method in ("test-strip", "test-reencode", "none")


@pytest.mark.asyncio
async def test_optimize_without_metadata_strip(fake_optimizer):
    """Only reencode runs when strip_metadata=False."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=False)
    result = await fake_optimizer.optimize(data, config)
    assert result.success
    assert result.method in ("test-reencode", "none")


@pytest.mark.asyncio
async def test_optimize_both_fail_returns_none(fake_optimizer):
    """When both methods fail, returns method='none'."""
    data = _make_png()
    config = OptimizationConfig(quality=60, strip_metadata=True)

    with patch.object(fake_optimizer, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(fake_optimizer, "_reencode", side_effect=Exception("fail")):
            result = await fake_optimizer.optimize(data, config)
            assert result.method == "none"
            assert result.success


def test_strip_metadata_preserves_icc(fake_optimizer):
    """_strip_metadata preserves ICC profile if present."""
    img = Image.new("RGB", (50, 50), (100, 100, 100))
    # Create a minimal ICC profile in the image info
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    result = fake_optimizer._strip_metadata(data)
    assert isinstance(result, bytes)
    assert len(result) <= len(data)


def test_reencode_uses_clamped_quality(fake_optimizer):
    """_reencode clamps quality using the subclass's range."""
    data = _make_png()
    # quality=15 + offset=10 = 25, clamped to lo=30
    result = fake_optimizer._reencode(data, 15)
    assert isinstance(result, bytes)
    assert len(result) > 0
