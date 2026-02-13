"""Tests for JXL optimizer — basic optimization, metadata strip, quality tiers."""

import io
from unittest.mock import MagicMock, patch

import pytest

try:
    try:
        import pillow_jxl  # noqa: F401
    except ImportError:
        import jxlpy  # noqa: F401
    HAS_JXL = True
except ImportError:
    HAS_JXL = False

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


# --- JXL mock tests (work even without jxlpy) ---


@pytest.mark.asyncio
async def test_jxl_optimizer_with_mock():
    """Cover JxlOptimizer _strip_metadata and _reencode via mocking."""
    opt = JxlOptimizer()

    original_data = b"\xff\x0a" + b"\x00" * 500
    small_output = b"\x00" * 100

    mock_jxlpy = MagicMock()
    with patch.dict("sys.modules", {"pillow_jxl": None, "jxlpy": mock_jxlpy}):
        with patch("optimizers.jxl.Image.open") as mock_open:
            mock_img = MagicMock(spec=Image.Image)
            mock_img.info = {}
            mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(small_output))
            mock_open.return_value = mock_img

            result_strip = opt._strip_metadata(original_data)
            assert isinstance(result_strip, bytes)
            assert len(result_strip) <= len(original_data)

            result_reencode = opt._reencode(original_data, quality=60)
            assert isinstance(result_reencode, bytes)


@pytest.mark.asyncio
async def test_jxl_optimizer_both_fail():
    """Cover JxlOptimizer fallback to 'none' when both methods fail."""
    opt = JxlOptimizer()
    data = b"\xff\x0a" + b"\x00" * 100

    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"


@pytest.mark.asyncio
async def test_jxl_optimizer_strip_returns_original():
    """Cover JxlOptimizer._strip_metadata returning original when result is bigger."""
    opt = JxlOptimizer()

    small_data = b"\xff\x0a" + b"\x00" * 10

    mock_jxlpy = MagicMock()
    with patch.dict("sys.modules", {"pillow_jxl": None, "jxlpy": mock_jxlpy}):
        with patch("optimizers.jxl.Image.open") as mock_open:
            mock_img = MagicMock(spec=Image.Image)
            mock_img.info = {}
            mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 500))
            mock_open.return_value = mock_img

            result = opt._strip_metadata(small_data)
            assert result == small_data
