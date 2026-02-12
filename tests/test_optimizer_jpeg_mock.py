"""Tests for JPEG optimizer with mocked CLI tools (cjpeg, jpegtran)."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.jpeg import JpegOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def jpeg_optimizer():
    return JpegOptimizer()


def _make_jpeg(quality=85, mode="RGB", size=(100, 80)):
    img = Image.new(mode, size, (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_jpeg_rgba(size=(50, 50)):
    """Create JPEG from RGBA source (converted internally)."""
    img = Image.new("RGBA", size, (128, 64, 32, 255))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def _mock_run_tool(cmd, data, **kwargs):
    """Simulate run_tool returning smaller data."""
    if cmd[0] == "cjpeg":
        # Simulate mozjpeg producing ~60% of input
        return data[: max(1, int(len(data) * 0.6))], b"", 0
    elif cmd[0] == "jpegtran":
        # Simulate jpegtran producing ~90% of input
        return data[: max(1, int(len(data) * 0.9))], b"", 0
    return data, b"", 0


@pytest.mark.asyncio
async def test_jpeg_optimize_basic(jpeg_optimizer):
    """Basic JPEG optimization: picks smallest of mozjpeg vs jpegtran."""
    data = _make_jpeg(quality=95)
    with patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool):
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method in ("mozjpeg", "jpegtran")


@pytest.mark.asyncio
async def test_jpeg_optimize_progressive(jpeg_optimizer):
    """Progressive flag passed to both cjpeg and jpegtran."""
    data = _make_jpeg()
    calls = []

    async def capture_run_tool(cmd, data_in, **kwargs):
        calls.append(cmd)
        return data_in[: max(1, int(len(data_in) * 0.8))], b"", 0

    with patch("optimizers.jpeg.run_tool", side_effect=capture_run_tool):
        await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60, progressive_jpeg=True))

    # Both calls should have -progressive flag
    for call in calls:
        assert "-progressive" in call


@pytest.mark.asyncio
async def test_jpeg_optimize_no_progressive(jpeg_optimizer):
    """No progressive flag when progressive_jpeg=False."""
    data = _make_jpeg()
    calls = []

    async def capture_run_tool(cmd, data_in, **kwargs):
        calls.append(cmd)
        return data_in[: max(1, int(len(data_in) * 0.8))], b"", 0

    with patch("optimizers.jpeg.run_tool", side_effect=capture_run_tool):
        await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60, progressive_jpeg=False))

    for call in calls:
        assert "-progressive" not in call


@pytest.mark.asyncio
async def test_jpeg_max_reduction_triggers_cap(jpeg_optimizer):
    """max_reduction caps mozjpeg when reduction exceeds limit."""
    data = _make_jpeg(quality=95)
    original_size = len(data)
    cjpeg_calls = []

    async def mock_run_tool(cmd, data_in, **kwargs):
        if cmd[0] == "cjpeg":
            quality = int(cmd[cmd.index("-quality") + 1])
            cjpeg_calls.append(quality)
            # Higher quality -> larger output. At q=60 produce 30% of original.
            # Scale: q=60 -> 0.30, q=100 -> 0.95
            ratio = 0.30 + (quality - 60) * 0.01625
            out_size = max(1, int(original_size * ratio))
            return b"\xff" * out_size, b"", 0
        elif cmd[0] == "jpegtran":
            return b"\xff" * max(1, int(original_size * 0.92)), b"", 0
        return data_in, b"", 0

    with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
        result = await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, max_reduction=10.0)
        )
    assert result.success
    # Cap binary search should have triggered additional cjpeg calls beyond initial 1
    assert len(cjpeg_calls) > 1


@pytest.mark.asyncio
async def test_jpeg_max_reduction_q100_exceeds_cap(jpeg_optimizer):
    """max_reduction: even q=100 cjpeg exceeds cap -> returns original data."""
    data = _make_jpeg(quality=95)

    async def mock_run_tool(cmd, data_in, **kwargs):
        if cmd[0] == "cjpeg":
            # Even at q=100, produces only 50% of input
            return data_in[: max(1, int(len(data_in) * 0.5))], b"", 0
        elif cmd[0] == "jpegtran":
            return data_in[: max(1, int(len(data_in) * 0.95))], b"", 0
        return data_in, b"", 0

    with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
        result = await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, max_reduction=0.1)
        )
    assert result.success


@pytest.mark.asyncio
async def test_jpeg_decode_to_bmp_rgba(jpeg_optimizer):
    """decode_to_bmp converts RGBA to RGB."""
    # Create an RGBA image saved as BMP
    img = Image.new("RGBA", (20, 20), (128, 64, 32, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_data = buf.getvalue()
    # JpegOptimizer._decode_to_bmp should handle RGBA mode
    bmp_data = jpeg_optimizer._decode_to_bmp(png_data, False)
    assert bmp_data[:2] == b"BM"


@pytest.mark.asyncio
async def test_jpeg_decode_to_bmp_cmyk(jpeg_optimizer):
    """decode_to_bmp converts CMYK to RGB."""
    img = Image.new("CMYK", (20, 20), (0, 128, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    tiff_data = buf.getvalue()
    bmp_data = jpeg_optimizer._decode_to_bmp(tiff_data, False)
    assert bmp_data[:2] == b"BM"


@pytest.mark.asyncio
async def test_jpeg_jpegtran_wins(jpeg_optimizer):
    """When jpegtran produces smaller output than mozjpeg."""
    data = _make_jpeg()

    async def mock_run_tool(cmd, data_in, **kwargs):
        if cmd[0] == "cjpeg":
            return data_in, b"", 0  # cjpeg returns same size
        elif cmd[0] == "jpegtran":
            return data_in[: max(1, int(len(data_in) * 0.5))], b"", 0  # jpegtran much smaller
        return data_in, b"", 0

    with patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool):
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result.method == "jpegtran"
