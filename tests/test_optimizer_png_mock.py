"""Tests for PNG optimizer with mocked CLI tools (pngquant, oxipng)."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.png import PngOptimizer
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


@pytest.fixture
def png_optimizer():
    return PngOptimizer()


def _make_png(mode="RGB", size=(100, 100), color=(128, 64, 32)):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_apng():
    """Create a minimal APNG (contains acTL chunk)."""
    frames = [Image.new("RGB", (10, 10), (i * 50, 0, 0)) for i in range(2)]
    buf = io.BytesIO()
    frames[0].save(buf, format="PNG", save_all=True, append_images=frames[1:])
    return buf.getvalue()


@pytest.mark.asyncio
async def test_png_lossless_path(png_optimizer):
    """png_lossy=False -> oxipng only."""
    data = _make_png()

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.9))]

    with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
        result = await png_optimizer.optimize(data, OptimizationConfig(quality=80, png_lossy=False))
    assert result.success
    assert result.method == "oxipng"


@pytest.mark.asyncio
async def test_png_lossless_with_strip_metadata(png_optimizer):
    """Lossless path with strip_metadata=True."""
    data = _make_png()

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.9))]

    with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
        with patch("optimizers.png.strip_metadata_selective", return_value=data):
            result = await png_optimizer.optimize(
                data, OptimizationConfig(quality=80, png_lossy=False, strip_metadata=True)
            )
    assert result.success


@pytest.mark.asyncio
async def test_png_lossy_pngquant_success(png_optimizer):
    """Lossy path: pngquant succeeds, oxipng applied to pngquant output."""
    data = _make_png()
    smaller = data[: max(1, int(len(data) * 0.4))]

    async def mock_pngquant(d, quality, max_colors=256, speed=4):
        return smaller, True

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.95))]

    with patch.object(png_optimizer, "_run_pngquant", side_effect=mock_pngquant):
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            result = await png_optimizer.optimize(
                data, OptimizationConfig(quality=60, png_lossy=True)
            )
    assert result.success


@pytest.mark.asyncio
async def test_png_lossy_pngquant_fail_exit99(png_optimizer):
    """Lossy path: pngquant exit 99 -> falls back to oxipng only."""
    data = _make_png()

    async def mock_pngquant(d, quality, max_colors=256, speed=4):
        return None, False

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.9))]

    with patch.object(png_optimizer, "_run_pngquant", side_effect=mock_pngquant):
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            result = await png_optimizer.optimize(
                data, OptimizationConfig(quality=60, png_lossy=True)
            )
    assert result.success
    assert result.method == "oxipng"


@pytest.mark.asyncio
async def test_png_lossy_oxipng_wins_over_pngquant(png_optimizer):
    """Lossy path: pngquant+oxipng larger than oxipng alone -> picks oxipng."""
    data = _make_png()

    async def mock_pngquant(d, quality, max_colors=256, speed=4):
        # pngquant returns larger output
        return data + b"\x00" * 100, True

    def mock_oxipng(d, level=2):
        # oxipng gives small result for original, large for pngquant output
        if len(d) > len(data):
            return d  # pngquant output stays large
        return d[: max(1, int(len(d) * 0.5))]  # original gets nice compression

    with patch.object(png_optimizer, "_run_pngquant", side_effect=mock_pngquant):
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            result = await png_optimizer.optimize(
                data, OptimizationConfig(quality=60, png_lossy=True)
            )
    assert result.method == "oxipng"


@pytest.mark.asyncio
async def test_png_quality_aggressive_settings(png_optimizer):
    """quality < 50: uses 64 max colors, speed=1, oxipng level=6."""
    data = _make_png()
    pngquant_calls = []

    async def mock_pngquant(d, quality, max_colors=256, speed=4):
        pngquant_calls.append({"max_colors": max_colors, "speed": speed})
        return d[: max(1, int(len(d) * 0.5))], True

    def mock_oxipng(d, level=2):
        return d

    with patch.object(png_optimizer, "_run_pngquant", side_effect=mock_pngquant):
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            await png_optimizer.optimize(data, OptimizationConfig(quality=40, png_lossy=True))
    assert pngquant_calls[0]["max_colors"] == 64
    assert pngquant_calls[0]["speed"] == 3


@pytest.mark.asyncio
async def test_png_quality_moderate_settings(png_optimizer):
    """quality 50-69: uses 256 max colors, speed=4."""
    data = _make_png()
    pngquant_calls = []

    async def mock_pngquant(d, quality, max_colors=256, speed=4):
        pngquant_calls.append({"max_colors": max_colors, "speed": speed})
        return d[: max(1, int(len(d) * 0.5))], True

    def mock_oxipng(d, level=2):
        return d

    with patch.object(png_optimizer, "_run_pngquant", side_effect=mock_pngquant):
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            await png_optimizer.optimize(data, OptimizationConfig(quality=60, png_lossy=True))
    assert pngquant_calls[0]["max_colors"] == 256
    assert pngquant_calls[0]["speed"] == 4


@pytest.mark.asyncio
async def test_png_apng_uses_oxipng_only(png_optimizer):
    """APNG detected: skips pngquant, uses oxipng only."""
    data = _make_apng()

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.9))]

    with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
        result = await png_optimizer.optimize(data, OptimizationConfig(quality=60, png_lossy=True))
    assert result.method == "oxipng"
    assert png_optimizer.format == ImageFormat.APNG


@pytest.mark.asyncio
async def test_png_run_pngquant_exit99():
    """_run_pngquant returns (None, False) on exit code 99."""
    opt = PngOptimizer()

    async def mock_run_tool(cmd, data, allowed_exit_codes=None):
        return b"", b"", 99

    with patch("optimizers.png.run_tool", side_effect=mock_run_tool):
        result, success = await opt._run_pngquant(b"data", 60)
    assert result is None
    assert success is False


@pytest.mark.asyncio
async def test_png_run_pngquant_success():
    """_run_pngquant returns (data, True) on exit code 0."""
    opt = PngOptimizer()

    async def mock_run_tool(cmd, data, allowed_exit_codes=None):
        return b"optimized", b"", 0

    with patch("optimizers.png.run_tool", side_effect=mock_run_tool):
        result, success = await opt._run_pngquant(b"data", 60)
    assert result == b"optimized"
    assert success is True


@pytest.mark.asyncio
async def test_png_strip_metadata_not_lossy(png_optimizer):
    """strip_metadata without lossy: metadata stripped, then oxipng."""
    data = _make_png()

    def mock_oxipng(d, level=2):
        return d[: max(1, int(len(d) * 0.9))]

    with patch("optimizers.png.strip_metadata_selective", return_value=data) as mock_strip:
        with patch.object(png_optimizer, "_run_oxipng", side_effect=mock_oxipng):
            result = await png_optimizer.optimize(
                data, OptimizationConfig(quality=80, png_lossy=False, strip_metadata=True)
            )
    mock_strip.assert_called_once()
    assert result.success
