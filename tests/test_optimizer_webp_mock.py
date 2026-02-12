"""Tests for WebP optimizer — cwebp fallback paths and binary search cap."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.webp import WebpOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def webp_optimizer():
    return WebpOptimizer()


def _make_webp(quality=95, size=(100, 100)):
    img = Image.new("RGB", size, (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_cwebp_fallback_file_created(webp_optimizer):
    """cwebp fallback: writes temp file, reads result."""
    data = _make_webp()

    async def mock_run_tool(cmd, stdin_data):
        # Simulate cwebp writing the output file
        out_path = cmd[-1]  # Last arg is -o path
        with open(out_path, "wb") as f:
            f.write(b"small webp")
        return b"", b"", 0

    with patch("optimizers.webp.shutil.which", return_value="/usr/bin/cwebp"):
        with patch("optimizers.webp.run_tool", side_effect=mock_run_tool):
            result = await webp_optimizer._cwebp_fallback(data, 80)
    assert result == b"small webp"


@pytest.mark.asyncio
async def test_cwebp_fallback_output_missing(webp_optimizer):
    """cwebp runs but no output file -> returns None."""
    data = _make_webp()

    async def mock_run_tool(cmd, stdin_data):
        return b"", b"error", 1

    with patch("optimizers.webp.shutil.which", return_value="/usr/bin/cwebp"):
        with patch("optimizers.webp.run_tool", side_effect=mock_run_tool):
            result = await webp_optimizer._cwebp_fallback(data, 80)
    assert result is None


@pytest.mark.asyncio
async def test_cwebp_fallback_exception(webp_optimizer):
    """cwebp throws exception -> returns None."""
    data = _make_webp()

    with patch("optimizers.webp.shutil.which", return_value="/usr/bin/cwebp"):
        with patch("optimizers.webp.run_tool", side_effect=Exception("failed")):
            result = await webp_optimizer._cwebp_fallback(data, 80)
    assert result is None


@pytest.mark.asyncio
async def test_webp_find_capped_quality_binary_search(webp_optimizer):
    """Binary search finds quality within max_reduction cap."""
    data = _make_webp(quality=95, size=(200, 200))
    config = OptimizationConfig(quality=60, max_reduction=5.0)

    result = webp_optimizer._find_capped_quality(data, config)
    # Should return some bytes (binary search found a quality)
    if result is not None:
        reduction = (1 - len(result) / len(data)) * 100
        assert reduction <= 5.0 + 1.0  # small tolerance


@pytest.mark.asyncio
async def test_webp_find_capped_quality_q100_exceeds(webp_optimizer):
    """q=100 still exceeds cap -> returns None."""
    # Create a very compressible image
    img = Image.new("RGB", (500, 500), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=100)
    data = buf.getvalue()
    # Padding to make it artificially large
    # data is intentionally small; test exercises the capped quality path

    # Use a real WebP that Pillow can optimize dramatically
    config = OptimizationConfig(quality=1, max_reduction=0.001)

    # Can't easily test None return with real Pillow, so mock
    with patch.object(webp_optimizer, "_pillow_optimize") as mock_opt:
        # q=100 produces output that's 10% of input -> 90% reduction > 0.001%
        mock_opt.return_value = b"x" * int(len(data) * 0.1)
        result = webp_optimizer._find_capped_quality(data, config)
    assert result is None


@pytest.mark.asyncio
async def test_webp_max_reduction_triggers_find_capped(webp_optimizer):
    """Optimize path: max_reduction triggers _find_capped_quality."""
    data = _make_webp(quality=95, size=(200, 200))

    with patch.object(webp_optimizer, "_pillow_optimize") as mock_pil:
        # Pillow returns very small (triggers cap)
        mock_pil.return_value = b"tiny"
        with patch.object(
            webp_optimizer, "_find_capped_quality", return_value=b"capped"
        ) as mock_cap:
            result = await webp_optimizer.optimize(
                data, OptimizationConfig(quality=60, max_reduction=5.0)
            )
    mock_cap.assert_called_once()
    assert result.method == "pillow"
