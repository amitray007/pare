"""Tests for estimation/estimator.py — estimate pipeline and helpers."""

import io

import pytest
from PIL import Image

from estimation.estimator import _combine_with_thumbnail, _thumbnail_compress, estimate
from estimation.header_analysis import HeaderInfo
from estimation.heuristics import Prediction
from schemas import OptimizationConfig
from utils.format_detect import ImageFormat


@pytest.mark.asyncio
async def test_estimate_png_default_config():
    """estimate() with default config returns valid EstimateResponse."""
    img = Image.new("RGB", (100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    result = await estimate(data)
    assert result.original_size == len(data)
    assert result.original_format == "png"
    assert result.estimated_reduction_percent >= 0


@pytest.mark.asyncio
async def test_estimate_jpeg_with_config():
    """estimate() with explicit config."""
    img = Image.new("RGB", (100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent >= 0


@pytest.mark.asyncio
async def test_estimate_none_config():
    """estimate() with config=None uses defaults."""
    img = Image.new("RGB", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    result = await estimate(data, None)
    assert result.original_format == "png"


# --- _thumbnail_compress ---


@pytest.mark.asyncio
async def test_thumbnail_compress_jpeg():
    """Thumbnail compress returns a compression ratio for JPEG."""
    img = Image.new("RGB", (200, 200), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    ratio = await _thumbnail_compress(data, ImageFormat.JPEG, 60)
    assert ratio is not None
    assert 0.0 < ratio < 1.0


@pytest.mark.asyncio
async def test_thumbnail_compress_webp():
    """Thumbnail compress returns a ratio for WebP."""
    img = Image.new("RGB", (200, 200), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()
    ratio = await _thumbnail_compress(data, ImageFormat.WEBP, 60)
    assert ratio is not None
    assert 0.0 < ratio < 1.5


@pytest.mark.asyncio
async def test_thumbnail_compress_invalid_data():
    """Invalid image data returns None."""
    ratio = await _thumbnail_compress(b"not an image", ImageFormat.JPEG, 60)
    assert ratio is None


# --- _combine_with_thumbnail ---


def test_combine_close_predictions():
    """Close heuristic and thumbnail -> high confidence."""
    pred = Prediction(
        estimated_size=50000,
        reduction_percent=30.0,
        potential="medium",
        method="jpegli",
        already_optimized=False,
        confidence="medium",
    )
    info = HeaderInfo(format=ImageFormat.JPEG, file_size=100000)
    result = _combine_with_thumbnail(pred, 0.72, info)
    # 0.72 -> 28% thumbnail reduction, close to 30% heuristic
    assert result.confidence == "high"
    assert abs(result.reduction_percent - 29.4) < 1.0


def test_combine_divergent_predictions():
    """Divergent heuristic and thumbnail -> medium confidence."""
    pred = Prediction(
        estimated_size=50000,
        reduction_percent=50.0,
        potential="high",
        method="jpegli",
        already_optimized=False,
        confidence="medium",
    )
    info = HeaderInfo(format=ImageFormat.JPEG, file_size=100000)
    result = _combine_with_thumbnail(pred, 0.90, info)
    # 0.90 -> 10% thumbnail reduction vs 50% heuristic = divergent
    assert result.confidence == "medium"


@pytest.mark.asyncio
async def test_thumbnail_compress_small_jpeg():
    """Cover _thumbnail_compress with a small JPEG."""
    img = Image.new("RGB", (64, 64), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    data = buf.getvalue()

    result = await _thumbnail_compress(data, ImageFormat.JPEG, 60)
    assert result is not None
    assert 0 < result <= 1.5
