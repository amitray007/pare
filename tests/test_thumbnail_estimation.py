"""Tests for the large-image thumbnail estimation path."""

import io

import pytest
from PIL import Image

from estimation.estimator import estimate_from_thumbnail
from schemas import OptimizationConfig


def _make_jpeg(width: int, height: int, quality: int = 95) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_thumbnail_estimation_returns_valid_result():
    """Thumbnail-based estimation returns a valid EstimateResponse."""
    # Simulate: original is 3000x2000 JPEG, thumbnail is 300x200
    original_data = _make_jpeg(3000, 2000, quality=95)
    thumbnail_data = _make_jpeg(300, 200, quality=95)

    result = await estimate_from_thumbnail(
        thumbnail_data=thumbnail_data,
        original_file_size=len(original_data),
        original_width=3000,
        original_height=2000,
        config=OptimizationConfig(quality=40),
    )
    assert result.original_format == "jpeg"
    assert result.original_size == len(original_data)
    assert result.dimensions == {"width": 3000, "height": 2000}
    assert result.estimated_reduction_percent > 0
    assert result.estimated_optimized_size < result.original_size


@pytest.mark.asyncio
async def test_thumbnail_estimation_confidence_is_medium():
    """Thumbnail estimates have medium confidence (CDN re-compression artifacts)."""
    thumbnail_data = _make_jpeg(300, 200, quality=95)
    original_data = _make_jpeg(3000, 2000, quality=95)

    result = await estimate_from_thumbnail(
        thumbnail_data=thumbnail_data,
        original_file_size=len(original_data),
        original_width=3000,
        original_height=2000,
        config=OptimizationConfig(quality=40),
    )
    assert result.confidence == "medium"
    assert result.estimated_optimized_size <= result.original_size
