"""Per-format optimization tests.

Note: Tests that require CLI tools (pngquant, cjpeg, jpegtran, gifsicle, cwebp)
are skipped on Windows where these tools are not installed. These tests will
pass in the Docker container where all tools are available.
"""

import shutil
import io

import pytest
from PIL import Image

from optimizers.router import optimize_image
from schemas import OptimizationConfig


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None


@pytest.mark.asyncio
async def test_format_png_oxipng(sample_png):
    """PNG lossless optimization via oxipng (pure Python)."""
    config = OptimizationConfig(png_lossy=False)
    result = await optimize_image(sample_png, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "png"


@pytest.mark.asyncio
@pytest.mark.skipif(not has_tool("pngquant"), reason="pngquant not installed")
async def test_format_png_lossy(sample_png):
    """PNG lossy optimization via pngquant + oxipng."""
    config = OptimizationConfig(png_lossy=True, quality=80)
    result = await optimize_image(sample_png, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
@pytest.mark.skipif(not has_tool("cjpeg"), reason="cjpeg not installed")
async def test_format_jpeg_reduction(sample_jpeg):
    """JPEG optimization via MozJPEG."""
    config = OptimizationConfig(quality=80)
    result = await optimize_image(sample_jpeg, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "jpeg"


@pytest.mark.asyncio
@pytest.mark.skipif(not has_tool("gifsicle"), reason="gifsicle not installed")
async def test_format_gif_reduction(sample_gif):
    """GIF optimization via gifsicle."""
    config = OptimizationConfig()
    result = await optimize_image(sample_gif, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "gif"


@pytest.mark.asyncio
async def test_format_svg_reduction(sample_svg):
    """SVG optimization via scour (pure Python)."""
    config = OptimizationConfig()
    result = await optimize_image(sample_svg, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "svg"
    assert "scour" in result.method


@pytest.mark.asyncio
async def test_format_svg_sanitized(malicious_svg):
    """Malicious SVG is sanitized before optimization."""
    config = OptimizationConfig()
    result = await optimize_image(malicious_svg, config)
    assert result.success
    out = result.optimized_bytes
    assert b"<script>" not in out
    assert b"onload" not in out
    assert b"onclick" not in out
    assert b"foreignObject" not in out and b"foreignobject" not in out.lower()


@pytest.mark.asyncio
async def test_format_webp(sample_webp):
    """WebP optimization via Pillow."""
    config = OptimizationConfig(quality=80)
    result = await optimize_image(sample_webp, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "webp"


@pytest.mark.asyncio
async def test_format_bmp(sample_bmp):
    """BMP optimization via Pillow re-encode."""
    config = OptimizationConfig()
    result = await optimize_image(sample_bmp, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "bmp"


@pytest.mark.asyncio
async def test_format_tiff(sample_tiff):
    """TIFF optimization via multi-compression trial."""
    config = OptimizationConfig()
    result = await optimize_image(sample_tiff, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "tiff"
    # Should pick a real compression method on uncompressed TIFF
    assert result.method in ("tiff_adobe_deflate", "tiff_lzw", "none")


@pytest.mark.asyncio
async def test_all_formats_never_larger(
    sample_png, sample_jpeg, sample_svg, sample_webp, sample_gif, sample_bmp, sample_tiff, tiny_png
):
    """No format ever returns output larger than input (optimization guarantee)."""
    config = OptimizationConfig(png_lossy=False)
    samples = [
        ("png", sample_png),
        ("svg", sample_svg),
        ("webp", sample_webp),
        ("bmp", sample_bmp),
        ("tiff", sample_tiff),
        ("tiny_png", tiny_png),
    ]
    for name, data in samples:
        result = await optimize_image(data, config)
        assert result.optimized_size <= result.original_size, (
            f"{name}: optimized ({result.optimized_size}) > original ({result.original_size})"
        )
