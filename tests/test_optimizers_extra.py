"""Extra tests for optimizers with low coverage: AVIF, HEIC, WebP, GIF, SVG, TIFF."""

import gzip
import io
import shutil
from unittest.mock import patch

import pytest
from PIL import Image

from optimizers.avif import AvifOptimizer
from optimizers.gif import GifOptimizer
from optimizers.heic import HeicOptimizer
from optimizers.svg import SvgOptimizer
from optimizers.tiff import TiffOptimizer
from optimizers.webp import WebpOptimizer
from schemas import OptimizationConfig

# --- AVIF Optimizer ---


@pytest.fixture
def avif_optimizer():
    return AvifOptimizer()


@pytest.mark.asyncio
async def test_avif_no_strip_metadata(avif_optimizer):
    """strip_metadata=False -> returns original."""
    data = b"fake avif data"
    result = await avif_optimizer.optimize(data, OptimizationConfig(strip_metadata=False))
    assert result.method == "none"
    assert result.optimized_bytes == data


@pytest.mark.asyncio
async def test_avif_strip_metadata_failure(avif_optimizer):
    """Metadata stripping exception -> returns original."""
    with patch.object(avif_optimizer, "_strip_metadata", side_effect=Exception("decode error")):
        result = await avif_optimizer.optimize(b"fake", OptimizationConfig(strip_metadata=True))
    assert result.method == "none"


@pytest.mark.asyncio
async def test_avif_strip_metadata_success(avif_optimizer):
    """Successful metadata strip -> smaller output."""
    with patch.object(avif_optimizer, "_strip_metadata", return_value=b"small"):
        result = await avif_optimizer.optimize(
            b"larger original", OptimizationConfig(strip_metadata=True)
        )
    assert result.method == "metadata-strip"


@pytest.mark.asyncio
async def test_avif_strip_metadata_larger(avif_optimizer):
    """Metadata strip produces larger output -> returns original."""
    with patch.object(
        avif_optimizer, "_strip_metadata", return_value=b"this is even larger than the original"
    ):
        result = await avif_optimizer.optimize(b"short", OptimizationConfig(strip_metadata=True))
    assert result.method == "none"


# --- HEIC Optimizer ---


@pytest.fixture
def heic_optimizer():
    return HeicOptimizer()


@pytest.mark.asyncio
async def test_heic_no_strip_metadata(heic_optimizer):
    result = await heic_optimizer.optimize(b"fake heic", OptimizationConfig(strip_metadata=False))
    assert result.method == "none"


@pytest.mark.asyncio
async def test_heic_strip_metadata_failure(heic_optimizer):
    with patch.object(heic_optimizer, "_strip_metadata", side_effect=Exception("error")):
        result = await heic_optimizer.optimize(b"fake", OptimizationConfig(strip_metadata=True))
    assert result.method == "none"


@pytest.mark.asyncio
async def test_heic_strip_metadata_success(heic_optimizer):
    with patch.object(heic_optimizer, "_strip_metadata", return_value=b"sm"):
        result = await heic_optimizer.optimize(b"larger", OptimizationConfig(strip_metadata=True))
    assert result.method == "metadata-strip"


# --- WebP Optimizer ---


@pytest.fixture
def webp_optimizer():
    return WebpOptimizer()


@pytest.mark.asyncio
async def test_webp_pillow_optimization(webp_optimizer):
    """Basic WebP optimization via Pillow."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()
    result = await webp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_webp_cwebp_fallback(webp_optimizer):
    """When Pillow result is poor, tries cwebp fallback."""
    img = Image.new("RGB", (10, 10), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=50)
    data = buf.getvalue()
    # Mock _pillow_optimize to return data that's >= 90% of input (triggering fallback)
    with patch.object(webp_optimizer, "_pillow_optimize", return_value=data):
        with patch.object(webp_optimizer, "_cwebp_fallback", return_value=b"tiny"):
            result = await webp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.method == "cwebp"


@pytest.mark.asyncio
async def test_webp_cwebp_fallback_none(webp_optimizer):
    """cwebp fallback returns None (not available) -> uses Pillow result."""
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=50)
    data = buf.getvalue()
    with patch.object(webp_optimizer, "_pillow_optimize", return_value=data):
        with patch.object(webp_optimizer, "_cwebp_fallback", return_value=None):
            result = await webp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_webp_max_reduction_cap(webp_optimizer):
    """max_reduction caps output."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()
    result = await webp_optimizer.optimize(data, OptimizationConfig(quality=60, max_reduction=5.0))
    assert result.success


@pytest.mark.asyncio
async def test_webp_max_reduction_no_quality_works(webp_optimizer):
    """max_reduction where even q=100 exceeds cap -> returns original."""
    img = Image.new("RGB", (100, 100), (255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()
    result = await webp_optimizer.optimize(data, OptimizationConfig(quality=60, max_reduction=0.01))
    assert result.success
    # Should either return original or very close to original


def test_webp_pillow_animated():
    """Animated WebP uses save_all."""
    opt = WebpOptimizer()
    frames = [Image.new("RGB", (10, 10), (i * 50, 0, 0)) for i in range(3)]
    buf = io.BytesIO()
    frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:], duration=100)
    data = buf.getvalue()
    result = opt._pillow_optimize(data, 60)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_webp_cwebp_not_available(webp_optimizer):
    """cwebp not on PATH -> returns None."""
    with patch("shutil.which", return_value=None):
        result = await webp_optimizer._cwebp_fallback(b"data", 80)
    assert result is None


# --- GIF Optimizer ---


@pytest.fixture
def gif_optimizer():
    return GifOptimizer()


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("gifsicle"), reason="gifsicle not installed")
async def test_gif_lossless(gif_optimizer, sample_gif):
    """quality>=70: lossless gifsicle."""
    result = await gif_optimizer.optimize(sample_gif, OptimizationConfig(quality=80))
    assert result.success
    assert result.method == "gifsicle"


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("gifsicle"), reason="gifsicle not installed")
async def test_gif_lossy_moderate(gif_optimizer, sample_gif):
    """quality 50-69: --lossy=30."""
    result = await gif_optimizer.optimize(sample_gif, OptimizationConfig(quality=60))
    assert result.success
    assert result.method == "gifsicle --lossy=30 --colors=192"


@pytest.mark.asyncio
@pytest.mark.skipif(not shutil.which("gifsicle"), reason="gifsicle not installed")
async def test_gif_lossy_aggressive(gif_optimizer, sample_gif):
    """quality<50: --lossy=80."""
    result = await gif_optimizer.optimize(sample_gif, OptimizationConfig(quality=30))
    assert result.success
    assert result.method == "gifsicle --lossy=80 --colors=128"


# --- SVG Optimizer ---


@pytest.fixture
def svg_optimizer():
    return SvgOptimizer()


@pytest.mark.asyncio
async def test_svg_lossless(svg_optimizer, sample_svg):
    """quality>=70: gentle scour."""
    result = await svg_optimizer.optimize(sample_svg, OptimizationConfig(quality=80))
    assert result.success
    assert result.method == "scour"


@pytest.mark.asyncio
async def test_svg_aggressive(svg_optimizer, sample_svg):
    """quality<50: aggressive precision=3."""
    result = await svg_optimizer.optimize(sample_svg, OptimizationConfig(quality=30))
    assert result.success


@pytest.mark.asyncio
async def test_svg_moderate(svg_optimizer, sample_svg):
    """quality 50-69: moderate precision=5."""
    result = await svg_optimizer.optimize(sample_svg, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_svgz_optimization(svg_optimizer):
    """SVGZ: decompress, optimize, recompress."""
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100" fill="red"/></svg>'
    )
    data = gzip.compress(svg)
    result = await svg_optimizer.optimize(data, OptimizationConfig(quality=30))
    assert result.success
    assert result.format == "svgz"
    # Output should be gzip compressed
    assert result.optimized_bytes[:2] == b"\x1f\x8b"


@pytest.mark.asyncio
async def test_svg_no_strip_metadata(svg_optimizer):
    """strip_metadata=False: still optimizes but without stripping."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100"/></svg>'
    result = await svg_optimizer.optimize(svg, OptimizationConfig(quality=80, strip_metadata=False))
    assert result.success


# --- TIFF Optimizer ---


@pytest.fixture
def tiff_optimizer():
    return TiffOptimizer()


@pytest.mark.asyncio
async def test_tiff_lossless(tiff_optimizer, sample_tiff):
    """quality>=70: lossless compression only."""
    result = await tiff_optimizer.optimize(sample_tiff, OptimizationConfig(quality=80))
    assert result.success
    assert result.method in ("tiff_adobe_deflate", "tiff_lzw", "none")


@pytest.mark.asyncio
async def test_tiff_lossy(tiff_optimizer, sample_tiff):
    """quality<70: tries JPEG-in-TIFF + lossless, picks smallest."""
    result = await tiff_optimizer.optimize(sample_tiff, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_tiff_strip_metadata(tiff_optimizer, sample_tiff):
    """Metadata stripping before optimization."""
    result = await tiff_optimizer.optimize(
        sample_tiff, OptimizationConfig(quality=80, strip_metadata=True)
    )
    assert result.success


@pytest.mark.asyncio
async def test_tiff_no_strip_metadata(tiff_optimizer, sample_tiff):
    """No metadata stripping preserves EXIF/ICC."""
    result = await tiff_optimizer.optimize(
        sample_tiff, OptimizationConfig(quality=80, strip_metadata=False)
    )
    assert result.success


@pytest.mark.asyncio
async def test_tiff_rgba():
    """RGBA TIFF: JPEG-in-TIFF not available (JPEG can't do RGBA)."""
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    opt = TiffOptimizer()
    result = await opt.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method != "tiff_jpeg"
