"""Extra tests for optimizers with low coverage: AVIF, HEIC, WebP, GIF, SVG, TIFF."""

import gzip
import io
import shutil
from unittest.mock import MagicMock, patch

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


@pytest.mark.asyncio
async def test_avif_reencode_success(avif_optimizer):
    """Re-encoding produces smaller output."""
    with patch.object(avif_optimizer, "_reencode", return_value=b"small"):
        result = await avif_optimizer.optimize(
            b"larger original avif", OptimizationConfig(strip_metadata=False)
        )
    assert result.method == "avif-reencode"


@pytest.mark.asyncio
async def test_avif_reencode_beats_strip(avif_optimizer):
    """Re-encoding smaller than metadata strip -> picks reencode."""
    with (
        patch.object(avif_optimizer, "_strip_metadata", return_value=b"medium_size"),
        patch.object(avif_optimizer, "_reencode", return_value=b"tiny"),
    ):
        result = await avif_optimizer.optimize(
            b"original avif data here", OptimizationConfig(strip_metadata=True)
        )
    assert result.method == "avif-reencode"


@pytest.mark.asyncio
async def test_avif_both_fail(avif_optimizer):
    """Both strip and reencode fail -> returns original."""
    with (
        patch.object(avif_optimizer, "_strip_metadata", side_effect=Exception("fail")),
        patch.object(avif_optimizer, "_reencode", side_effect=Exception("fail")),
    ):
        result = await avif_optimizer.optimize(b"original", OptimizationConfig(strip_metadata=True))
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


@pytest.mark.asyncio
async def test_heic_reencode_success(heic_optimizer):
    """Re-encoding produces smaller output."""
    with patch.object(heic_optimizer, "_reencode", return_value=b"small"):
        result = await heic_optimizer.optimize(
            b"larger original heic", OptimizationConfig(strip_metadata=False)
        )
    assert result.method == "heic-reencode"


@pytest.mark.asyncio
async def test_heic_reencode_beats_strip(heic_optimizer):
    """Re-encoding smaller than metadata strip -> picks reencode."""
    with (
        patch.object(heic_optimizer, "_strip_metadata", return_value=b"medium_size"),
        patch.object(heic_optimizer, "_reencode", return_value=b"tiny"),
    ):
        result = await heic_optimizer.optimize(
            b"original heic data here", OptimizationConfig(strip_metadata=True)
        )
    assert result.method == "heic-reencode"


@pytest.mark.asyncio
async def test_heic_both_fail(heic_optimizer):
    """Both strip and reencode fail -> returns original."""
    with (
        patch.object(heic_optimizer, "_strip_metadata", side_effect=Exception("fail")),
        patch.object(heic_optimizer, "_reencode", side_effect=Exception("fail")),
    ):
        result = await heic_optimizer.optimize(b"original", OptimizationConfig(strip_metadata=True))
    assert result.method == "none"


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


# --- HEIC _reencode coverage ---


@pytest.mark.asyncio
async def test_heic_reencode():
    """Cover HeicOptimizer._reencode."""
    import pillow_heif

    pillow_heif.register_heif_opener()

    opt = HeicOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = img

    with patch.object(pillow_heif, "open_heif", return_value=mock_heif_file):
        result = opt._reencode(b"\x00" * 100, quality=60)
        assert isinstance(result, bytes)
        assert len(result) > 0


@pytest.mark.asyncio
async def test_heic_reencode_with_icc():
    """Cover HeicOptimizer._reencode ICC profile preservation."""
    import pillow_heif

    pillow_heif.register_heif_opener()

    opt = HeicOptimizer()

    img = Image.new("RGB", (64, 64), (100, 150, 200))
    img.info["icc_profile"] = b"\x00" * 100

    mock_heif_file = MagicMock()
    mock_heif_file.to_pillow.return_value = img

    with patch.object(pillow_heif, "open_heif", return_value=mock_heif_file):
        result = opt._reencode(b"\x00" * 100, quality=40)
        assert isinstance(result, bytes)


# --- AVIF additional coverage ---


@pytest.mark.asyncio
async def test_avif_strip_returns_original_when_bigger():
    """Cover AvifOptimizer._strip_metadata returning original."""
    opt = AvifOptimizer()

    original_data = b"\x00" * 200

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 500))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            result = opt._strip_metadata(original_data)
            assert result == original_data


@pytest.mark.asyncio
async def test_avif_reencode_with_icc():
    """Cover AvifOptimizer._reencode ICC profile line."""
    opt = AvifOptimizer()

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {"icc_profile": b"\x00" * 50}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 100))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            result = opt._reencode(b"\x00" * 200, quality=60)
            assert isinstance(result, bytes)
            call_kwargs = mock_img.save.call_args[1]
            assert "icc_profile" in call_kwargs


@pytest.mark.asyncio
async def test_avif_strip_with_icc():
    """Cover AvifOptimizer._strip_metadata ICC profile line."""
    opt = AvifOptimizer()

    with patch("optimizers.avif.Image.open") as mock_open:
        mock_img = MagicMock(spec=Image.Image)
        mock_img.info = {"icc_profile": b"\x00" * 50}
        mock_img.save = MagicMock(side_effect=lambda buf, **kw: buf.write(b"\x00" * 10))
        mock_open.return_value = mock_img

        mock_avif = MagicMock()
        with patch.dict("sys.modules", {"pillow_avif": mock_avif}):
            opt._strip_metadata(b"\x00" * 200)
            call_kwargs = mock_img.save.call_args[1]
            assert "icc_profile" in call_kwargs


# --- TIFF additional coverage ---


@pytest.mark.asyncio
async def test_tiff_exif_preservation():
    """Cover TIFF _try_compression with exif and strip_metadata=False."""
    opt = TiffOptimizer()

    img = Image.new("RGB", (32, 32), (100, 150, 200))
    buf = io.BytesIO()
    exif_bytes = b"Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x00\x00\x00\x00\x00"
    img.save(buf, format="TIFF", exif=exif_bytes)
    data = buf.getvalue()

    config = OptimizationConfig(quality=80, strip_metadata=False)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)


@pytest.mark.asyncio
async def test_tiff_compression_failure():
    """Cover TIFF _try_compression exception path."""
    opt = TiffOptimizer()

    img = Image.new("RGBA", (32, 32), (100, 150, 200, 128))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()

    config = OptimizationConfig(quality=40)
    result = await opt.optimize(data, config)
    assert result.original_size == len(data)
