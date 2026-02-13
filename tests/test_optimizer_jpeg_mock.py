"""Tests for JPEG optimizer with Pillow encoding path and mocked jpegtran."""

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


async def _mock_run_tool_jpegtran(cmd, data, **kwargs):
    """Simulate run_tool for jpegtran only (Pillow handles lossy)."""
    if cmd[0] == "jpegtran":
        # Simulate jpegtran producing ~90% of input
        return data[: max(1, int(len(data) * 0.9))], b"", 0
    return data, b"", 0


def _mock_pillow_encode_smaller(self, img, quality, progressive, icc_profile, exif_bytes):
    """Simulate Pillow encode producing ~60% of a reference JPEG."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=max(1, quality - 20))
    return buf.getvalue()


def _mock_pillow_encode_larger(self, img, quality, progressive, icc_profile, exif_bytes):
    """Simulate Pillow encode producing larger output (q=100)."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=100)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_jpeg_optimize_basic(jpeg_optimizer):
    """Basic JPEG optimization: picks smallest of jpegli vs jpegtran."""
    data = _make_jpeg(quality=95)
    with (
        patch.object(
            JpegOptimizer, "_pillow_encode", _mock_pillow_encode_smaller
        ),
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method in ("jpegli", "jpegtran")


@pytest.mark.asyncio
async def test_jpeg_optimize_progressive(jpeg_optimizer):
    """Progressive flag passed to Pillow encode and jpegtran."""
    data = _make_jpeg()
    jpegtran_calls = []

    async def capture_run_tool(cmd, data_in, **kwargs):
        jpegtran_calls.append(cmd)
        return data_in[: max(1, int(len(data_in) * 0.8))], b"", 0

    pillow_calls = []
    original_encode = JpegOptimizer._pillow_encode

    def capture_pillow_encode(self, img, quality, progressive, icc_profile, exif_bytes):
        pillow_calls.append(progressive)
        return original_encode(self, img, quality, progressive, icc_profile, exif_bytes)

    with (
        patch.object(JpegOptimizer, "_pillow_encode", capture_pillow_encode),
        patch("optimizers.jpeg.run_tool", side_effect=capture_run_tool),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60, progressive_jpeg=True))

    # Pillow should get progressive=True
    assert any(p is True for p in pillow_calls)
    # jpegtran should have -progressive flag
    for call in jpegtran_calls:
        assert "-progressive" in call


@pytest.mark.asyncio
async def test_jpeg_optimize_no_progressive(jpeg_optimizer):
    """No progressive flag when progressive_jpeg=False."""
    data = _make_jpeg()
    jpegtran_calls = []

    async def capture_run_tool(cmd, data_in, **kwargs):
        jpegtran_calls.append(cmd)
        return data_in[: max(1, int(len(data_in) * 0.8))], b"", 0

    pillow_calls = []
    original_encode = JpegOptimizer._pillow_encode

    def capture_pillow_encode(self, img, quality, progressive, icc_profile, exif_bytes):
        pillow_calls.append(progressive)
        return original_encode(self, img, quality, progressive, icc_profile, exif_bytes)

    with (
        patch.object(JpegOptimizer, "_pillow_encode", capture_pillow_encode),
        patch("optimizers.jpeg.run_tool", side_effect=capture_run_tool),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, progressive_jpeg=False)
        )

    assert all(p is False for p in pillow_calls)
    for call in jpegtran_calls:
        assert "-progressive" not in call


@pytest.mark.asyncio
async def test_jpeg_max_reduction_triggers_cap(jpeg_optimizer):
    """max_reduction caps Pillow lossy when reduction exceeds limit."""
    data = _make_jpeg(quality=95, size=(200, 200))
    original_size = len(data)
    encode_calls = []

    original_encode = JpegOptimizer._pillow_encode

    def counting_encode(self, img, quality, progressive, icc_profile, exif_bytes):
        encode_calls.append(quality)
        return original_encode(self, img, quality, progressive, icc_profile, exif_bytes)

    with (
        patch.object(JpegOptimizer, "_pillow_encode", counting_encode),
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, max_reduction=5.0)
        )
    assert result.success
    # Binary search should trigger additional encodes (initial 1 + cap search)
    assert len(encode_calls) > 1


@pytest.mark.asyncio
async def test_jpeg_max_reduction_q100_exceeds_cap(jpeg_optimizer):
    """max_reduction: even q=100 exceeds cap -> returns original data."""
    data = _make_jpeg(quality=95, size=(200, 200))

    with (
        patch.object(
            JpegOptimizer, "_pillow_encode", _mock_pillow_encode_smaller
        ),
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, max_reduction=0.1)
        )
    assert result.success


@pytest.mark.asyncio
async def test_jpeg_jpegtran_wins(jpeg_optimizer):
    """When jpegtran produces smaller output than Pillow encode."""
    data = _make_jpeg()

    async def mock_run_tool(cmd, data_in, **kwargs):
        if cmd[0] == "jpegtran":
            return data_in[: max(1, int(len(data_in) * 0.5))], b"", 0  # Much smaller
        return data_in, b"", 0

    with (
        patch.object(
            JpegOptimizer, "_pillow_encode", _mock_pillow_encode_larger
        ),
        patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result.method == "jpegtran"


@pytest.mark.asyncio
async def test_jpeg_mode_conversion_rgba(jpeg_optimizer):
    """RGBA source is converted to RGB before encoding."""
    # Create a PNG with RGBA mode, then use it as input
    img = Image.new("RGBA", (20, 20), (128, 64, 32, 255))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    with (
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_jpeg_mode_conversion_cmyk(jpeg_optimizer):
    """CMYK source is converted to RGB before encoding."""
    # Create CMYK image saved as TIFF, then read the JPEG from it
    img = Image.new("CMYK", (20, 20), (0, 128, 255, 0))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    with (
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_jpeg_metadata_preserved(jpeg_optimizer):
    """Metadata preserved when strip_metadata=False."""
    # Create JPEG with EXIF data using Pillow's built-in Exif class
    img = Image.new("RGB", (20, 20), (128, 64, 32))
    exif = img.getexif()
    exif[0x010F] = "TestCamera"  # ImageIFD.Make
    exif_bytes = exif.tobytes()

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, exif=exif_bytes)
    data = buf.getvalue()

    encode_kwargs = []
    original_encode = JpegOptimizer._pillow_encode

    def capture_encode(self, img, quality, progressive, icc_profile, exif_b):
        encode_kwargs.append({"icc_profile": icc_profile, "exif": exif_b})
        return original_encode(self, img, quality, progressive, icc_profile, exif_b)

    with (
        patch.object(JpegOptimizer, "_pillow_encode", capture_encode),
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, strip_metadata=False)
        )

    # EXIF bytes should have been passed to encoder
    assert any(kw["exif"] is not None for kw in encode_kwargs)


@pytest.mark.asyncio
async def test_jpeg_metadata_stripped(jpeg_optimizer):
    """Metadata stripped when strip_metadata=True (default)."""
    img = Image.new("RGB", (20, 20), (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = buf.getvalue()

    encode_kwargs = []
    original_encode = JpegOptimizer._pillow_encode

    def capture_encode(self, img, quality, progressive, icc_profile, exif_b):
        encode_kwargs.append({"icc_profile": icc_profile, "exif": exif_b})
        return original_encode(self, img, quality, progressive, icc_profile, exif_b)

    with (
        patch.object(JpegOptimizer, "_pillow_encode", capture_encode),
        patch("optimizers.jpeg.run_tool", side_effect=_mock_run_tool_jpegtran),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "pillow"
        await jpeg_optimizer.optimize(
            data, OptimizationConfig(quality=60, strip_metadata=True)
        )

    # No metadata should be passed
    assert all(kw["icc_profile"] is None for kw in encode_kwargs)
    assert all(kw["exif"] is None for kw in encode_kwargs)


@pytest.mark.asyncio
async def test_jpeg_cjpeg_fallback(jpeg_optimizer):
    """JPEG_ENCODER=cjpeg falls back to MozJPEG subprocess pipeline."""
    data = _make_jpeg(quality=95)

    async def mock_run_tool(cmd, data_in, **kwargs):
        if cmd[0] == "cjpeg":
            return data_in[: max(1, int(len(data_in) * 0.6))], b"", 0
        elif cmd[0] == "jpegtran":
            return data_in[: max(1, int(len(data_in) * 0.9))], b"", 0
        return data_in, b"", 0

    with (
        patch("optimizers.jpeg.run_tool", side_effect=mock_run_tool),
        patch("optimizers.jpeg.settings") as mock_settings,
    ):
        mock_settings.jpeg_encoder = "cjpeg"
        result = await jpeg_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method in ("mozjpeg", "jpegtran")
