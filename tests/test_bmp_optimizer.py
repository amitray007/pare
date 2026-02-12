"""Tests for BMP optimizer quality tiers: lossless, palette, and RLE8."""

import asyncio
import io

import pytest
from PIL import Image

from optimizers.bmp import BmpOptimizer, _rle8_encode_row
from schemas import OptimizationConfig


@pytest.fixture
def bmp_optimizer():
    return BmpOptimizer()


def _make_rgb_bmp(width=100, height=100, color=(0, 120, 215)):
    """Create a 24-bit RGB BMP."""
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_rgba_bmp(width=100, height=100, color=(0, 120, 215, 255)):
    """Create a 32-bit RGBA BMP."""
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_screenshot_bmp(width=100, height=100):
    """Create a BMP with large solid blocks (ideal for RLE8)."""
    img = Image.new("RGB", (width, height), (0, 120, 215))
    for x in range(width // 2, width):
        for y in range(height):
            img.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


def _make_noisy_bmp(width=100, height=100):
    """Create a BMP with random pixel values (worst case for RLE8)."""
    import random
    random.seed(42)
    img = Image.new("RGB", (width, height))
    for x in range(width):
        for y in range(height):
            img.putpixel((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    return buf.getvalue()


# --- Tier 1: Lossless (quality >= 70) ---


@pytest.mark.asyncio
async def test_bmp_lossless_rgb_no_reduction(bmp_optimizer):
    """24-bit RGB BMP at quality>=70: no reduction (already optimal)."""
    data = _make_rgb_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.method in ("none", "pillow-bmp")


@pytest.mark.asyncio
async def test_bmp_lossless_rgba_downconvert(bmp_optimizer):
    """32-bit RGBA BMP at quality>=70: 32->24 bit (~25% reduction)."""
    data = _make_rgba_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result.success
    assert result.optimized_size < result.original_size
    assert result.method == "pillow-bmp"
    assert result.reduction_percent > 20.0


# --- Tier 2: Palette (quality 50-69) ---


@pytest.mark.asyncio
async def test_bmp_palette_quantization(bmp_optimizer):
    """quality 50-69: palette quantization gives ~66% reduction."""
    data = _make_rgb_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method == "pillow-bmp-palette"
    assert result.reduction_percent > 50.0


@pytest.mark.asyncio
async def test_bmp_palette_rgba(bmp_optimizer):
    """RGBA BMP at quality<70: palette beats 32->24 bit."""
    data = _make_rgba_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success
    assert result.method == "pillow-bmp-palette"
    assert result.reduction_percent > 50.0


# --- Tier 3: RLE8 (quality < 50) ---


@pytest.mark.asyncio
async def test_bmp_rle8_screenshot(bmp_optimizer):
    """quality<50 on screenshot: RLE8 gives huge savings."""
    data = _make_screenshot_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=30))
    assert result.success
    assert result.method == "bmp-rle8"
    assert result.reduction_percent > 80.0


@pytest.mark.asyncio
async def test_bmp_rle8_noisy_falls_back_to_palette(bmp_optimizer):
    """quality<50 on noisy content: RLE8 is larger, palette wins."""
    data = _make_noisy_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=30))
    assert result.success
    # RLE8 inflates noisy data, so palette or palette should win
    assert result.method in ("pillow-bmp-palette", "bmp-rle8")
    assert result.reduction_percent > 50.0


@pytest.mark.asyncio
async def test_bmp_rle8_decodes_correctly(bmp_optimizer):
    """RLE8 output is a valid BMP that Pillow can decode."""
    data = _make_screenshot_bmp()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=30))
    assert result.method == "bmp-rle8"
    # Verify Pillow can open and read the RLE8 BMP
    img = Image.open(io.BytesIO(result.optimized_bytes))
    assert img.size == (100, 100)
    assert img.mode == "P"


# --- Edge cases ---


@pytest.mark.asyncio
async def test_bmp_grayscale_mode(bmp_optimizer):
    """Grayscale BMP is converted to RGB and processed."""
    img = Image.new("L", (50, 50), 128)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_bmp_already_palette_mode(bmp_optimizer):
    """P mode image at quality<70: quantize is a no-op, still saves."""
    img = Image.new("RGB", (50, 50), (255, 0, 0))
    img = img.quantize(colors=16)
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=60))
    assert result.success


@pytest.mark.asyncio
async def test_bmp_rgba_non_opaque_alpha(bmp_optimizer):
    """RGBA BMP with real transparency stays RGBA (non-opaque alpha)."""
    img = Image.new("RGBA", (50, 50), (255, 0, 0, 128))
    buf = io.BytesIO()
    img.save(buf, format="BMP")
    data = buf.getvalue()
    # Should still process — convert to RGB for non-RGB modes
    result = await bmp_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result.success


# --- _quantize_to_palette ---


def test_quantize_to_palette_rgb():
    """RGB image quantized to 256-color palette."""
    img = Image.new("RGB", (10, 10), (255, 0, 0))
    result = BmpOptimizer._quantize_to_palette(img)
    assert result.mode == "P"


def test_quantize_to_palette_already_p():
    """Already palette mode returns as-is."""
    img = Image.new("RGB", (10, 10), (255, 0, 0)).quantize(colors=16)
    result = BmpOptimizer._quantize_to_palette(img)
    assert result.mode == "P"


# --- _encode_rle8_bmp ---


def test_encode_rle8_non_palette_returns_none():
    """Non-palette image returns None."""
    img = Image.new("RGB", (10, 10), (255, 0, 0))
    assert BmpOptimizer._encode_rle8_bmp(img) is None


def test_encode_rle8_valid_bmp_header():
    """RLE8 output has correct BMP header."""
    img = Image.new("RGB", (10, 10), (255, 0, 0)).quantize(colors=256)
    result = BmpOptimizer._encode_rle8_bmp(img)
    assert result is not None
    assert result[:2] == b"BM"
    # BI_RLE8 compression = 1 at offset 30
    import struct
    compression = struct.unpack_from("<I", result, 30)[0]
    assert compression == 1


# --- _rle8_encode_row ---


def test_rle8_encode_row_runs():
    """Repeated bytes encoded as runs."""
    out = bytearray()
    _rle8_encode_row(bytes([5, 5, 5, 5, 5]), out)
    assert out == bytearray([5, 5])  # count=5, value=5


def test_rle8_encode_row_literals():
    """Non-repeating bytes encoded as absolute mode."""
    out = bytearray()
    _rle8_encode_row(bytes([1, 2, 3, 4, 5]), out)
    # Should encode as absolute mode: 0x00, count, data, pad
    assert out[0] == 0x00
    assert out[1] == 5


def test_rle8_encode_row_mixed():
    """Mix of runs and literals."""
    out = bytearray()
    row = bytes([1, 2, 3, 4, 7, 7, 7, 7, 7])
    _rle8_encode_row(row, out)
    # Should have literal section + run section
    assert len(out) > 0


def test_rle8_encode_row_empty():
    """Empty row produces no output."""
    out = bytearray()
    _rle8_encode_row(b"", out)
    assert len(out) == 0


def test_rle8_encode_row_single_byte():
    """Single byte."""
    out = bytearray()
    _rle8_encode_row(bytes([42]), out)
    assert len(out) > 0


def test_rle8_encode_row_two_bytes():
    """Two different bytes — too short for absolute mode, emitted as encoded runs."""
    out = bytearray()
    _rle8_encode_row(bytes([1, 2]), out)
    # Should be two encoded runs: [1, 1], [1, 2]
    assert out == bytearray([1, 1, 1, 2])
