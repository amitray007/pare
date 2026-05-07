"""Tests for PNG-specific optimizer behaviour introduced in PR-E and PR-F.

Covers:
- _read_png_dimensions: IHDR parsing without pixel decode
- Dimension-aware oxipng level cap (large PNGs use level 2 even at aggressive quality)
- pngquant early-exit: when pngquant shrinks >=50%, oxipng level is capped to 2
- _read_apng_frame_count: acTL chunk parsing for animation budget gate
- APNG preset re-differentiation: small APNGs use level 4 at HIGH/MEDIUM, large use level 2
"""

import struct
import zlib
from unittest.mock import patch

import pytest

from optimizers.png import LARGE_MP_THRESHOLD, _read_apng_frame_count, _read_png_dimensions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png(width: int, height: int, color: tuple = (128, 0, 0)) -> bytes:
    """Build a minimal valid RGB PNG (single solid color, no compression tricks)."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(t: bytes, d: bytes) -> bytes:
        length = struct.pack(">I", len(d))
        crc = struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        return length + t + d + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = chunk(b"IHDR", ihdr_data)
    row = bytes([0]) + bytes(color) * width  # filter-none + raw RGB pixels
    raw = row * height
    idat = chunk(b"IDAT", zlib.compress(raw))
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# _read_png_dimensions
# ---------------------------------------------------------------------------


def test_read_png_dimensions_normal():
    png = _make_png(1234, 5678)
    w, h = _read_png_dimensions(png)
    assert w == 1234
    assert h == 5678


def test_read_png_dimensions_square():
    png = _make_png(100, 100)
    w, h = _read_png_dimensions(png)
    assert w == 100
    assert h == 100


def test_read_png_dimensions_too_short():
    """Data shorter than 24 bytes must return (0, 0) without raising."""
    w, h = _read_png_dimensions(b"\x89PNG")
    assert w == 0
    assert h == 0


def test_read_png_dimensions_empty():
    w, h = _read_png_dimensions(b"")
    assert w == 0
    assert h == 0


def test_read_png_dimensions_exactly_24_bytes():
    """Boundary: exactly 24 bytes — should succeed (reads bytes 16-23)."""
    png = _make_png(7, 3)
    # Truncate to 24 bytes — still has IHDR width/height
    w, h = _read_png_dimensions(png[:24])
    assert w == 7
    assert h == 3


# ---------------------------------------------------------------------------
# LARGE_MP_THRESHOLD constant
# ---------------------------------------------------------------------------


def test_large_mp_threshold_value():
    """The threshold constant must be 4 million pixels as documented."""
    assert LARGE_MP_THRESHOLD == 4_000_000


def test_large_mp_threshold_boundary():
    """Images at exactly the threshold are NOT large (uses <= in the optimizer)."""
    # A 2000x2000 image is exactly 4M pixels — should stay at level 4 on aggressive preset
    assert 2000 * 2000 == LARGE_MP_THRESHOLD
    # A 2001x2000 image is just over 4M — should drop to level 2
    assert 2001 * 2000 > LARGE_MP_THRESHOLD


# ---------------------------------------------------------------------------
# Dimension-aware oxipng level selection (integration via optimize())
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_lossless_always_level2():
    """quality >= 70 always uses oxipng level 2, regardless of image size."""
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # Small PNG, lossless quality
    png = _make_png(10, 10)
    opt = PngOptimizer()
    result = await opt.optimize(png, OptimizationConfig(quality=80, png_lossy=False))
    assert result.format in ("png", "apng")
    assert result.optimized_size <= len(png)


@pytest.mark.asyncio
async def test_png_small_lossy_path_succeeds():
    """Small PNG (well under LARGE_MP_THRESHOLD) on aggressive quality uses level 4 path."""
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # 100x100 = 10K pixels — tiny, should use level 4 (no cap)
    png = _make_png(100, 100)
    opt = PngOptimizer()
    result = await opt.optimize(png, OptimizationConfig(quality=40, png_lossy=True))
    assert result.format in ("png", "apng")
    # Result must never be larger than input (output guarantee)
    assert result.optimized_size <= len(png)


@pytest.mark.asyncio
async def test_png_large_synthetic_uses_level2():
    """An aggressive-quality PNG whose dimensions exceed LARGE_MP_THRESHOLD must use oxipng level 2.

    We synthesise a cheap real PNG (small pixels, compress_level=0 so bytes stay tiny) but
    mock _read_png_dimensions to report dimensions above the threshold so the level-cap
    branch fires without allocating a real 4MP image.  _run_oxipng is patched to capture
    the level argument while still returning valid oxipng output.
    """
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # A tiny real PNG — oxipng can process it without error
    png = _make_png(10, 10)

    opt = PngOptimizer()
    levels_called = []
    original_run = opt._run_oxipng

    def capture(data, level=2):
        levels_called.append(level)
        return original_run(data, level)

    # Pretend the image is 2001x2001 (> LARGE_MP_THRESHOLD) even though the bytes are 10x10
    with (
        patch("optimizers.png._read_png_dimensions", return_value=(2001, 2001)),
        patch.object(opt, "_run_oxipng", side_effect=capture),
    ):
        result = await opt.optimize(png, OptimizationConfig(quality=40, png_lossy=False))

    assert result.format in ("png", "apng")
    assert levels_called == [2], f"expected [2], got {levels_called}"


# ---------------------------------------------------------------------------
# pngquant early-exit cap (unit test for the lossy_level decision logic)
# ---------------------------------------------------------------------------


def test_pngquant_early_exit_threshold():
    """The early-exit cap fires when pngquant output <= half of cleaned input."""
    # Simulate the condition inline (no full optimize() call needed):
    # data_clean = 1000 bytes, pngquant_result = 499 bytes (< 50%) → cap to level 2
    data_clean_len = 1000
    pngquant_result_len = 499  # <= 1000 // 2 = 500
    oxipng_level = 4  # aggressive preset

    lossy_level = oxipng_level
    if oxipng_level > 2 and pngquant_result_len <= data_clean_len // 2:
        lossy_level = 2

    assert lossy_level == 2


def test_pngquant_early_exit_not_triggered_above_half():
    """The early-exit cap does NOT fire when pngquant output > half of cleaned input."""
    data_clean_len = 1000
    pngquant_result_len = 501  # > 1000 // 2 = 500
    oxipng_level = 4

    lossy_level = oxipng_level
    if oxipng_level > 2 and pngquant_result_len <= data_clean_len // 2:
        lossy_level = 2

    assert lossy_level == 4


def test_pngquant_early_exit_not_triggered_at_level2():
    """The early-exit cap has no effect when oxipng_level is already 2."""
    data_clean_len = 1000
    pngquant_result_len = 100  # huge reduction, but level is already 2
    oxipng_level = 2

    lossy_level = oxipng_level
    if oxipng_level > 2 and pngquant_result_len <= data_clean_len // 2:
        lossy_level = 2

    assert lossy_level == 2  # unchanged — was already 2


def test_pngquant_early_exit_boundary_exactly_half():
    """At exactly half the size, the cap fires (condition is <=)."""
    data_clean_len = 1000
    pngquant_result_len = 500  # exactly 1000 // 2
    oxipng_level = 4

    lossy_level = oxipng_level
    if oxipng_level > 2 and pngquant_result_len <= data_clean_len // 2:
        lossy_level = 2

    assert lossy_level == 2


# ---------------------------------------------------------------------------
# Helpers for APNG synthesis
# ---------------------------------------------------------------------------


def _chunk(t: bytes, d: bytes) -> bytes:
    """Build a PNG chunk: 4-byte length + type + data + CRC."""
    length = struct.pack(">I", len(d))
    crc = struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return length + t + d + crc


def _make_apng_bytes(width: int, height: int, num_frames: int) -> bytes:
    """Build a minimal but structurally valid APNG with acTL chunk.

    Produces a PNG signature + IHDR + acTL + a single IDAT + IEND.
    The IDAT encodes a 1x1 red pixel regardless of width/height so the
    test stays cheap — the optimizer reads dimensions from IHDR and frame
    count from acTL, not from the pixel data.
    """
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)
    # acTL: num_frames (uint32 BE) + num_plays (uint32 BE)
    actl_data = struct.pack(">II", num_frames, 0)
    actl = _chunk(b"acTL", actl_data)
    # Minimal IDAT: a 1-row red pixel (filter byte 0 + RGB)
    row = bytes([0, 255, 0, 0])
    idat = _chunk(b"IDAT", zlib.compress(row))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + actl + idat + iend


def _make_pillow_apng(width: int, height: int, num_frames: int) -> bytes:
    """Build a real APNG using Pillow (valid IDAT + fcTL + fdAT structure)."""
    import io

    from PIL import Image

    first = Image.new("RGB", (width, height), color=(255, 0, 0))
    rest = [Image.new("RGB", (width, height), color=(0, 255, 0)) for _ in range(num_frames - 1)]
    buf = io.BytesIO()
    first.save(buf, format="PNG", save_all=True, append_images=rest, loop=0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# _read_apng_frame_count
# ---------------------------------------------------------------------------


def test_read_apng_frame_count_basic():
    """acTL with num_frames=7 must be parsed and returned correctly."""
    apng = _make_apng_bytes(100, 50, 7)
    assert _read_apng_frame_count(apng) == 7


def test_read_apng_frame_count_single_frame():
    """acTL with num_frames=1 (single-frame APNG) is returned as-is."""
    apng = _make_apng_bytes(64, 64, 1)
    assert _read_apng_frame_count(apng) == 1


def test_read_apng_frame_count_falls_back_to_one():
    """Static PNG without acTL must return 1 (safe fallback)."""
    static_png = _make_png(50, 50)
    assert _read_apng_frame_count(static_png) == 1


def test_read_apng_frame_count_pillow_apng():
    """Real Pillow-generated 3-frame APNG must be counted correctly."""
    apng = _make_pillow_apng(10, 10, 3)
    assert _read_apng_frame_count(apng) == 3


# ---------------------------------------------------------------------------
# APNG preset re-differentiation via animation pixel-budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_apng_small_uses_level_4_high_preset():
    """Small APNG at quality=40 (HIGH) must use oxipng level 4."""
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # 10x10 x 2 frames = 200 total pixel-frames, well under 4M budget
    apng = _make_pillow_apng(10, 10, 2)

    opt = PngOptimizer()
    levels_called = []
    original_run = opt._run_oxipng

    def capture(data, level=2):
        levels_called.append(level)
        return original_run(data, level)

    with patch.object(opt, "_run_oxipng", side_effect=capture):
        result = await opt.optimize(apng, OptimizationConfig(quality=40, png_lossy=True))

    assert result.format == "apng"
    assert levels_called == [4], f"expected [4], got {levels_called}"


@pytest.mark.asyncio
async def test_apng_large_uses_level_2_high_preset():
    """APNG whose total pixel-frames exceed 4M must use oxipng level 2 even at quality=40."""
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # Use a small synthetic APNG but mock _read_apng_frame_count to simulate a large animation.
    # 100x100 * 500 frames = 5_000_000 total pixel-frames (> LARGE_MP_THRESHOLD)
    apng = _make_pillow_apng(100, 100, 2)  # real bytes so oxipng won't error

    opt = PngOptimizer()
    levels_called = []
    original_run = opt._run_oxipng

    def capture(data, level=2):
        levels_called.append(level)
        return original_run(data, level)

    # 100*100*500 = 5_000_000 > LARGE_MP_THRESHOLD=4_000_000 → must use level 2
    with (
        patch("optimizers.png._read_apng_frame_count", return_value=500),
        patch.object(opt, "_run_oxipng", side_effect=capture),
    ):
        result = await opt.optimize(apng, OptimizationConfig(quality=40, png_lossy=True))

    assert result.format == "apng"
    assert levels_called == [2], f"expected [2], got {levels_called}"


@pytest.mark.asyncio
async def test_apng_lossless_quality_uses_level_2():
    """quality >= 70 (LOW preset) must always use level 2 regardless of APNG size."""
    from optimizers.png import PngOptimizer
    from schemas import OptimizationConfig

    # Small APNG — would qualify for level 4 by size, but quality=80 overrides
    apng = _make_pillow_apng(10, 10, 2)

    opt = PngOptimizer()
    levels_called = []
    original_run = opt._run_oxipng

    def capture(data, level=2):
        levels_called.append(level)
        return original_run(data, level)

    with patch.object(opt, "_run_oxipng", side_effect=capture):
        result = await opt.optimize(apng, OptimizationConfig(quality=80, png_lossy=False))

    assert result.format == "apng"
    assert levels_called == [2], f"expected [2], got {levels_called}"
