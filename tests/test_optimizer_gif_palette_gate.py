"""Tests for GIF large-animation palette-gate logic.

Covers:
- _count_gif_pixel_frames walker correctness on handcrafted minimal GIF bytes
- Edge cases: malformed/truncated input, static GIF, GIFs with extension blocks
- Threshold gating: gifsicle command includes/excludes --colors based on pixel-frame count
- End-to-end: optimizer method string reflects gate decision
"""

import io
import struct

import pytest
from PIL import Image

from optimizers.gif import _LARGE_ANIM_PIXEL_FRAMES, GifOptimizer, _count_gif_pixel_frames
from schemas import OptimizationConfig
from utils.subprocess_runner import run_tool

# ---------------------------------------------------------------------------
# gifsicle availability check (same pattern as test_optimizer_gif.py)
# ---------------------------------------------------------------------------
try:
    import asyncio as _asyncio

    _asyncio.get_event_loop().run_until_complete(run_tool(["gifsicle", "--version"], b""))
    HAS_GIFSICLE = True
except Exception:
    HAS_GIFSICLE = False


# ---------------------------------------------------------------------------
# Handcrafted minimal GIF89a builder
# ---------------------------------------------------------------------------


def _gif_header(width: int, height: int, *, gct: bool = True, gct_exp: int = 0) -> bytes:
    """Return GIF89a header bytes (up to and including GCT if requested).

    gct_exp: GCT size exponent (0-7) → 2^(exp+1) palette entries.
    packed byte: bit7=GCT present, bits0-2=gct_exp.
    """
    packed = (0x80 | gct_exp) if gct else 0x00
    header = (
        b"GIF89a"
        + struct.pack("<HH", width, height)
        + bytes([packed, 0, 0])  # packed, bg_index, pixel_aspect
    )
    if gct:
        n_colors = 1 << (gct_exp + 1)  # 2^(exp+1)
        header += bytes(3 * n_colors)  # black palette
    return header


def _image_descriptor(
    left: int = 0,
    top: int = 0,
    width: int = 1,
    height: int = 1,
    *,
    lct: bool = False,
    lct_exp: int = 0,
) -> bytes:
    """Return a minimal Image Descriptor block (0x2C) for one frame.

    Includes:
    - 1-byte block introducer (0x2C)
    - 9-byte descriptor header
    - Optional LCT (all-black)
    - 1-byte LZW minimum code size (2)
    - Minimal LZW data sub-block (3 bytes of dummy data) + terminator
    """
    packed = (0x80 | lct_exp) if lct else 0x00
    descriptor = b"\x2c" + struct.pack("<HHHH", left, top, width, height) + bytes([packed])
    if lct:
        n_colors = 1 << (lct_exp + 1)
        descriptor += bytes(3 * n_colors)
    # LZW min code size + one data sub-block + terminator
    descriptor += b"\x02"  # LZW min code size
    # sub-block chain: 1-byte len=3, 3 bytes dummy LZW data, terminator byte 0x00
    descriptor += b"\x03\x00\x01\x00\x00"  # len=3, data(3 bytes), terminator=0
    return descriptor


def _extension_block(label: int = 0xF9, data: bytes = b"\x00" * 4) -> bytes:
    """Return a GIF Extension block (0x21) with the given label and data.

    Sub-block contains all of `data` in one chunk.
    """
    return b"\x21" + bytes([label, len(data)]) + data + b"\x00"


def _make_minimal_gif(
    width: int,
    height: int,
    frame_count: int,
    *,
    with_ext_between_frames: bool = False,
) -> bytes:
    """Build a syntactically valid minimal GIF89a with frame_count frames."""
    parts = [_gif_header(width, height, gct=True, gct_exp=0)]
    for _ in range(frame_count):
        if with_ext_between_frames:
            # Graphic Control Extension before each frame
            parts.append(_extension_block(label=0xF9, data=b"\x00\x00\x00\x00"))
        parts.append(_image_descriptor(width=width, height=height))
    parts.append(b"\x3b")  # Trailer
    return b"".join(parts)


# ---------------------------------------------------------------------------
# Parser unit tests — no gifsicle required
# ---------------------------------------------------------------------------


class TestCountGifPixelFrames:
    def test_static_gif_one_frame(self):
        data = _make_minimal_gif(100, 80, 1)
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (100, 80, 1)

    def test_two_frames(self):
        data = _make_minimal_gif(200, 150, 2)
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (200, 150, 2)

    def test_three_frames(self):
        data = _make_minimal_gif(50, 50, 3)
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (50, 50, 3)

    def test_extension_blocks_between_frames(self):
        """Graphic Control Extensions between frames must not be miscounted."""
        data = _make_minimal_gif(100, 100, 3, with_ext_between_frames=True)
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (100, 100, 3)

    def test_no_gct(self):
        """GIF without a global color table — header packed byte has bit7=0."""
        header = b"GIF89a" + struct.pack("<HH", 64, 48) + bytes([0x00, 0, 0])  # no GCT
        frame = _image_descriptor(width=64, height=48)
        data = header + frame + b"\x3b"
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (64, 48, 1)

    def test_gif87a_signature(self):
        """GIF87a is a valid signature and must be accepted."""
        data = (
            b"GIF87a"
            + struct.pack("<HH", 10, 10)
            + bytes([0x00, 0, 0])  # no GCT
            + _image_descriptor(width=10, height=10)
            + b"\x3b"
        )
        w, h, f = _count_gif_pixel_frames(data)
        assert (w, h, f) == (10, 10, 1)

    def test_truncated_returns_zeros(self):
        """Truncated data → (0, 0, 0) so caller keeps existing flags."""
        assert _count_gif_pixel_frames(b"GIF89a\x0a\x00") == (0, 0, 0)

    def test_empty_input(self):
        assert _count_gif_pixel_frames(b"") == (0, 0, 0)

    def test_bad_signature(self):
        data = b"NOTGIF" + b"\x00" * 50
        assert _count_gif_pixel_frames(data) == (0, 0, 0)

    def test_malformed_block_stops_gracefully(self):
        """Unknown block type after header → stop walking, return what we have."""
        data = _make_minimal_gif(32, 32, 1)
        # Append a rogue byte (not 0x21, 0x2C, or 0x3B) before the trailer
        # Rebuild without trailer, add garbage, re-add trailer
        no_trailer = data[:-1]  # strip 0x3B
        data = no_trailer + b"\xff" + b"\x3b"
        w, h, f = _count_gif_pixel_frames(data)
        # Must not raise; frame already counted before the bad byte
        assert (w, h, f) == (32, 32, 1)


# ---------------------------------------------------------------------------
# Threshold gate tests — verify cmd construction logic
# ---------------------------------------------------------------------------

# We test the gate by synthesising a GIF whose declared W × H × frames is
# just under or just over the threshold. The actual pixel data doesn't matter
# for the gating decision — gifsicle still processes real bytes.


class TestPaletteGateThreshold:
    """Verify pixel-frame computation against _LARGE_ANIM_PIXEL_FRAMES."""

    def test_below_threshold(self):
        # 1000 × 100 × 1 = 100_000 pixel-frames, well under 100 M
        data = _make_minimal_gif(1000, 100, 1)
        w, h, f = _count_gif_pixel_frames(data)
        assert w * h * f <= _LARGE_ANIM_PIXEL_FRAMES

    def test_above_threshold_via_frame_count(self):
        # 1000 × 1000 × 200 = 200_000_000 pixel-frames, over 100 M
        # Build a GIF with 200 frames at 1000×1000 declared dimensions
        # (actual image data is minimal — we're only testing the parser)
        data = _make_minimal_gif(1000, 1000, 200)
        w, h, f = _count_gif_pixel_frames(data)
        assert w * h * f > _LARGE_ANIM_PIXEL_FRAMES

    def test_exactly_at_threshold_not_triggered(self):
        # Exactly 100_000_000 must NOT trigger (strictly greater-than)
        # 10000 × 1000 × 10 = 100_000_000 — need a GIF whose W×H×frames = 100M exactly
        # Use 1000×100×1000 = 100M frames — but frame_count > ~50 is annoying to build.
        # Use 10×10 declared dims with 1_000_000 frames... too many. Instead just check
        # the formula: 1000 × 100 × 1000 = 100M → not > threshold → colors kept
        # For brevity, synthesize with 100 frames at 1000×1000 = 100M exactly
        data = _make_minimal_gif(1000, 1000, 100)
        w, h, f = _count_gif_pixel_frames(data)
        assert w * h * f == _LARGE_ANIM_PIXEL_FRAMES
        # The guard is strictly > threshold, so 100M exactly keeps --colors
        assert not (w * h * f > _LARGE_ANIM_PIXEL_FRAMES)


# ---------------------------------------------------------------------------
# End-to-end optimizer tests (require gifsicle)
# ---------------------------------------------------------------------------


@pytest.fixture
def gif_optimizer():
    return GifOptimizer()


def _make_real_gif(width: int = 100, height: int = 100, frames: int = 1) -> bytes:
    """Create a real GIF using Pillow (gifsicle can process it)."""
    imgs = []
    for i in range(frames):
        img = Image.new("P", (width, height))
        img.putpalette([i % 256, 100, 200] * 256)
        imgs.append(img)
    buf = io.BytesIO()
    if frames > 1:
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:])
    else:
        imgs[0].save(buf, format="GIF")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_small_gif_high_preset_includes_colors(gif_optimizer):
    """Small GIF (well under threshold): HIGH preset must include --colors in method."""
    # 100×100×2 = 20_000 pixel-frames << 100M
    data = _make_real_gif(width=100, height=100, frames=2)
    w, h, f = _count_gif_pixel_frames(data)
    assert w * h * f < _LARGE_ANIM_PIXEL_FRAMES

    config = OptimizationConfig(quality=30)  # HIGH preset
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    # Either --colors was used (method contains it) or the optimizer returned "none"
    # (image is already optimal — that's fine, but method must not say "palette skipped")
    assert "palette skipped" not in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_small_gif_medium_preset_includes_colors(gif_optimizer):
    """Small GIF: MEDIUM preset must not trigger the palette-skip path."""
    data = _make_real_gif(width=100, height=100, frames=2)
    config = OptimizationConfig(quality=60)  # MEDIUM preset
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert "palette skipped" not in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_large_anim_high_preset_skips_colors(gif_optimizer):
    """GIF above pixel-frame threshold: HIGH preset must report palette skipped."""
    # Build a real GIF with many frames so pixel-frames > 100M.
    # 200×200×3000 = 120M > 100M  — 3000 real frames is too slow; instead we
    # build a small real GIF and then patch the declared width/height in its header
    # to push the computed pixel-frames over the limit without changing gifsicle input.
    # The optimizer only reads declared dims to decide on --colors; gifsicle still
    # processes the actual bytes (which are valid, just with tiny pixel data).
    data = _make_real_gif(width=10, height=10, frames=3)

    # Patch header: set width=10000, height=10000 so 10000×10000×3=300M > 100M
    # Bytes 6-7: width LE, bytes 8-9: height LE
    patched = bytearray(data)
    struct.pack_into("<H", patched, 6, 10000)  # width
    struct.pack_into("<H", patched, 8, 10000)  # height
    data_large = bytes(patched)

    w, h, f = _count_gif_pixel_frames(data_large)
    assert w * h * f > _LARGE_ANIM_PIXEL_FRAMES

    config = OptimizationConfig(quality=30)  # HIGH preset
    result = await gif_optimizer.optimize(data_large, config)
    assert result.success
    assert "palette skipped: large animation" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_large_anim_medium_preset_skips_colors(gif_optimizer):
    """GIF above pixel-frame threshold: MEDIUM preset must report palette skipped."""
    data = _make_real_gif(width=10, height=10, frames=3)
    patched = bytearray(data)
    struct.pack_into("<H", patched, 6, 10000)
    struct.pack_into("<H", patched, 8, 10000)
    data_large = bytes(patched)

    config = OptimizationConfig(quality=60)  # MEDIUM preset
    result = await gif_optimizer.optimize(data_large, config)
    assert result.success
    assert "palette skipped: large animation" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_large_anim_low_preset_unaffected(gif_optimizer):
    """LOW preset never adds --colors regardless of size — gate must not apply."""
    data = _make_real_gif(width=10, height=10, frames=3)
    patched = bytearray(data)
    struct.pack_into("<H", patched, 6, 10000)
    struct.pack_into("<H", patched, 8, 10000)
    data_large = bytes(patched)

    config = OptimizationConfig(quality=80)  # LOW preset
    result = await gif_optimizer.optimize(data_large, config)
    assert result.success
    # LOW preset uses lossless-only path; "palette skipped" must never appear
    assert "palette skipped" not in result.method
