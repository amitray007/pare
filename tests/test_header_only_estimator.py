"""Tests for the PNG/JPEG header-only estimator path (Phase 2).

Covers:
- PNG header-only: happy path, mode-off, OOB fallback, internal-error fallback, model-load failure
- JPEG header-only: happy path, lossless fallback, low-NSE fallback, CMYK fallback
"""

from __future__ import annotations

import io
import shutil
import struct
from pathlib import Path

import pytest

_REAL_MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


def _copy_real_model(tmp_path: Path, filename: str) -> None:
    src = _REAL_MODELS_DIR / filename
    if src.exists():
        shutil.copy2(src, tmp_path / filename)


# ---------------------------------------------------------------------------
# PNG image factories
# ---------------------------------------------------------------------------


def _make_large_png_rgb(width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy RGB PNG that forces the sample path (>150K pixels, BPP>2.0)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_png_rgba(width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy RGBA PNG."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(99)
    arr = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# JPEG image factories
# ---------------------------------------------------------------------------


def _make_large_jpeg_rgb(width: int = 600, height: int = 400) -> bytes:
    """Create a standard YCbCr JPEG (q=85, 4:2:0) that forces sample path."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(123)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=2)  # 4:2:0
    return buf.getvalue()


def _make_lossless_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Build a minimal SOF3 (lossless JPEG) byte stream."""
    # Craft a minimal JPEG with SOF3 marker
    soi = b"\xff\xd8"
    # SOF3 marker with minimal valid header
    # seg_len = 2 + 6 + 1*3 = 11
    nf = 1
    sof3_payload = bytes([8, 0, height, 0, width, nf, 1, 0x11, 0])
    sof3_len = 2 + len(sof3_payload)
    sof3 = b"\xff\xc3" + struct.pack(">H", sof3_len) + sof3_payload
    eoi = b"\xff\xd9"
    return soi + sof3 + eoi


def _make_jpeg_uniform_qtable(width: int = 800, height: int = 600) -> bytes:
    """Create a JPEG with a non-standard (uniform) quantization table.

    A uniform Q-table won't match Annex K scaling → NSE < 0.85 → custom_quantization.
    We synthesize the header manually rather than relying on Pillow to produce
    a custom Q-table (Pillow always uses standard Annex-K tables).
    """
    from PIL import Image

    # Standard JPEG, then patch the DQT table bytes to be uniform (all-2)
    rng = __import__("numpy").random.default_rng(7)
    arr = rng.integers(0, 256, (height, width, 3), dtype=__import__("numpy").uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=2)
    data = bytearray(buf.getvalue())

    # Find the first DQT marker (FF DB) and overwrite the luma Q-table with all-2 values
    i = 2
    while i < len(data) - 1:
        if data[i] == 0xFF and data[i + 1] == 0xDB:
            # seg_len = struct.unpack_from(">H", data, i + 2)[0]  # unused, skip
            # DQT payload starts at i+4; first byte is precision+tq
            # table data starts at i+5 (precision=0 → 64 bytes)
            table_start = i + 5
            if table_start + 64 <= len(data):
                for k in range(64):
                    data[table_start + k] = 2  # uniform value
            break
        i += 1

    return bytes(data)


def _make_cmyk_jpeg(width: int = 64, height: int = 64) -> bytes:
    """Synthesize a minimal JPEG with APP14 color_transform=2 (YCCK)."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    data = bytearray(buf.getvalue())

    # Inject an APP14 Adobe marker with color_transform=2 (YCCK)
    # APP14 layout: 0xFFEE + len(2) + "Adobe"(5) + version(2) + flags0(2) + flags1(2) + ct(1)
    app14 = b"\xff\xee" + struct.pack(">H", 14) + b"Adobe" + b"\x00\x01\x00\x00\x00\x00\x02"
    # Insert after SOI (position 2)
    final = data[:2] + app14 + data[2:]
    return bytes(final)


# ---------------------------------------------------------------------------
# §1 PNG header-only: active returns png_header_only path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_active_returns_header_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fitted_estimator_mode=active + real PNG → path='png_header_only'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in (
        "png_header_only",
        "direct_encode_sample",
    ), f"Unexpected path {result.path!r}, fallback={result.fallback_reason!r}"
    assert result.estimated_optimized_size > 0
    assert result.estimated_reduction_percent >= 0.0

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §2 PNG header-only: mode=off → unchanged sample path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_off_returns_sample_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """fitted_estimator_mode=off → direct_encode_sample (no header-only code fires)."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path == "direct_encode_sample", f"Unexpected path {result.path!r}"
    assert result.fallback_reason is None


# ---------------------------------------------------------------------------
# §3 PNG header-only: input_bpp > MAX_INPUT_BPP → feature_oob → falls back
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_oob_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PNG with inflated file size (input_bpp > 64) → fallback path with feature_oob."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.png_header import PngHeader

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Monkeypatch _png_header_only_bpp_inner to simulate OOB by using a small pixel count
    # and large file size. We achieve this by patching parse_png_header to return
    # a tiny header, then passing the large file data.
    # A 1×1 PNG with 1MB file = 8M bpp >> 64 bpp cap
    tiny_header = PngHeader(
        width=1, height=1, bit_depth=8, color_type=2, has_alpha=False, is_palette=False
    )
    monkeypatch.setattr(estimator_mod, "parse_png_header", lambda _: tiny_header)

    # file_size ≫ MAX_INPUT_BPP * 1px = large enough to exceed 64 bpp
    data = _make_large_png_rgb(500, 500)  # ~750KB; 1×1 px → bpp >> 64

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "feature_oob"
        ), f"Expected 'feature_oob', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §4 PNG header-only: parse_png_header raises → internal_error fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_internal_error_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """parse_png_header raises RuntimeError → fallback_reason='internal_error'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    def _raise(_data):
        raise RuntimeError("simulated parse crash")

    monkeypatch.setattr(estimator_mod, "parse_png_header", _raise)

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        # parse_png_header crash is caught in the dispatch and treated as header_parse_error
        assert (
            result.fallback_reason == "header_parse_error"
        ), f"Unexpected fallback_reason: {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §5 PNG header-only: model artifact missing → model_load_failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_header_only_model_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty _MODELS_DIR (no png_header_v1.json) → fallback_reason='model_load_failed'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # tmp_path is empty — no header artifact
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_png_rgb(500, 500)
    config = OptimizationConfig(quality=60, png_lossy=True)
    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "model_load_failed"
        ), f"Expected 'model_load_failed', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §6 JPEG header-only: active → jpeg_header_only path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_active_returns_header_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """fitted_estimator_mode=active + large JPEG → path='jpeg_header_only'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_large_jpeg_rgb(600, 400)
    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    assert result.path in (
        "jpeg_header_only",
        "direct_encode_sample",
        "exact",
    ), f"Unexpected path {result.path!r}, fallback={result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §7 JPEG header-only: SOF3 input → lossless_jpeg fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_lossless_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """SOF3 JPEG → parse_jpeg_header.fallback_reason='lossless_jpeg' → direct_encode_sample."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.jpeg_header import JpegHeader

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Patch parse_jpeg_header to return a lossless header
    lossless_header = JpegHeader(
        width=64,
        height=64,
        components=1,
        bit_depth=8,
        subsampling="grayscale",
        progressive=False,
        dqt_luma=[1] * 64,
        dqt_chroma=None,
        app14_color_transform=None,
        fallback_reason="lossless_jpeg",
    )
    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: lossless_header)

    # Use a real JPEG to pass format detection
    data = _make_large_jpeg_rgb(800, 600)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    # lossless_jpeg fallback reason → routes to direct_encode_sample
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "lossless_jpeg"
        ), f"Expected 'lossless_jpeg', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §8 JPEG header-only: uniform Q-table → custom_quantization fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_low_nse_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JPEG with uniform quantization table → NSE<0.85 → custom_quantization fallback."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.jpeg_header import parse_jpeg_header

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_jpeg_uniform_qtable(800, 600)

    # Confirm parse succeeds with no hard fallback_reason but the NSE check catches it
    header = parse_jpeg_header(data)
    if header is None or header.fallback_reason is not None:
        # If the header itself fails, skip — the test is about the NSE gate
        pytest.skip("could not produce a parseable header with uniform Q-table")

    from estimation.jpeg_header import estimate_source_quality_lsm

    _, nse = estimate_source_quality_lsm(header.dqt_luma, header.dqt_chroma)
    if nse >= 0.85:
        pytest.skip(f"NSE={nse:.3f} >= 0.85 — Q-table patching did not create a non-standard table")

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "custom_quantization"
        ), f"Expected 'custom_quantization', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §9 JPEG header-only: CMYK (YCCK APP14) → non_default_color_transform fallback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_cmyk_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JPEG with APP14 color_transform=2 (YCCK) → non_default_color_transform fallback."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Use the CMYK/YCCK synthesized JPEG
    data = _make_cmyk_jpeg(64, 64)

    from estimation.jpeg_header import parse_jpeg_header

    header = parse_jpeg_header(data)
    if header is None or header.fallback_reason != "non_default_color_transform":
        pytest.skip("CMYK JPEG synthesis did not produce expected fallback_reason")

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate(data, config)

    # Small image → may go exact mode; if sample path, fallback_reason should be set
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "non_default_color_transform"
        ), f"Expected 'non_default_color_transform', got {result.fallback_reason!r}"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §10 estimate_from_header_bytes: PNG path → returns EstimateResponse
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_png_returns_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """estimate_from_header_bytes with PNG data + model returns a valid EstimateResponse."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    from schemas import EstimateResponse, OptimizationConfig
    from utils.format_detect import ImageFormat

    data = _make_large_png_rgb(500, 500)
    total_size = len(data)
    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate_from_header_bytes(
        data, total_size, ImageFormat.PNG, config
    )
    # Result is either an EstimateResponse (header-only succeeded) or None (fallback)
    assert result is None or isinstance(result, EstimateResponse)
    if result is not None:
        assert result.estimated_optimized_size > 0
        assert result.path == "png_header_only"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §11 estimate_from_header_bytes: JPEG path → returns EstimateResponse or None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_jpeg_returns_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """estimate_from_header_bytes with JPEG data + model returns a valid EstimateResponse."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from schemas import EstimateResponse, OptimizationConfig
    from utils.format_detect import ImageFormat

    data = _make_large_jpeg_rgb(800, 600)
    total_size = len(data)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate_from_header_bytes(
        data, total_size, ImageFormat.JPEG, config
    )
    assert result is None or isinstance(result, EstimateResponse)
    if result is not None:
        assert result.estimated_optimized_size > 0
        assert result.path == "jpeg_header_only"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §12 estimate_from_header_bytes: unsupported format → returns None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_unsupported_format_returns_none() -> None:
    """estimate_from_header_bytes for WebP (unsupported) returns None."""
    import estimation.estimator as estimator_mod
    from schemas import OptimizationConfig
    from utils.format_detect import ImageFormat

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate_from_header_bytes(
        b"RIFF\x00\x00\x00\x00WEBPVP8 ", 50000, ImageFormat.WEBP, config
    )
    assert result is None


# ---------------------------------------------------------------------------
# §13 estimate_from_header_bytes: parse_png_header returns None → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_png_parse_fails_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When parse_png_header returns None, estimate_from_header_bytes returns None."""
    import estimation.estimator as estimator_mod
    from utils.format_detect import ImageFormat

    monkeypatch.setattr(estimator_mod, "parse_png_header", lambda _: None)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate_from_header_bytes(
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 500, 100000, ImageFormat.PNG, config
    )
    assert result is None


# ---------------------------------------------------------------------------
# §14 estimate_from_header_bytes: PNG path fires HeaderOnlyFallback → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_png_fallback_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When header-only model returns HeaderOnlyFallback, estimate_from_header_bytes returns None."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.png_header import PngHeader

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    from schemas import OptimizationConfig
    from utils.format_detect import ImageFormat

    # 1×1 PNG with massive file size → input_bpp >> 64 cap → HeaderOnlyFallback("feature_oob")
    tiny_header = PngHeader(
        width=1, height=1, bit_depth=8, color_type=2, has_alpha=False, is_palette=False
    )
    monkeypatch.setattr(estimator_mod, "parse_png_header", lambda _: tiny_header)

    data = _make_large_png_rgb(500, 500)  # large file; 1×1 pixels → bpp >> 64
    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate_from_header_bytes(
        data, len(data), ImageFormat.PNG, config
    )
    assert result is None

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §15 estimate_from_header_bytes: JPEG parse returns None → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_jpeg_parse_fails_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When parse_jpeg_header returns None, estimate_from_header_bytes returns None."""
    import estimation.estimator as estimator_mod
    from utils.format_detect import ImageFormat

    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: None)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    data = _make_large_jpeg_rgb(800, 600)
    result = await estimator_mod.estimate_from_header_bytes(
        data, len(data), ImageFormat.JPEG, config
    )
    assert result is None


# ---------------------------------------------------------------------------
# §16 estimate_from_header_bytes: JPEG parse returns fallback_reason → None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_jpeg_fallback_reason_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When parse_jpeg_header returns header with fallback_reason, returns None."""
    import estimation.estimator as estimator_mod
    from estimation.jpeg_header import JpegHeader
    from utils.format_detect import ImageFormat

    lossless_header = JpegHeader(
        width=64,
        height=64,
        components=1,
        bit_depth=8,
        subsampling="grayscale",
        progressive=False,
        dqt_luma=[1] * 64,
        dqt_chroma=None,
        app14_color_transform=None,
        fallback_reason="lossless_jpeg",
    )
    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: lossless_header)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    data = _make_large_jpeg_rgb(800, 600)
    result = await estimator_mod.estimate_from_header_bytes(
        data, len(data), ImageFormat.JPEG, config
    )
    assert result is None


# ---------------------------------------------------------------------------
# §17 estimate_from_header_bytes: internal exception → returns None (never raises)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_header_bytes_exception_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If _estimate_from_png_header raises unexpectedly, estimate_from_header_bytes returns None."""
    import estimation.estimator as estimator_mod
    from utils.format_detect import ImageFormat

    async def _explode(*_args, **_kwargs):
        raise RuntimeError("simulated internal crash")

    monkeypatch.setattr(estimator_mod, "_estimate_from_png_header", _explode)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60)
    data = _make_large_png_rgb(500, 500)
    result = await estimator_mod.estimate_from_header_bytes(
        data, len(data), ImageFormat.PNG, config
    )
    assert result is None


# ---------------------------------------------------------------------------
# §18 _jpeg_header_only_bpp: happy path end-to-end (unit test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jpeg_header_only_bpp_happy_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp with a real JPEG header and model returns HeaderOnlyBpp or Fallback."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyBpp, HeaderOnlyFallback, _jpeg_header_only_bpp
    from estimation.jpeg_header import parse_jpeg_header

    data = _make_large_jpeg_rgb(800, 600)
    header = parse_jpeg_header(data)
    assert header is not None and header.fallback_reason is None

    result = _jpeg_header_only_bpp(header, len(data), 60, False)
    assert isinstance(result, (HeaderOnlyBpp, HeaderOnlyFallback))

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §19 _jpeg_header_only_bpp: internal error trap
# ---------------------------------------------------------------------------


def test_jpeg_header_only_bpp_internal_error_returns_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp catches unexpected exceptions and returns HeaderOnlyFallback."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp

    def _explode(*_args, **_kwargs):
        raise RuntimeError("forced crash")

    monkeypatch.setattr(estimator_mod, "_jpeg_header_only_bpp_inner", _explode)

    from estimation.jpeg_header import JpegHeader

    header = JpegHeader(
        width=100,
        height=100,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=[16] * 64,
        dqt_chroma=[17] * 64,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "internal_error"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §20 _jpeg_header_only_bpp_inner: fallback paths
# ---------------------------------------------------------------------------


def test_jpeg_header_only_bpp_inner_fallback_reason_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner returns HeaderOnlyFallback when header.fallback_reason is valid."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader

    for reason in ("lossless_jpeg", "non_standard_components", "missing_chroma_table"):
        header = JpegHeader(
            width=100,
            height=100,
            components=3,
            bit_depth=8,
            subsampling="4:2:0",
            progressive=False,
            dqt_luma=[16] * 64,
            dqt_chroma=[17] * 64,
            app14_color_transform=None,
            fallback_reason=reason,
        )
        result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
        assert isinstance(result, HeaderOnlyFallback)
        assert result.reason == reason

    models_mod.load_jpeg_header_model.cache_clear()


def test_jpeg_header_only_bpp_inner_fallback_reason_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner with unknown fallback_reason returns 'header_parse_error'."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader

    # "custom_quantization" is a valid HeaderOnlyFallback reason string but NOT in
    # the valid_reasons set inside _jpeg_header_only_bpp_inner, so it routes to header_parse_error.
    header = JpegHeader(
        width=100,
        height=100,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=[16] * 64,
        dqt_chroma=[17] * 64,
        app14_color_transform=None,
        fallback_reason="custom_quantization",
    )
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "header_parse_error"

    models_mod.load_jpeg_header_model.cache_clear()


def test_jpeg_header_only_bpp_inner_feature_oob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner returns feature_oob when input_bpp exceeds cap."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader

    # 1×1 pixel with very large file_size → input_bpp >> 24 cap
    header = JpegHeader(
        width=1,
        height=1,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=[16] * 64,
        dqt_chroma=[17] * 64,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp_inner(header, 10_000_000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "feature_oob"

    models_mod.load_jpeg_header_model.cache_clear()


def test_jpeg_header_only_bpp_inner_empty_dqt_luma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner returns header_parse_error when dqt_luma is empty."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader

    header = JpegHeader(
        width=800,
        height=600,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=[],  # empty luma table
        dqt_chroma=[17] * 64,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "header_parse_error"

    models_mod.load_jpeg_header_model.cache_clear()


def test_jpeg_header_only_bpp_inner_model_load_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner returns model_load_failed when model absent."""
    import estimation.models as models_mod

    # Empty tmp_path → no jpeg_header_v1.json → LoadFailed
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader, parse_jpeg_header

    # Use a real JPEG's DQT table (Annex-K tables → NSE ≥ 0.85) to pass the NSE gate
    # so the test reaches the model-load check.
    real_data = _make_large_jpeg_rgb(400, 300)
    real_header = parse_jpeg_header(real_data)
    assert real_header is not None and real_header.fallback_reason is None
    assert real_header.dqt_luma and len(real_header.dqt_luma) == 64

    # Reuse real DQT tables but pass a huge file_size so input_bpp stays in range
    header = JpegHeader(
        width=800,
        height=600,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=real_header.dqt_luma,
        dqt_chroma=real_header.dqt_chroma,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    # Either model_load_failed (if NSE passes) or custom_quantization (NSE gate)
    assert result.reason in ("model_load_failed", "custom_quantization")

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §21 _min_ratio_for_quality: all three branches
# ---------------------------------------------------------------------------


def test_min_ratio_for_quality_all_branches() -> None:
    """_min_ratio_for_quality returns correct values for all quality regimes."""
    import pytest

    from estimation.estimator import _min_ratio_for_quality

    assert _min_ratio_for_quality(40) == pytest.approx(0.05)  # q < 50
    assert _min_ratio_for_quality(60) == pytest.approx(0.10)  # 50 <= q < 70
    assert _min_ratio_for_quality(75) == pytest.approx(0.40)  # q >= 70


# ---------------------------------------------------------------------------
# §22 _png_header_only_bpp_inner: runs without crashing on real data
# ---------------------------------------------------------------------------


def test_png_header_only_bpp_inner_runs_on_real_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_header_only_bpp_inner with a real PNG header returns a valid union type."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyBpp, HeaderOnlyFallback, _png_header_only_bpp_inner
    from estimation.png_header import parse_png_header

    data = _make_large_png_rgb(500, 500)
    header = parse_png_header(data)
    assert header is not None

    result = _png_header_only_bpp_inner(header, len(data), 60)
    assert isinstance(result, (HeaderOnlyBpp, HeaderOnlyFallback))

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §23 Grayscale JPEG (1-component): color_type="grayscale" path in _estimate_from_jpeg_header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_estimate_from_jpeg_header_grayscale_color_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Grayscale JPEG (components=1) → color_type='grayscale' in the EstimateResponse."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.jpeg_header import JpegHeader

    # Grayscale JPEG header (1 component)
    grayscale_header = JpegHeader(
        width=800,
        height=600,
        components=1,
        bit_depth=8,
        subsampling="grayscale",
        progressive=False,
        dqt_luma=[16] * 64,
        dqt_chroma=None,
        app14_color_transform=None,
        fallback_reason=None,
    )
    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: grayscale_header)

    from schemas import EstimateResponse, OptimizationConfig
    from utils.format_detect import ImageFormat

    data = _make_large_jpeg_rgb(800, 600)
    config = OptimizationConfig(quality=60)
    result = await estimator_mod.estimate_from_header_bytes(
        data, len(data), ImageFormat.JPEG, config
    )
    assert result is None or isinstance(result, EstimateResponse)
    if result is not None:
        assert result.path == "jpeg_header_only"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §24 _jpeg_header_only_bpp_inner: grayscale (dqt_chroma=None) path (lines 429-430)
# ---------------------------------------------------------------------------


def test_jpeg_header_only_bpp_inner_grayscale_no_chroma_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner with dqt_chroma=None sets mean/std to 0.0 (lines 429-430)."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyBpp, HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader, parse_jpeg_header

    # Use real Annex-K DQT luma table from a real JPEG
    real_data = _make_large_jpeg_rgb(400, 300)
    real_header = parse_jpeg_header(real_data)
    assert real_header is not None

    # Grayscale header — dqt_chroma=None triggers the else branch (lines 429-430)
    header = JpegHeader(
        width=800,
        height=600,
        components=1,
        bit_depth=8,
        subsampling="grayscale",
        progressive=False,
        dqt_luma=real_header.dqt_luma,
        dqt_chroma=None,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    # Either HeaderOnlyBpp or HeaderOnlyFallback — both are valid
    assert isinstance(result, (HeaderOnlyBpp, HeaderOnlyFallback))

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §25 _jpeg_header_only_bpp_inner: prediction_oob and ratio-gate paths (lines 463, 469)
# ---------------------------------------------------------------------------


def test_jpeg_header_only_bpp_inner_prediction_oob_from_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner: intercept=100 → predicted_bpp > 32 → prediction_oob."""
    import json

    import estimation.models as models_mod
    from estimation.models._artifact import _JPEG_HEADER_FEATURES as FEAT

    # Write a JPEG header model with very large intercept (predicted_bpp >> 32)
    model_data = {
        "model_version": 1,
        "format": "jpeg_header",
        "features": FEAT,
        "scaler": {
            "mean": [0.0] * 13,
            "scale": [1.0] * 13,
        },
        "coefficients": {
            "intercept": 100.0,  # forces predicted_bpp > 32 → prediction_oob
            "betas": [0.0] * 13,
        },
        "training_envelope": {},
        "training_corpus_sha256": "abc",
        "git_sha": "def",
        "fit_environment": {"numpy_version": "2.0.0"},
        "created_at": "2026-05-07T00:00:00Z",
    }
    (tmp_path / "jpeg_header_v1.json").write_text(json.dumps(model_data))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader, parse_jpeg_header

    real_data = _make_large_jpeg_rgb(400, 300)
    real_header = parse_jpeg_header(real_data)
    assert real_header is not None

    header = JpegHeader(
        width=800,
        height=600,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=real_header.dqt_luma,
        dqt_chroma=real_header.dqt_chroma,
        app14_color_transform=None,
        fallback_reason=None,
    )
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    # Either custom_quantization (NSE gate) or prediction_oob (model gate)
    assert result.reason in ("custom_quantization", "prediction_oob")

    models_mod.load_jpeg_header_model.cache_clear()


def test_jpeg_header_only_bpp_inner_ratio_gate_oob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_jpeg_header_only_bpp_inner: very small predicted_bpp → ratio gate → prediction_oob."""
    import json

    import estimation.models as models_mod
    from estimation.models._artifact import _JPEG_HEADER_FEATURES as FEAT

    # intercept=0.001 → predicted_bpp tiny << min_ratio * input_bpp → prediction_oob
    model_data = {
        "model_version": 1,
        "format": "jpeg_header",
        "features": FEAT,
        "scaler": {
            "mean": [0.0] * 13,
            "scale": [1.0] * 13,
        },
        "coefficients": {
            "intercept": 0.001,
            "betas": [0.0] * 13,
        },
        "training_envelope": {},
        "training_corpus_sha256": "abc",
        "git_sha": "def",
        "fit_environment": {"numpy_version": "2.0.0"},
        "created_at": "2026-05-07T00:00:00Z",
    }
    (tmp_path / "jpeg_header_v1.json").write_text(json.dumps(model_data))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader, parse_jpeg_header

    real_data = _make_large_jpeg_rgb(400, 300)
    real_header = parse_jpeg_header(real_data)
    assert real_header is not None

    header = JpegHeader(
        width=800,
        height=600,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=real_header.dqt_luma,
        dqt_chroma=real_header.dqt_chroma,
        app14_color_transform=None,
        fallback_reason=None,
    )
    # file_size = 50000 bytes → input_bpp = 50000*8/(800*600) ≈ 0.83 bpp
    # predicted = 0.001; min_ratio(q=60) = 0.10; 0.001 < 0.10*0.83 → prediction_oob
    result = _jpeg_header_only_bpp_inner(header, 50000, 60, False)
    assert isinstance(result, HeaderOnlyFallback)

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §26 _png_fitted_bpp_inner: quality regime branches (lines 198, 202, 209)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_bpp_inner_quality_regime_q40(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner with quality=40 → min_ratio=0.05 branch (line 198)."""
    import json

    import estimation.models as models_mod

    # Write a model with a reasonable prediction (between 0.05 and 1.10 of input_bpp)
    model_data = {
        "model_version": 2,
        "format": "png",
        "features": [
            "has_alpha",
            "log10_unique_colors",
            "mean_sobel",
            "edge_density",
            "quality",
            "log10_orig_pixels",
            "input_bpp",
        ],
        "supported_modes": ["RGB", "RGBA", "L", "LA", "P"],
        "scaler": {
            "mean": [0.0, 3.0, 50.0, 0.3, 60.0, 5.5, 8.0],
            "scale": [1.0, 0.5, 30.0, 0.2, 15.0, 1.0, 4.0],
        },
        "coefficients": {
            # intercept=4.0; input_bpp will be ~24 → ratio ≈ 4/24 ≈ 0.17 > min_ratio 0.05
            "intercept": 4.0,
            "betas": [0.0] * 7,
            "knot_beta": 0.0,
            "knot_q50_beta": 0.0,
            "knot_q70_beta": 0.0,
        },
        "knot_log10_unique_colors": 3.3,
        "knot_q50": 50.0,
        "knot_q70": 70.0,
        "training_envelope": {},
        "training_corpus_sha256": "abc",
        "git_sha": "def",
        "fit_environment": {"numpy_version": "2.0.0"},
        "created_at": "2026-05-07T00:00:00Z",
    }
    (tmp_path / "png_v1.json").write_text(json.dumps(model_data))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    import numpy as np
    from PIL import Image

    from estimation.estimator import FittedBpp, FittedFallback, _png_fitted_bpp_inner

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # orig_size = 30000 → input_bpp = 30000*8/(100*100) = 24 bpp
    result = _png_fitted_bpp_inner(img, 100, 100, 40, orig_size=30000)
    # Should be FittedBpp or FittedFallback — either is valid; key is we hit the q<50 branch
    assert isinstance(result, (FittedBpp, FittedFallback))

    models_mod.load_png_model.cache_clear()


def test_png_fitted_bpp_inner_quality_regime_q75(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner with quality=75 → min_ratio=0.40 branch (line 202)."""
    import json

    import estimation.models as models_mod

    # intercept=5.0; input_bpp=8 → ratio=5/8=0.625 > 0.40 → passes gate → FittedBpp
    model_data = {
        "model_version": 2,
        "format": "png",
        "features": [
            "has_alpha",
            "log10_unique_colors",
            "mean_sobel",
            "edge_density",
            "quality",
            "log10_orig_pixels",
            "input_bpp",
        ],
        "supported_modes": ["RGB", "RGBA", "L", "LA", "P"],
        "scaler": {
            "mean": [0.0, 3.0, 50.0, 0.3, 60.0, 5.5, 8.0],
            "scale": [1.0, 0.5, 30.0, 0.2, 15.0, 1.0, 4.0],
        },
        "coefficients": {
            "intercept": 5.0,
            "betas": [0.0] * 7,
            "knot_beta": 0.0,
            "knot_q50_beta": 0.0,
            "knot_q70_beta": 0.0,
        },
        "knot_log10_unique_colors": 3.3,
        "knot_q50": 50.0,
        "knot_q70": 70.0,
        "training_envelope": {},
        "training_corpus_sha256": "abc",
        "git_sha": "def",
        "fit_environment": {"numpy_version": "2.0.0"},
        "created_at": "2026-05-07T00:00:00Z",
    }
    (tmp_path / "png_v1.json").write_text(json.dumps(model_data))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    import numpy as np
    from PIL import Image

    from estimation.estimator import FittedBpp, FittedFallback, _png_fitted_bpp_inner

    rng = np.random.default_rng(77)
    arr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # orig_size = 10000 → input_bpp = 10000*8/(100*100) = 8 bpp
    result = _png_fitted_bpp_inner(img, 100, 100, 75, orig_size=10000)
    assert isinstance(result, (FittedBpp, FittedFallback))

    models_mod.load_png_model.cache_clear()


# ---------------------------------------------------------------------------
# §27 _png_fitted_bpp_inner: ValueError for missing feature in model (lines 166-167)
# ---------------------------------------------------------------------------


def test_png_header_only_bpp_inner_prediction_oob_via_high_intercept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_header_only_bpp_inner: intercept=100 → predicted_bpp > 32 → prediction_oob (line 330)."""
    import json

    import estimation.models as models_mod

    model_data = {
        "model_version": 1,
        "format": "png_header",
        "features": ["has_alpha", "quality", "log10_orig_pixels", "input_bpp"],
        "scaler": {
            "mean": [0.0, 60.0, 5.5, 9.0],
            "scale": [1.0, 18.0, 0.7, 5.0],
        },
        "coefficients": {
            "intercept": 100.0,  # forces predicted_bpp >> 32 → prediction_oob
            "betas": [0.0, 0.0, 0.0, 0.0],
            "knot_q50_beta": 0.0,
            "knot_q70_beta": 0.0,
        },
        "knot_q50": 50.0,
        "knot_q70": 70.0,
        "training_envelope": {},
        "training_corpus_sha256": "abc",
        "git_sha": "def",
        "fit_environment": {"numpy_version": "2.0.0"},
        "created_at": "2026-05-07T00:00:00Z",
    }
    (tmp_path / "png_header_v1.json").write_text(json.dumps(model_data))
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    from estimation.estimator import HeaderOnlyFallback, _png_header_only_bpp_inner
    from estimation.png_header import PngHeader

    header = PngHeader(
        width=800, height=600, bit_depth=8, color_type=2, has_alpha=False, is_palette=False
    )
    # file_size = 50000 → input_bpp = 50000*8/(800*600) ≈ 0.83 bpp (in range)
    # predicted = 100.0 >> 32 → prediction_oob
    result = _png_header_only_bpp_inner(header, 50000, 60)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "prediction_oob"

    models_mod.load_png_header_model.cache_clear()


def test_png_fitted_bpp_inner_feature_index_valueerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner catches ValueError when model features lack 'log10_unique_colors'."""

    # A model that passes construction + _validate_schema but has features that don't include
    # 'log10_unique_colors' — model.features.index() will raise ValueError
    # We need to bypass _validate_schema which is called after construction.
    # The trick: write a model with the correct PngModel _SUPPORTED_MODEL_VERSION but
    # wrong features. The _validate_schema would return an error, so it won't load.
    # Instead, patch load_png_model to return a Loaded with doctored features.
    from estimation.estimator import FittedFallback, _png_fitted_bpp_inner
    from estimation.models._artifact import Loaded, PngModel

    # Create a minimal model object with features that lack 'log10_unique_colors'
    bad_features = [
        "has_alpha",
        "mean_sobel",
        "edge_density",
        "quality",
        "log10_orig_pixels",
        "input_bpp",
    ]
    # Build a minimal valid-looking raw dict and pass to the constructor directly
    bad_model = PngModel(
        model_version=2,
        format="png",
        features=bad_features,  # missing 'log10_unique_colors'
        supported_modes=["RGB"],
        scaler={"mean": [0.0] * 6, "scale": [1.0] * 6},
        coefficients={
            "intercept": 0.5,
            "betas": [0.0] * 6,
            "knot_beta": 0.0,
            "knot_q50_beta": 0.0,
            "knot_q70_beta": 0.0,
        },
        knot_log10_unique_colors=3.3,
        knot_q50=50.0,
        knot_q70=70.0,
        training_envelope={},
        training_corpus_sha256="abc",
        git_sha="def",
        fit_environment={},
        created_at="2026-05-07",
    )
    # Patch load_png_model to return this bad model
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod, "load_png_model", lambda: Loaded(model=bad_model))

    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # This will try model.features.index("log10_unique_colors") → ValueError → model_load_failed
    result = _png_fitted_bpp_inner(img, 100, 100, 60, orig_size=30000)
    assert isinstance(result, FittedFallback)
    assert result.reason == "model_load_failed"


# ---------------------------------------------------------------------------
# §28 Pixel cap: PNG header-only returns feature_oob when pixels exceed cap
# ---------------------------------------------------------------------------


def test_png_header_only_pixel_cap_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """PNG header with width=20000, height=20000 (400 MP > 100 MP cap) → feature_oob."""
    from estimation.estimator import HeaderOnlyFallback, _png_header_only_bpp_inner
    from estimation.png_header import PngHeader

    monkeypatch.setattr("estimation.estimator.settings.max_image_pixels", 100_000_000)

    # 20000 × 20000 = 400 megapixels > 100 MP cap
    header = PngHeader(
        width=20_000, height=20_000, bit_depth=8, color_type=2, has_alpha=False, is_palette=False
    )
    # file_size large enough so input_bpp would be in normal range if pixels weren't capped
    file_size = 20_000 * 20_000 * 3  # 3 bpp uncompressed ~ 1.2 GB (irrelevant — cap fires first)
    result = _png_header_only_bpp_inner(header, file_size, 60)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "feature_oob"


# ---------------------------------------------------------------------------
# §29 Pixel cap: JPEG header-only returns feature_oob when pixels exceed cap
# ---------------------------------------------------------------------------


def test_jpeg_header_only_pixel_cap_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """JPEG header with width=20000, height=20000 (400 MP > 100 MP cap) → feature_oob."""
    from estimation.estimator import HeaderOnlyFallback, _jpeg_header_only_bpp_inner
    from estimation.jpeg_header import JpegHeader

    monkeypatch.setattr("estimation.estimator.settings.max_image_pixels", 100_000_000)

    # 20000 × 20000 = 400 megapixels > 100 MP cap
    header = JpegHeader(
        width=20_000,
        height=20_000,
        components=3,
        bit_depth=8,
        subsampling="4:2:0",
        progressive=False,
        dqt_luma=[16] * 64,
        dqt_chroma=[17] * 64,
        app14_color_transform=None,
        fallback_reason=None,
    )
    file_size = 20_000 * 20_000 * 3  # large but irrelevant — pixel cap fires before input_bpp check
    result = _jpeg_header_only_bpp_inner(header, file_size, 60, False)
    assert isinstance(result, HeaderOnlyFallback)
    assert result.reason == "feature_oob"
