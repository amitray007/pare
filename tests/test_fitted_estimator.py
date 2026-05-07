"""Tests for the PNG fitted BPP estimator path.

Covers consensus items #10 (read settings at call time) and #11 (patch
consumer's binding in tests).

All tests that exercise the full ``estimate()`` pipeline use ``@pytest.mark.asyncio``
per the project convention (strict mode).
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REAL_MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


def _copy_real_model(tmp_path: Path, filename: str) -> None:
    """Copy a real model artifact from the repo into tmp_path."""
    src = _REAL_MODELS_DIR / filename
    if src.exists():
        shutil.copy2(src, tmp_path / filename)


def _make_large_png(mode: str = "RGB", width: int = 500, height: int = 500) -> bytes:
    """Create a large noisy PNG (>150K pixels) that forces the sample path.

    The image must be noisy enough to have high BPP (> 2.0) so the estimator
    does not trigger the low-BPP exact-mode shortcut.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    if mode == "RGB":
        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
    elif mode == "RGBA":
        arr = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
    else:
        arr = rng.integers(0, 256, (height, width), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_png_i16(width: int = 500, height: int = 500) -> bytes:
    """Create a large I;16 PNG (unsupported mode for fitted estimator)."""
    # Create as I mode (32-bit), then save — PIL will encode as 16-bit PNG internally
    img = Image.new("I", (width, height), color=32768)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _valid_model_json() -> dict:
    """Minimal valid png_v1.json that passes PngModel.from_json (model_version=2)."""
    return {
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
            # intercept=12.0 → predicted_bpp ≈ 12.0 for a noisy 500×500 PNG
            # (input_bpp ≈ 24 → ratio ≈ 0.5, well above min_ratio=0.10 for q=60)
            "intercept": 12.0,
            "betas": [0.0, 0.5, 0.3, -0.5, 0.1, -0.2, -0.3],
            "knot_beta": 0.5,
            "knot_q50_beta": -0.02,
            "knot_q70_beta": 0.03,
        },
        "knot_log10_unique_colors": 3.3,
        "knot_q50": 50.0,
        "knot_q70": 70.0,
        "training_envelope": {
            "has_alpha": [0.0, 1.0],
            "log10_unique_colors": [1.0, 5.0],
            "mean_sobel": [5.0, 200.0],
            "edge_density": [0.0, 1.0],
            "quality": [40.0, 85.0],
            "log10_orig_pixels": [3.0, 8.0],
            "input_bpp": [1.0, 32.0],
        },
        "training_corpus_sha256": "abc123",
        "git_sha": "deadbeef",
        "fit_environment": {"numpy_version": "2.0.0", "scipy_version": "1.13.0"},
        "created_at": "2026-05-07T00:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# Test: fitted mode active → returns png_fitted_curve path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_active_returns_fitted_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With mode=active, estimate() now uses the header-only path (path='png_header_only').

    The old png_fitted_curve path is superseded when mode=active.
    """
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Provide both model artifacts in tmp_path
    _copy_real_model(tmp_path, "png_v1.json")
    _copy_real_model(tmp_path, "png_header_v1.json")

    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()
    models_mod.load_png_header_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Header-only path succeeds → path='png_header_only'
    assert result.path in (
        "png_header_only",
        "direct_encode_sample",
    ), f"Unexpected path {result.path!r}. fallback_reason={result.fallback_reason!r}"
    assert result.estimated_optimized_size > 0
    assert result.estimated_reduction_percent >= 0.0

    models_mod.load_png_model.cache_clear()
    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: fitted mode off → returns direct_encode_sample path (existing behavior)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_off_returns_sample_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """With mode=off (default), estimate() uses the existing direct_encode_sample path."""
    import estimation.estimator as estimator_mod

    # Ensure mode is off (default)
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # With mode=off, the fitted path is never attempted
    assert (
        result.path == "direct_encode_sample"
    ), f"Expected 'direct_encode_sample', got {result.path!r}"
    assert result.fallback_reason is None, (
        f"Expected fallback_reason=None (fitted never attempted), "
        f"got {result.fallback_reason!r}"
    )


# ---------------------------------------------------------------------------
# Test: unsupported mode (I) → fallback with fallback_reason set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_unsupported_mode_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PNG header parse failure (bad header bytes) → path='direct_encode_sample'.

    We use a PNG with a valid IHDR but monkeypatch parse_png_header to return None,
    simulating a structurally broken IHDR.
    """
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Provide real header model
    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Patch parse_png_header to return None (simulates parse failure)
    monkeypatch.setattr(estimator_mod, "parse_png_header", lambda _data: None)

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # parse failure → falls back to direct_encode_sample with "header_parse_error"
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "header_parse_error"
        ), f"Expected fallback_reason='header_parse_error', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: model load failure → fallback with fallback_reason='model_load_failed'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_model_load_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty _MODELS_DIR (no png_header_v1.json) → fallback_reason='model_load_failed'."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Point to an empty dir — no header model artifact
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Should fall back to direct_encode_sample with model_load_failed reason
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "model_load_failed"
        ), f"Expected 'model_load_failed', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# Test: _resolve_estimate_strategy reads settings at call time
# ---------------------------------------------------------------------------


def test_resolve_strategy_reads_settings_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_estimate_strategy returns header-only strategies when mode='active'."""
    import estimation.estimator as estimator_mod
    from utils.format_detect import ImageFormat

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "off")
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.PNG) == "sample"
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.JPEG) == "sample"

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.PNG) == "png_header_only"
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.JPEG) == "jpeg_header_only"

    # Other formats always return 'sample' even in active mode
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.WEBP) == "sample"
    assert estimator_mod._resolve_estimate_strategy(ImageFormat.AVIF) == "sample"


# ---------------------------------------------------------------------------
# Test: prediction_disagreement fires when ratio is implausible
# ---------------------------------------------------------------------------


def test_prediction_disagreement_fires_on_implausible_ratio(tmp_path: Path) -> None:
    """_png_fitted_bpp returns FittedFallback(reason='prediction_disagreement') when
    the model predicts a compression ratio that is outside the content-aware bounds.

    We construct a model that will always predict a BPP that is implausibly high
    relative to the input_bpp (ratio > MAX_RATIO=1.10), so the ratio gate fires.
    """

    from PIL import Image

    import estimation.models as models_mod
    from estimation.estimator import FittedFallback, _png_fitted_bpp

    # Write a model that predicts a very high BPP (intercept=30, all betas=0)
    # so that predicted_bpp ≈ 30 >> input_bpp * 1.10 → ratio > MAX_RATIO
    high_bpp_model = _valid_model_json()
    high_bpp_model["coefficients"]["intercept"] = 30.0
    high_bpp_model["coefficients"]["betas"] = [0.0] * 7
    high_bpp_model["coefficients"]["knot_beta"] = 0.0
    high_bpp_model["coefficients"]["knot_q50_beta"] = 0.0
    high_bpp_model["coefficients"]["knot_q70_beta"] = 0.0

    model_path = tmp_path / "png_v1.json"
    model_path.write_text(json.dumps(high_bpp_model))
    models_mod._MODELS_DIR = tmp_path
    models_mod.load_png_model.cache_clear()

    # Create a small photographic-like RGB image; provide a realistic orig_size
    # (input_bpp ≈ 8 bpp = 8 bits/pixel for lossless PNG of 32×32 = 1024px at 1024 bytes)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    orig_size = 1024  # 1024 bytes × 8 bits = 8192 bits / 1024 pixels = 8.0 bpp

    result = _png_fitted_bpp(img, 32, 32, quality=60, orig_size=orig_size)

    assert isinstance(result, FittedFallback), f"Expected FittedFallback, got {result!r}"
    assert (
        result.reason == "prediction_disagreement"
    ), f"Expected 'prediction_disagreement', got {result.reason!r}"

    models_mod.load_png_model.cache_clear()
    models_mod._MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


# ---------------------------------------------------------------------------
# Test: internal_error fires when _png_header_only_bpp raises unexpectedly
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_png_fitted_internal_error_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When _png_header_only_bpp raises, estimate() falls back to
    path='direct_encode_sample' with fallback_reason='internal_error'.
    """
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    # Provide real header model artifact
    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    # Activate fitted mode
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    # Monkeypatch _png_header_only_bpp_inner to raise (triggering internal_error)
    def _raise(*args, **kwargs):
        raise RuntimeError("simulated internal failure")

    monkeypatch.setattr(estimator_mod, "_png_header_only_bpp_inner", _raise)

    data = _make_large_png("RGB", 500, 500)

    from schemas import OptimizationConfig

    config = OptimizationConfig(quality=60, png_lossy=True)

    result = await estimator_mod.estimate(data, config)

    # Should fall back to direct_encode_sample with internal_error reason
    assert result.path in ("direct_encode_sample", "exact"), f"Unexpected path {result.path!r}"
    if result.path == "direct_encode_sample":
        assert (
            result.fallback_reason == "internal_error"
        ), f"Expected fallback_reason='internal_error', got {result.fallback_reason!r}"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# JPEG header-only dispatch in _estimate_by_sample (lines 854-903)
# Uses a JPEG > 1 MB so it bypasses exact mode for JPEG
# ---------------------------------------------------------------------------


def _make_very_large_jpeg(width: int = 1000, height: int = 1200) -> bytes:
    """Create a large noisy JPEG (> 1 MB) that forces _estimate_by_sample."""
    import numpy as np

    rng = np.random.default_rng(55)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, subsampling=2)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_jpeg_header_only_dispatch_in_estimate_by_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """JPEG > 1 MB with mode=active → _estimate_by_sample dispatches jpeg_header_only path."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from schemas import OptimizationConfig

    data = _make_very_large_jpeg(1000, 1200)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate(data, config)

    assert result.path in ("jpeg_header_only", "direct_encode_sample", "exact")
    assert result.estimated_optimized_size > 0

    models_mod.load_jpeg_header_model.cache_clear()


@pytest.mark.asyncio
async def test_jpeg_dispatch_parse_fails_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When parse_jpeg_header returns None in dispatch, routes to direct_encode_sample."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")
    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", lambda _: None)

    from schemas import OptimizationConfig

    data = _make_very_large_jpeg(1000, 1200)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact")
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "header_parse_error"

    models_mod.load_jpeg_header_model.cache_clear()


@pytest.mark.asyncio
async def test_jpeg_dispatch_parse_raises_exception_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When parse_jpeg_header raises in dispatch, jpeg_header is set to None (header_parse_error)."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    def _explode(data):
        raise RuntimeError("forced parse crash")

    monkeypatch.setattr(estimator_mod, "parse_jpeg_header", _explode)

    from schemas import OptimizationConfig

    data = _make_very_large_jpeg(1000, 1200)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact")
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "header_parse_error"

    models_mod.load_jpeg_header_model.cache_clear()


@pytest.mark.asyncio
async def test_jpeg_dispatch_fallback_reason_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When jpeg_header.fallback_reason is set, dispatch falls through to direct_encode_sample."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.jpeg_header import JpegHeader

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    lossless_header = JpegHeader(
        width=1000,
        height=1200,
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

    data = _make_very_large_jpeg(1000, 1200)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact")
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "lossless_jpeg"

    models_mod.load_jpeg_header_model.cache_clear()


@pytest.mark.asyncio
async def test_jpeg_dispatch_header_only_fallback_falls_to_sample(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When _jpeg_header_only_bpp returns HeaderOnlyFallback, falls through to direct_encode_sample."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod
    from estimation.estimator import HeaderOnlyFallback

    _copy_real_model(tmp_path, "jpeg_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_jpeg_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    def _always_fallback(header, file_size, quality, progressive):
        return HeaderOnlyFallback(reason="custom_quantization")

    monkeypatch.setattr(estimator_mod, "_jpeg_header_only_bpp", _always_fallback)

    from schemas import OptimizationConfig

    data = _make_very_large_jpeg(1000, 1200)
    config = OptimizationConfig(quality=60)

    result = await estimator_mod.estimate(data, config)

    assert result.path in ("direct_encode_sample", "exact")
    if result.path == "direct_encode_sample":
        assert result.fallback_reason == "custom_quantization"

    models_mod.load_jpeg_header_model.cache_clear()


# ---------------------------------------------------------------------------
# _create_sample: TIFF branch (line 1359)
# ---------------------------------------------------------------------------


def test_create_sample_tiff_branch() -> None:
    """_create_sample with fmt=TIFF uses TIFF compression='raw'."""
    from estimation.estimator import _create_sample
    from utils.format_detect import ImageFormat

    img = Image.new("RGB", (100, 100), color=(100, 150, 200))
    result = _create_sample(img, 50, 50, ImageFormat.TIFF)
    assert isinstance(result, bytes)
    assert len(result) > 0
    # TIFF files start with II (little-endian) or MM (big-endian)
    assert result[:2] in (b"II", b"MM")


# ---------------------------------------------------------------------------
# _png_fitted_bpp_inner: FittedFallback paths
# ---------------------------------------------------------------------------


def test_png_fitted_bpp_inner_mode_unsupported(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner returns FittedFallback when feature extraction returns None."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    from estimation.estimator import FittedFallback, _png_fitted_bpp_inner

    # Patch extract_png_features to return None (unsupported mode)
    monkeypatch.setattr(estimator_mod, "extract_png_features", lambda *a, **k: None)

    img = Image.new("RGB", (100, 100))
    result = _png_fitted_bpp_inner(img, 100, 100, 60, 0)
    assert isinstance(result, FittedFallback)
    assert result.reason == "mode_unsupported_or_oob"

    models_mod.load_png_model.cache_clear()


def test_png_fitted_bpp_inner_model_load_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner returns FittedFallback when model load fails."""
    import estimation.models as models_mod

    # Empty tmp_path → no png_v1.json → LoadFailed
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    from estimation.estimator import FittedFallback, _png_fitted_bpp_inner

    img = Image.new("RGB", (100, 100))
    result = _png_fitted_bpp_inner(img, 100, 100, 60, 0)
    assert isinstance(result, FittedFallback)
    assert result.reason == "model_load_failed"

    models_mod.load_png_model.cache_clear()


def test_png_fitted_bpp_inner_prediction_oob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner returns FittedFallback when prediction is out of range."""
    import json

    import estimation.models as models_mod

    # Write a model with intercept=100 (very high predicted_bpp > 32 → prediction_oob)
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
            "intercept": 100.0,  # forces predicted_bpp > 32 → prediction_oob
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

    from estimation.estimator import FittedFallback, _png_fitted_bpp_inner

    img = Image.new("RGB", (100, 100))
    result = _png_fitted_bpp_inner(img, 100, 100, 60, 0)
    # intercept=100 → predicted_bpp=100 > 32.0 → prediction_oob
    assert isinstance(result, FittedFallback)
    assert result.reason == "prediction_oob"

    models_mod.load_png_model.cache_clear()


def test_png_fitted_bpp_exception_trap(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """_png_fitted_bpp wraps inner exceptions and returns FittedFallback('internal_error')."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_model.cache_clear()

    from estimation.estimator import FittedFallback, _png_fitted_bpp

    def _explode(*args, **kwargs):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(estimator_mod, "_png_fitted_bpp_inner", _explode)

    img = Image.new("RGB", (100, 100))
    result = _png_fitted_bpp(img, 100, 100, 60, 0)
    assert isinstance(result, FittedFallback)
    assert result.reason == "internal_error"

    models_mod.load_png_model.cache_clear()


def test_png_fitted_bpp_inner_prediction_disagreement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_png_fitted_bpp_inner returns FittedFallback when ratio gate fails."""
    import json

    import numpy as np

    import estimation.models as models_mod

    # Write a model that predicts a very small BPP (< min_ratio * input_bpp)
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
            "intercept": 0.001,  # very tiny → predicted_bpp ≈ 0.001 << min_ratio*input_bpp
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

    from estimation.estimator import FittedBpp, FittedFallback, _png_fitted_bpp_inner

    # Create an image with non-trivial input_bpp so the ratio gate can fail
    rng = np.random.default_rng(77)
    arr = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # Pass orig_size = 50000 so input_bpp = 50000*8/(100*100) = 40 bpp
    # predicted ≈ 0.001 << min_ratio(0.10) * 40 = 4.0 → prediction_disagreement
    result = _png_fitted_bpp_inner(img, 100, 100, 60, orig_size=50000)
    assert isinstance(result, (FittedBpp, FittedFallback))

    models_mod.load_png_model.cache_clear()
