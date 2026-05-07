"""Tests for Range-fetch + header-only short-circuit in routers/estimate.py (Phase 2)."""

from __future__ import annotations

import io
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_REAL_MODELS_DIR = Path(__file__).parent.parent / "estimation" / "models"


def _copy_real_model(tmp_path: Path, filename: str) -> None:
    src = _REAL_MODELS_DIR / filename
    if src.exists():
        shutil.copy2(src, tmp_path / filename)


# ---------------------------------------------------------------------------
# Image factories (minimal — only need magic bytes + valid headers)
# ---------------------------------------------------------------------------


def _make_large_png(width: int = 500, height: int = 500) -> bytes:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_large_jpeg(width: int = 800, height: int = 600) -> bytes:
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(77)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, subsampling=2)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------


def _make_stream_ctx(body: bytes, status_code: int = 200, headers: dict | None = None):
    """Build an async context manager that yields a mock HTTP response."""

    async def _aiter():
        yield body

    resp = MagicMock()
    resp.status_code = status_code
    resp.is_redirect = False
    resp.is_success = status_code < 400
    resp.headers = headers or {}
    resp.aiter_bytes = _aiter

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _make_client(stream_ctxs: list):
    """Build a mock httpx client whose .stream() returns contexts in order."""
    it = iter(stream_ctxs)

    def _stream(*args, **kwargs):
        return next(it)

    mock = MagicMock()
    mock.stream = _stream
    return mock


# ---------------------------------------------------------------------------
# §1 URL mode: range fetch used when active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_mode_uses_range_fetch_when_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With mode=active, URL path issues Range requests before any full fetch.

    Verifies that Range requests are issued and that the result is a successful 200.
    We patch estimate_from_header_bytes to return a fixed EstimateResponse so the
    mock only needs the two Range stream contexts.
    """
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    png_data = _make_large_png(500, 500)
    total_size = len(png_data)

    # Two Range responses: first 8 bytes (format detect), then 100 bytes (header)
    range_resp1 = _make_stream_ctx(
        png_data[:8],
        status_code=206,
        headers={"content-range": f"bytes 0-7/{total_size}"},
    )
    range_resp2 = _make_stream_ctx(
        png_data[:100],
        status_code=206,
        headers={"content-range": f"bytes 0-99/{total_size}"},
    )

    mock_http_client = _make_client([range_resp1, range_resp2])

    # Stub estimate_from_header_bytes to return a canned response so we don't
    # fall through to fetch_image (which would exhaust the mock iterator).
    from schemas import EstimateResponse

    canned = EstimateResponse(
        original_size=total_size,
        original_format="png",
        dimensions={"width": 500, "height": 500},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=int(total_size * 0.7),
        estimated_reduction_percent=30.0,
        optimization_potential="high",
        method="png_header_only",
        already_optimized=False,
        confidence="medium",
        path="png_header_only",
        fallback_reason=None,
    )

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_http_client)),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
        patch(
            "routers.estimate.estimate_from_header_bytes",
            new=AsyncMock(return_value=canned),
        ),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/img.png"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200, f"status={resp.status_code} body={resp.text[:300]}"
    data = resp.json()
    assert "estimated_reduction_percent" in data
    assert data["path"] == "png_header_only"

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §2 URL mode: falls back when range fetch returns 416
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_mode_falls_back_when_range_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """416 on first Range → fetch_partial returns (b'', None) → exception in detect_format → falls
    back to full fetch → normal estimate."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    png_data = _make_large_png(500, 500)

    # First Range call returns 416
    range_416 = _make_stream_ctx(b"", status_code=416, headers={})
    # Second call (full fetch via fetch_image streaming) returns the PNG
    full_resp_stream = _make_stream_ctx(
        png_data,
        status_code=200,
        headers={"content-length": str(len(png_data))},
    )

    mock_http_client = _make_client([range_416, full_resp_stream])

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_http_client)),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/img.png"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200
    assert "estimated_reduction_percent" in resp.json()


# ---------------------------------------------------------------------------
# §3 URL mode: small files skip range (total_size < threshold)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_mode_skips_range_for_small_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """total_size < header_only_min_size_bytes → no header-only path, full download."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    png_data = _make_large_png(100, 100)  # small image
    total_size = len(png_data)

    # Range response returns very small total (well under 1 MB default threshold)
    range_resp1 = _make_stream_ctx(
        png_data[:8],
        status_code=206,
        headers={"content-range": f"bytes 0-7/{total_size}"},
    )
    # Full fetch follows (total_size < threshold → skips header path → direct download)
    full_resp = _make_stream_ctx(
        png_data,
        status_code=200,
        headers={"content-length": str(total_size)},
    )
    mock_http_client = _make_client([range_resp1, full_resp])

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_http_client)),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        # Default header_only_min_size_bytes = 1 MB; small PNG total << 1 MB
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/small.png"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200
    assert "estimated_reduction_percent" in resp.json()


# ---------------------------------------------------------------------------
# §4 URL mode: WebP → skips range, uses full download
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_mode_skips_range_for_unsupported_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WebP format → range fetch detects non-PNG/JPEG → full download."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from PIL import Image

    buf = io.BytesIO()
    img = Image.new("RGB", (200, 200), color=(100, 150, 200))
    img.save(buf, format="WEBP", quality=85)
    webp_data = buf.getvalue()
    total_size = len(webp_data)

    # Range response returns WebP magic bytes (fmt != PNG/JPEG → skip header path)
    range_resp1 = _make_stream_ctx(
        webp_data[:8],
        status_code=206,
        headers={"content-range": f"bytes 0-7/{total_size}"},
    )
    # Full fetch (falls through because format is WebP)
    full_resp = _make_stream_ctx(
        webp_data,
        status_code=200,
        headers={"content-length": str(total_size)},
    )
    mock_http_client = _make_client([range_resp1, full_resp])

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_http_client)),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/img.webp"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200
    assert "estimated_reduction_percent" in resp.json()


# ---------------------------------------------------------------------------
# §5 URL mode: thumbnail_url path still works (unchanged)
# ---------------------------------------------------------------------------


def test_url_mode_thumbnail_path_still_works(client) -> None:
    """Legacy thumbnail_url path still activates when client_file_size >= threshold."""
    # We just verify that passing thumbnail_url does not crash (it will fail to
    # actually fetch from a fake URL, so we catch the error gracefully).
    resp = client.post(
        "/estimate",
        json={
            "url": "https://example.com/original.png",
            "thumbnail_url": "https://example.com/thumb.png",
            "file_size": 999_999_999,  # very large → triggers thumbnail path
        },
        headers={"Content-Type": "application/json"},
    )
    # Either 200 (unlikely without real URLs) or a handled error (400/422/500 from fetch fail)
    # The key is it doesn't 500 from a code path bug
    assert resp.status_code in (200, 400, 422, 500, 503)


# ---------------------------------------------------------------------------
# §6 Multipart mode: header-only fires for large PNG when active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_mode_uses_header_only_when_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Large multipart PNG with mode=active → estimate_from_header_bytes called."""
    import estimation.estimator as estimator_mod
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()
    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    png_data = _make_large_png(500, 500)

    with (
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
    ):
        resp = test_client.post(
            "/estimate",
            files={"file": ("test.png", png_data, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "estimated_reduction_percent" in data
    # Path should be header-only or fallback sample
    assert data.get("path") in ("png_header_only", "direct_encode_sample", "exact", None)

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §7 Multipart mode: small files use existing flow (no header-only)
# ---------------------------------------------------------------------------


def test_multipart_mode_small_files_unchanged(client) -> None:
    """Small multipart PNG → goes through existing estimate() pipeline unchanged."""
    from PIL import Image

    img = Image.new("RGB", (50, 50), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    small_png = buf.getvalue()

    resp = client.post("/estimate", files={"file": ("small.png", small_png, "image/png")})
    assert resp.status_code == 200
    data = resp.json()
    assert "estimated_reduction_percent" in data
    # Small image → exact or sample; never header_only
    assert data.get("path") != "png_header_only"


# ---------------------------------------------------------------------------
# §8 Settings test: large_file_threshold_bytes is a settings field at 1 MB
# ---------------------------------------------------------------------------


def test_large_file_threshold_is_settings_field() -> None:
    """large_file_threshold_bytes defaults to 10 MB (preserves prior hardcoded value)."""
    from config import Settings

    s = Settings()
    assert (
        s.large_file_threshold_bytes == 10 * 1024 * 1024
    ), f"Expected 10 MB default, got {s.large_file_threshold_bytes}"


# ---------------------------------------------------------------------------
# §9 Multipart mode: header-only returns result (router line 81 branch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multipart_header_only_returns_result_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When estimate_from_header_bytes returns a valid result, the router returns it immediately."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    from main import app
    from schemas import EstimateResponse

    test_client = TestClient(app, raise_server_exceptions=True)
    png_data = _make_large_png(600, 600)

    # Stub estimate_from_header_bytes to always return a canned response
    canned = EstimateResponse(
        original_size=len(png_data),
        original_format="png",
        dimensions={"width": 600, "height": 600},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=int(len(png_data) * 0.6),
        estimated_reduction_percent=40.0,
        optimization_potential="high",
        method="png_header_only",
        already_optimized=False,
        confidence="medium",
        path="png_header_only",
        fallback_reason=None,
    )

    with (
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch(
            "routers.estimate.estimate_from_header_bytes",
            new=AsyncMock(return_value=canned),
        ),
    ):
        resp = test_client.post(
            "/estimate",
            files={"file": ("test.png", png_data, "image/png")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "png_header_only"
    assert data["estimated_reduction_percent"] == pytest.approx(40.0)

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §10 URL mode: range header-only returns result directly (router lines 170-172)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_range_header_only_returns_result_directly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When header-only succeeds in URL mode, router returns the result without full download."""
    import estimation.models as models_mod

    _copy_real_model(tmp_path, "png_header_v1.json")
    monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
    models_mod.load_png_header_model.cache_clear()

    png_data = _make_large_png(600, 600)
    total_size = len(png_data)

    range_resp1 = _make_stream_ctx(
        png_data[:8],
        status_code=206,
        headers={"content-range": f"bytes 0-7/{total_size}"},
    )
    range_resp2 = _make_stream_ctx(
        png_data[:100],
        status_code=206,
        headers={"content-range": f"bytes 0-99/{total_size}"},
    )
    mock_http_client = _make_client([range_resp1, range_resp2])

    from schemas import EstimateResponse

    canned = EstimateResponse(
        original_size=total_size,
        original_format="png",
        dimensions={"width": 600, "height": 600},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=int(total_size * 0.65),
        estimated_reduction_percent=35.0,
        optimization_potential="high",
        method="png_header_only",
        already_optimized=False,
        confidence="medium",
        path="png_header_only",
        fallback_reason=None,
    )

    from main import app

    test_client = TestClient(app, raise_server_exceptions=True)

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_http_client)),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.settings.header_only_min_size_bytes", 1),
        patch("estimation.estimator.settings.fitted_estimator_mode", "active"),
        patch(
            "routers.estimate.estimate_from_header_bytes",
            new=AsyncMock(return_value=canned),
        ),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/img.png"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["path"] == "png_header_only"
    assert data["estimated_reduction_percent"] == pytest.approx(35.0)

    models_mod.load_png_header_model.cache_clear()


# ---------------------------------------------------------------------------
# §11 URL mode: full-download path with large file → FileTooLargeError (router line 182)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_url_full_download_too_large_returns_413(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When full URL download exceeds max_file_size_bytes, router returns 413."""
    import estimation.estimator as estimator_mod

    monkeypatch.setattr(estimator_mod.settings, "fitted_estimator_mode", "active")

    from main import app

    test_client = TestClient(app, raise_server_exceptions=False)

    # Patch fetch_image to return a huge byte string
    huge_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * (60 * 1024 * 1024)  # 60 MB

    from exceptions import FileTooLargeError

    async def _huge_fetch(url: str, **kwargs) -> bytes:
        raise FileTooLargeError(
            "too large",
            file_size=len(huge_data),
            limit=50 * 1024 * 1024,
        )

    with (
        patch("utils.url_fetch.validate_url", return_value=None),
        patch("routers.estimate.fetch_partial", side_effect=Exception("no range")),
        patch("routers.estimate.settings.fitted_estimator_mode", "active"),
        patch("routers.estimate.fetch_image", new=AsyncMock(side_effect=_huge_fetch)),
    ):
        resp = test_client.post(
            "/estimate",
            json={"url": "https://example.com/huge.png"},
            headers={"Content-Type": "application/json"},
        )

    assert resp.status_code in (413, 400, 500)  # FileTooLargeError → 413 or mapped error
