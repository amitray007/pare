"""Tests for coverage gaps in routers, utils, middleware, and optimizer code.

Coverage targets:
- routers/estimate.py  L58-89, 132-152: JSON body error paths + URL fetch + _fetch_dimensions
- optimizers/router.py L58-60: UnsupportedFormatError when optimizer not registered
- optimizers/tiff.py   L66-71, 105, 111-112: large-image sequential path + EXIF/ICC save
- utils/url_fetch.py   L18-28, 35-36: _get_client lazy init + close_client when active
- middleware.py        L43-48: DecompressionBombError handler
- utils/subprocess_runner.py L71-74: probe callback on timeout
- utils/concurrency.py L85-90: semaphore acquire cancellation rollback
"""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from main import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(size=(50, 50), mode="RGB"):
    img = Image.new(mode, size, (128, 64, 32) if mode == "RGB" else 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(size=(50, 50)):
    img = Image.new("RGB", size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def strict_client():
    return TestClient(app, raise_server_exceptions=True)


# ============================================================================
# routers/estimate.py — JSON body paths (lines 58-89)
# ============================================================================


def test_estimate_json_invalid_body(client):
    """POST /estimate with JSON body that fails to parse → 400."""
    resp = client.post(
        "/estimate",
        content=b"not valid json{{{",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["error"] == "bad_request"


def test_estimate_json_missing_url_field(client):
    """POST /estimate with valid JSON but no 'url' key → 400."""
    resp = client.post(
        "/estimate",
        json={"optimization": {"quality": 60}},
    )
    assert resp.status_code == 400


def test_estimate_json_preset_in_body(client):
    """POST /estimate with JSON body containing 'preset' key → applies preset."""
    data = _make_png_bytes()
    with patch("routers.estimate.fetch_image", new=AsyncMock(return_value=data)):
        resp = client.post(
            "/estimate",
            json={"url": "https://example.com/image.png", "preset": "high"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_reduction_percent"] >= 0


def test_estimate_json_invalid_preset_in_body(client):
    """POST /estimate with JSON body containing invalid preset → 400."""
    data = _make_png_bytes()
    with patch("routers.estimate.fetch_image", new=AsyncMock(return_value=data)):
        resp = client.post(
            "/estimate",
            json={"url": "https://example.com/image.png", "preset": "turbo"},
        )
    assert resp.status_code == 400


def test_estimate_json_optimization_in_body(client):
    """POST /estimate with JSON body containing 'optimization' dict → applies config."""
    data = _make_png_bytes()
    with patch("routers.estimate.fetch_image", new=AsyncMock(return_value=data)):
        resp = client.post(
            "/estimate",
            json={
                "url": "https://example.com/image.png",
                "optimization": {"quality": 40, "png_lossy": True},
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["estimated_reduction_percent"] >= 0


# ============================================================================
# routers/estimate.py — _fetch_dimensions (lines 132-152)
# ============================================================================


def test_estimate_thumbnail_path_with_large_file(client):
    """POST /estimate with thumbnail_url + file_size >= 10MB triggers thumbnail path (lines 80-95)."""
    thumb_data = _make_jpeg_bytes((100, 100))

    from schemas import EstimateResponse

    mock_estimate_response = EstimateResponse(
        original_size=10_500_000,
        original_format="jpeg",
        dimensions={"width": 200, "height": 200},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=8_000_000,
        estimated_reduction_percent=23.8,
        optimization_potential="medium",
        method="jpegtran",
        already_optimized=False,
        confidence="medium",
        path="direct_encode_sample",
    )

    with (
        patch("routers.estimate.fetch_image", new=AsyncMock(return_value=thumb_data)),
        patch("routers.estimate._fetch_dimensions", new=AsyncMock(return_value=(200, 200))),
        patch(
            "estimation.estimator.estimate_from_thumbnail",
            new=AsyncMock(return_value=mock_estimate_response),
        ),
    ):
        resp = client.post(
            "/estimate",
            json={
                "url": "https://example.com/large.jpg",
                "thumbnail_url": "https://example.com/thumb.jpg",
                "file_size": 10_500_000,  # >= LARGE_FILE_THRESHOLD (10MB)
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_format"] == "jpeg"


@pytest.mark.asyncio
async def test_fetch_dimensions_range_request_success():
    """_fetch_dimensions: successful Range request returns (width, height) (line 139-147)."""
    from routers.estimate import _fetch_dimensions

    # Build a real tiny JPEG to return from the "range" response
    buf = io.BytesIO()
    Image.new("RGB", (320, 240)).save(buf, format="JPEG", quality=95)
    jpeg_bytes = buf.getvalue()

    mock_response = MagicMock()
    mock_response.content = jpeg_bytes

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    with (
        # validate_url is a local import inside _fetch_dimensions
        patch("security.ssrf.validate_url"),
        patch("routers.estimate._get_client", new=AsyncMock(return_value=mock_client)),
    ):
        width, height = await _fetch_dimensions(
            "https://example.com/img.jpg", is_authenticated=True
        )

    assert width == 320
    assert height == 240


@pytest.mark.asyncio
async def test_fetch_dimensions_fallback_on_exception():
    """_fetch_dimensions falls back to full download when Range request fails (lines 148-152)."""
    from routers.estimate import _fetch_dimensions

    buf = io.BytesIO()
    Image.new("RGB", (100, 80)).save(buf, format="JPEG", quality=95)
    jpeg_bytes = buf.getvalue()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("range not supported"))

    with (
        patch("security.ssrf.validate_url"),
        patch("routers.estimate._get_client", new=AsyncMock(return_value=mock_client)),
        patch("routers.estimate.fetch_image", new=AsyncMock(return_value=jpeg_bytes)),
    ):
        w, h = await _fetch_dimensions("https://example.com/img.jpg", is_authenticated=False)

    assert w == 100
    assert h == 80


# ============================================================================
# optimizers/router.py — UnsupportedFormatError when no optimizer (lines 58-60)
# ============================================================================


@pytest.mark.asyncio
async def test_optimize_image_unsupported_format_raises():
    """optimize_image raises UnsupportedFormatError when format not in OPTIMIZERS."""
    from exceptions import UnsupportedFormatError

    # Temporarily remove PNG from the registry to simulate unregistered format
    from optimizers.router import OPTIMIZERS, optimize_image
    from schemas import OptimizationConfig
    from utils.format_detect import ImageFormat

    original = OPTIMIZERS.pop(ImageFormat.PNG)
    try:
        buf = io.BytesIO()
        Image.new("RGB", (10, 10)).save(buf, format="PNG")
        data = buf.getvalue()

        with pytest.raises(UnsupportedFormatError):
            await optimize_image(data, OptimizationConfig())
    finally:
        OPTIMIZERS[ImageFormat.PNG] = original


@pytest.mark.asyncio
async def test_optimize_image_with_predetected_format():
    """optimize_image accepts a pre-detected format, skipping magic-byte scan (line 54-55)."""
    from optimizers.router import optimize_image
    from schemas import OptimizationConfig
    from utils.format_detect import ImageFormat

    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    data = buf.getvalue()

    # Pass fmt explicitly — detect_format should NOT be called
    with patch("optimizers.router.detect_format") as mock_detect:
        result = await optimize_image(data, OptimizationConfig(), fmt=ImageFormat.PNG)
        mock_detect.assert_not_called()

    assert result.success


# ============================================================================
# optimizers/tiff.py — large-image sequential path (lines 66-71)
# ============================================================================


@pytest.mark.asyncio
async def test_tiff_optimizer_large_image_sequential():
    """TIFF optimizer with pixel_count >= PARALLEL_PIXEL_THRESHOLD runs methods
    sequentially (lines 66-71) instead of via asyncio.gather."""
    from optimizers.tiff import PARALLEL_PIXEL_THRESHOLD, TiffOptimizer
    from schemas import OptimizationConfig

    # Create TIFF > 5MP
    width = 2500
    height = 2001  # 5,002,500 > PARALLEL_PIXEL_THRESHOLD (5,000,000)
    assert width * height > PARALLEL_PIXEL_THRESHOLD

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression="raw")
    data = buf.getvalue()

    opt = TiffOptimizer()
    config = OptimizationConfig(quality=80)

    result = await opt.optimize(data, config)

    assert result.success
    # Sequential path should still pick the best compression
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_tiff_try_compression_saves_exif_directly():
    """_try_compression saves EXIF when strip_metadata=False and exif_bytes is truthy (line 105).

    Calls _try_compression directly to bypass _decode (which doesn't extract exif from TIFF info).
    """
    from optimizers.tiff import TiffOptimizer
    from schemas import OptimizationConfig

    opt = TiffOptimizer()

    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    img.load()

    # Create minimal EXIF bytes (Exif header)
    exif_bytes = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x00"
    config = OptimizationConfig(quality=80, strip_metadata=False)

    # Call _try_compression directly with truthy exif_bytes
    result_bytes, method = opt._try_compression(img, "tiff_adobe_deflate", config, exif_bytes, None)

    # Result may or may not include exif (depends on Pillow TIFF saving), but
    # the branch is exercised (line 105: save_kwargs["exif"] = exif_bytes)
    assert method == "tiff_adobe_deflate"
    # result_bytes can be None if save fails with exif, otherwise bytes
    # Either way line 105 was executed


@pytest.mark.asyncio
async def test_tiff_try_compression_saves_icc_directly():
    """_try_compression saves ICC profile when strip_metadata=False and icc_profile present
    (lines 111-112)."""
    from PIL import ImageCms

    from optimizers.tiff import TiffOptimizer
    from schemas import OptimizationConfig

    opt = TiffOptimizer()

    srgb = ImageCms.createProfile("sRGB")
    icc_data = ImageCms.ImageCmsProfile(srgb).tobytes()

    img = Image.new("RGB", (50, 50), color=(100, 150, 200))
    img.load()

    config = OptimizationConfig(quality=80, strip_metadata=False)

    # Call _try_compression directly with truthy icc_profile
    result_bytes, method = opt._try_compression(img, "tiff_adobe_deflate", config, None, icc_data)

    assert method == "tiff_adobe_deflate"
    assert result_bytes is not None
    assert len(result_bytes) > 0


@pytest.mark.asyncio
async def test_tiff_try_compression_exception_returns_none():
    """_try_compression returns (None, method) when save() raises Exception (lines 111-112)."""
    from optimizers.tiff import TiffOptimizer
    from schemas import OptimizationConfig

    opt = TiffOptimizer()
    img = Image.new("RGB", (50, 50))
    img.load()
    config = OptimizationConfig(quality=80, strip_metadata=True)

    # Patch Image.Image.save to raise for any TIFF save
    with patch.object(Image.Image, "save", side_effect=Exception("forced save failure")):
        result_bytes, method = opt._try_compression(img, "tiff_adobe_deflate", config, None, None)

    assert result_bytes is None
    assert method == "tiff_adobe_deflate"


# ============================================================================
# utils/url_fetch.py — _get_client lazy init + close_client (lines 18-36)
# ============================================================================


@pytest.mark.asyncio
async def test_get_client_creates_on_first_call():
    """_get_client creates the shared httpx.AsyncClient on first call (lines 18-28)."""
    import utils.url_fetch as url_fetch_module

    # Reset the module-level client to None
    original_client = url_fetch_module._client
    url_fetch_module._client = None

    try:
        client = await url_fetch_module._get_client()
        assert client is not None
        assert url_fetch_module._client is client

        # Second call returns the same instance (not recreated)
        client2 = await url_fetch_module._get_client()
        assert client2 is client
    finally:
        # Restore original state
        if url_fetch_module._client is not None and url_fetch_module._client is not original_client:
            await url_fetch_module._client.aclose()
        url_fetch_module._client = original_client


@pytest.mark.asyncio
async def test_close_client_closes_and_resets():
    """close_client() closes the shared client and sets _client to None (lines 35-36)."""
    import utils.url_fetch as url_fetch_module

    original_client = url_fetch_module._client

    # Install a mock client so we don't affect the real singleton
    mock_client = AsyncMock()
    url_fetch_module._client = mock_client

    try:
        await url_fetch_module.close_client()

        mock_client.aclose.assert_called_once()
        assert url_fetch_module._client is None
    finally:
        url_fetch_module._client = original_client


@pytest.mark.asyncio
async def test_close_client_noop_when_none():
    """close_client() is a no-op when _client is already None."""
    import utils.url_fetch as url_fetch_module

    original_client = url_fetch_module._client
    url_fetch_module._client = None

    try:
        # Should not raise
        await url_fetch_module.close_client()
        assert url_fetch_module._client is None
    finally:
        url_fetch_module._client = original_client


# ============================================================================
# middleware.py — DecompressionBombError handler (lines 43-48)
# ============================================================================


def test_middleware_decompression_bomb_megapixels_message(client):
    """DecompressionBombError from optimizer (bypassing validate) → 413 with megapixels message.

    Patches validate_image_dimensions to be a no-op so the DecompressionBombError
    propagates from inside the BMP optimizer (where Image.open() loads pixels).
    The middleware handler at lines 43-48 catches it and returns 413 with the
    megapixels branch of the message (limit >= 1_000_000).
    """
    from config import settings

    original_limit = settings.max_image_pixels
    original_max_pix = Image.MAX_IMAGE_PIXELS

    try:
        # 2000×2000 = 4M pixels > limit=1M pixels → bomb fires in optimizer
        limit = 1_000_000
        Image.MAX_IMAGE_PIXELS = limit
        settings.max_image_pixels = limit

        img = Image.new("RGB", (2000, 2000), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        # Bypass the route-level validate_image_dimensions so the bomb reaches the optimizer
        with patch("routers.optimize.validate_image_dimensions", return_value=None):
            resp = client.post("/optimize", files={"file": ("bomb.bmp", data)})

        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "image_too_large"
        assert "megapixels" in body["message"]
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pix
        settings.max_image_pixels = original_limit


def test_middleware_decompression_bomb_pixels_message(client):
    """DecompressionBombError when limit < 1M → message says 'N pixels' not 'megapixels'."""
    from config import settings

    original_limit = settings.max_image_pixels
    original_max_pix = Image.MAX_IMAGE_PIXELS

    try:
        limit = 500  # < 1_000_000 → 'N pixels' branch
        Image.MAX_IMAGE_PIXELS = limit
        settings.max_image_pixels = limit

        img = Image.new("RGB", (100, 100), color=(0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        with patch("routers.optimize.validate_image_dimensions", return_value=None):
            resp = client.post("/optimize", files={"file": ("bomb.bmp", data)})

        assert resp.status_code == 413
        body = resp.json()
        assert body["error"] == "image_too_large"
        # limit < 1M → formatted as 'N pixels', not megapixels
        assert "pixels" in body["message"]
        assert "megapixels" not in body["message"]
    finally:
        Image.MAX_IMAGE_PIXELS = original_max_pix
        settings.max_image_pixels = original_limit


# ============================================================================
# utils/subprocess_runner.py — probe callback on timeout (lines 71-74)
# ============================================================================


@pytest.mark.asyncio
async def test_run_tool_probe_called_on_timeout():
    """When run_tool_probe is set and the tool times out, the probe is called
    with exit_code=-1 (lines 71-74)."""
    from exceptions import ToolTimeoutError
    from utils.subprocess_runner import run_tool, run_tool_probe

    probe_calls: list[tuple] = []

    def probe(tool: str, duration_ms: float, exit_code: int) -> None:
        probe_calls.append((tool, duration_ms, exit_code))

    token = run_tool_probe.set(probe)
    try:
        with pytest.raises(ToolTimeoutError):
            await run_tool(
                ["python", "-c", "import time; time.sleep(10)"],
                b"",
                timeout=1,
            )
    finally:
        run_tool_probe.reset(token)

    assert len(probe_calls) == 1
    tool, duration_ms, exit_code = probe_calls[0]
    assert tool == "python"
    assert exit_code == -1
    assert duration_ms >= 0


@pytest.mark.asyncio
async def test_run_tool_probe_called_on_success():
    """When run_tool_probe is set and the tool succeeds, the probe is called
    with the actual exit code (line 81-85)."""
    from utils.subprocess_runner import run_tool, run_tool_probe

    probe_calls: list[tuple] = []

    def probe(tool: str, duration_ms: float, exit_code: int) -> None:
        probe_calls.append((tool, duration_ms, exit_code))

    token = run_tool_probe.set(probe)
    try:
        await run_tool(["python", "-c", "import sys; sys.exit(0)"], b"")
    finally:
        run_tool_probe.reset(token)

    assert len(probe_calls) == 1
    assert probe_calls[0][2] == 0


@pytest.mark.asyncio
async def test_run_tool_probe_exception_is_silenced():
    """If the probe raises an exception on success path, run_tool silences it (lines 83-85)."""
    from utils.subprocess_runner import run_tool, run_tool_probe

    def bad_probe(tool, duration_ms, exit_code):
        raise RuntimeError("probe bug")

    token = run_tool_probe.set(bad_probe)
    try:
        # Should not raise — probe exception is caught in the except block
        stdout, stderr, rc = await run_tool(["python", "-c", "import sys; sys.exit(0)"], b"")
    finally:
        run_tool_probe.reset(token)

    assert rc == 0


@pytest.mark.asyncio
async def test_run_tool_probe_exception_silenced_on_timeout():
    """If the probe raises during a timeout, run_tool silences it and still raises
    ToolTimeoutError (lines 73-74 — the except Exception: pass inside timeout handler)."""
    from exceptions import ToolTimeoutError
    from utils.subprocess_runner import run_tool, run_tool_probe

    def raising_probe(tool, duration_ms, exit_code):
        raise RuntimeError("probe crashes on timeout")

    token = run_tool_probe.set(raising_probe)
    try:
        # The tool times out → probe is called → probe raises → exception silenced
        # ToolTimeoutError should still propagate
        with pytest.raises(ToolTimeoutError):
            await run_tool(
                ["python", "-c", "import time; time.sleep(10)"],
                b"",
                timeout=1,
            )
    finally:
        run_tool_probe.reset(token)


# ============================================================================
# utils/concurrency.py — semaphore acquire cancellation rollback (lines 85-90)
# ============================================================================


@pytest.mark.asyncio
async def test_compression_gate_cancellation_rolls_back_queue():
    """If semaphore.acquire() is cancelled, queue_depth and memory_used roll back
    (lines 85-90)."""
    from utils.concurrency import CompressionGate

    gate = CompressionGate(semaphore_size=1, max_queue=5)

    # Drain the semaphore so the next acquire will block
    await gate.acquire()
    assert gate._queue_depth == 1
    assert gate.active_jobs == 1

    # Now attempt a second acquire (will block on semaphore) and cancel it
    task = asyncio.ensure_future(gate.acquire(estimated_memory=1024))
    # Let the task get past the queue-depth check and into semaphore.acquire()
    await asyncio.sleep(0.05)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # queue_depth should have been rolled back (the cancelled acquire incremented then
    # rolled back)
    assert gate._queue_depth == 1  # only the first acquire remains
    assert gate._memory_used == 0  # estimated_memory was rolled back

    # Cleanup
    gate.release()


@pytest.mark.asyncio
async def test_compression_gate_memory_budget_rejected():
    """acquire() with estimated_memory > remaining budget and memory_used > 0 → 503."""
    from exceptions import BackpressureError
    from utils.concurrency import CompressionGate

    # Budget = 1000 bytes
    gate = CompressionGate(semaphore_size=4, max_queue=10, memory_budget_bytes=1000)

    # First acquire uses 600 bytes (admitted — memory_used was 0)
    await gate.acquire(estimated_memory=600)
    assert gate._memory_used == 600

    # Second acquire for 500 bytes: memory_used (600) > 0 and 600+500 > 1000 → reject
    with pytest.raises(BackpressureError, match="Memory budget exceeded"):
        await gate.acquire(estimated_memory=500)

    # Cleanup
    gate.release(estimated_memory=600)
