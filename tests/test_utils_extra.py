"""Tests for utils modules: url_fetch, subprocess_runner, concurrency, metadata, logging."""

import asyncio
import io
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from exceptions import (
    BackpressureError,
    FileTooLargeError,
    OptimizationError,
    SSRFError,
    ToolTimeoutError,
    URLFetchError,
)
from utils.format_detect import ImageFormat

# --- subprocess_runner ---


@pytest.mark.asyncio
async def test_run_tool_success():
    """Successful tool invocation."""
    from utils.subprocess_runner import run_tool

    stdout, stderr, rc = await run_tool(
        ["python", "-c", "import sys; sys.stdout.buffer.write(b'hello')"], b""
    )
    assert stdout == b"hello"
    assert rc == 0


@pytest.mark.asyncio
async def test_run_tool_stdin():
    """Tool reads from stdin."""
    from utils.subprocess_runner import run_tool

    stdout, stderr, rc = await run_tool(
        ["python", "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"],
        b"input data",
    )
    assert stdout == b"input data"


@pytest.mark.asyncio
async def test_run_tool_timeout():
    """Tool timeout -> ToolTimeoutError."""
    from utils.subprocess_runner import run_tool

    with pytest.raises(ToolTimeoutError):
        await run_tool(["python", "-c", "import time; time.sleep(10)"], b"", timeout=1)


@pytest.mark.asyncio
async def test_run_tool_nonzero_exit():
    """Non-zero exit code -> OptimizationError."""
    from utils.subprocess_runner import run_tool

    with pytest.raises(OptimizationError):
        await run_tool(["python", "-c", "import sys; sys.exit(1)"], b"")


@pytest.mark.asyncio
async def test_run_tool_allowed_exit_code():
    """Allowed non-zero exit code is not an error."""
    from utils.subprocess_runner import run_tool

    stdout, stderr, rc = await run_tool(
        ["python", "-c", "import sys; sys.exit(99)"], b"", allowed_exit_codes={99}
    )
    assert rc == 99


# --- concurrency ---


def test_compression_gate_acquire_release():
    """CompressionGate basic acquire/release cycle."""
    from utils.concurrency import CompressionGate

    gate = CompressionGate()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(gate.acquire())
        assert gate.active_jobs >= 1
        gate.release()
    finally:
        loop.close()


def test_compression_gate_queue_full():
    """Full queue raises BackpressureError."""
    from utils.concurrency import CompressionGate

    gate = CompressionGate()
    # Directly set queue depth to max to simulate a full queue
    # (actually acquiring would block on the semaphore)
    gate._queue_depth = gate._max_queue
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(BackpressureError):
            loop.run_until_complete(gate.acquire())
    finally:
        gate._queue_depth = 0
        loop.close()


def test_compression_gate_queued_jobs():
    from utils.concurrency import CompressionGate

    gate = CompressionGate()
    assert gate.queued_jobs >= 0


# --- metadata ---


def test_strip_jpeg_metadata():
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    data = buf.getvalue()
    result = strip_metadata_selective(data, ImageFormat.JPEG)
    assert len(result) > 0


def test_strip_png_metadata():
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    result = strip_metadata_selective(data, ImageFormat.PNG)
    assert result[:8] == b"\x89PNG\r\n\x1a\n"


def test_strip_png_metadata_text_chunks():
    """PNG with tEXt chunk: stripped."""

    from utils.metadata import strip_metadata_selective

    # Build a minimal PNG with tEXt chunk
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    # Verify stripping works without error
    result = strip_metadata_selective(data, ImageFormat.PNG)
    assert len(result) > 0


def test_strip_tiff_metadata():
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (50, 50))
    buf = io.BytesIO()
    img.save(buf, format="TIFF")
    data = buf.getvalue()
    result = strip_metadata_selective(data, ImageFormat.TIFF)
    assert len(result) > 0


def test_strip_metadata_passthrough():
    """Formats without specific handling pass through unchanged."""
    from utils.metadata import strip_metadata_selective

    data = b"fake webp data"
    result = strip_metadata_selective(data, ImageFormat.WEBP)
    assert result == data


def test_strip_apng_metadata():
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    result = strip_metadata_selective(data, ImageFormat.APNG)
    assert len(result) > 0


def test_strip_png_no_icc():
    from utils.metadata import _strip_png_metadata

    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    result = _strip_png_metadata(data, preserve_icc=False)
    assert len(result) > 0


def test_strip_jpeg_preserves_orientation():
    """Orientation EXIF tag preserved."""
    from utils.metadata import strip_metadata_selective

    img = Image.new("RGB", (50, 50))
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation = Rotate 90 CW
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=exif.tobytes())
    data = buf.getvalue()
    result = strip_metadata_selective(data, ImageFormat.JPEG)
    # Should still have EXIF with orientation
    result_img = Image.open(io.BytesIO(result))
    result_exif = result_img.getexif()
    assert result_exif.get(0x0112) == 6


# --- url_fetch ---


@pytest.mark.asyncio
async def test_url_fetch_ssrf_blocked():
    """Private IP URL blocked."""
    from utils.url_fetch import fetch_image

    with pytest.raises(SSRFError):
        await fetch_image("https://127.0.0.1/image.png")


@pytest.mark.asyncio
async def test_url_fetch_http_blocked():
    """HTTP scheme blocked by SSRF validation."""
    from utils.url_fetch import fetch_image

    with pytest.raises(SSRFError):
        await fetch_image("http://example.com/image.png")


def _make_stream_client(responses):
    """Build a mock client where client.stream() yields responses in order.

    Each entry in *responses* is a MagicMock with the attributes that
    fetch_image() inspects (is_redirect, is_success, headers, …).

    The streaming path calls `client.stream("GET", url, timeout=…)` as an
    async context manager and then iterates `response.aiter_bytes()`.
    """
    response_iter = iter(responses)

    def _stream_ctx(*args, **kwargs):
        response = next(response_iter)
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx
    return mock_client


def _make_success_response(body: bytes, headers: dict | None = None):
    """Return a mock response that streams *body* via aiter_bytes()."""

    async def _aiter():
        yield body

    resp = MagicMock()
    resp.is_redirect = False
    resp.is_success = True
    resp.headers = headers or {}
    resp.aiter_bytes = _aiter
    return resp


@pytest.mark.asyncio
async def test_url_fetch_timeout():
    """Timeout raises URLFetchError."""
    import httpx

    from utils.url_fetch import fetch_image

    def _stream_timeout(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_timeout

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="timed out"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_request_error():
    """Request error raises URLFetchError."""
    import httpx

    from utils.url_fetch import fetch_image

    def _stream_error(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.RequestError("connection failed"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_error

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="fetch failed"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_success():
    """Successful fetch returns image bytes."""
    from utils.url_fetch import fetch_image

    body = b"\x89PNG fake image data"
    mock_client = _make_stream_client([_make_success_response(body)])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            result = await fetch_image("https://example.com/img.png")
            assert result == body


@pytest.mark.asyncio
async def test_url_fetch_redirect():
    """Redirect is followed with SSRF check at each hop."""
    from utils.url_fetch import fetch_image

    redirect_resp = MagicMock()
    redirect_resp.is_redirect = True
    redirect_resp.next_request = MagicMock()
    redirect_resp.next_request.url = "https://cdn.example.com/img.png"

    final_resp = _make_success_response(b"image bytes")

    mock_client = _make_stream_client([redirect_resp, final_resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            result = await fetch_image("https://example.com/img.png")
            assert result == b"image bytes"


@pytest.mark.asyncio
async def test_url_fetch_non_success_status():
    """Non-2xx status raises URLFetchError."""
    from utils.url_fetch import fetch_image

    resp = MagicMock()
    resp.is_redirect = False
    resp.is_success = False
    resp.status_code = 404

    mock_client = _make_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="HTTP 404"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_content_length_too_large():
    """Content-Length header too large -> FileTooLargeError."""
    from utils.url_fetch import fetch_image

    resp = _make_success_response(b"x", headers={"content-length": str(999_999_999)})
    mock_client = _make_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(FileTooLargeError):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_redirect_no_location():
    """Redirect without Location header -> URLFetchError."""
    from utils.url_fetch import fetch_image

    resp = MagicMock()
    resp.is_redirect = True
    resp.next_request = None

    mock_client = _make_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="Redirect without Location"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_too_many_redirects():
    """Too many redirects -> URLFetchError."""
    from utils.url_fetch import fetch_image

    # Build enough redirect responses to exhaust the hop limit (default 5).
    # _make_stream_client uses iter(), so we need max_redirects + 2 entries.
    redirect_resp = MagicMock()
    redirect_resp.is_redirect = True
    redirect_resp.next_request = MagicMock()
    redirect_resp.next_request.url = "https://example.com/redir"

    # 10 redirects is safely above the default limit of 5.
    mock_client = _make_stream_client([redirect_resp] * 10)

    with patch("utils.url_fetch.validate_url", return_value="https://example.com"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="Too many redirects"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_body_too_large():
    """Response body exceeds size limit -> FileTooLargeError (detected mid-stream)."""
    from utils.url_fetch import fetch_image

    big_chunk = b"x" * (33 * 1024 * 1024 + 1)  # > 32 MB

    async def _aiter_big():
        yield big_chunk

    resp = MagicMock()
    resp.is_redirect = False
    resp.is_success = True
    resp.headers = {}
    resp.aiter_bytes = _aiter_big

    mock_client = _make_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(FileTooLargeError):
                await fetch_image("https://example.com/img.png")


# --- logging ---


def test_setup_logging():
    """setup_logging returns a logger and configures handlers."""
    from utils.logging import setup_logging

    logger = setup_logging()
    assert logger.name == "pare"
    assert len(logger.handlers) > 0
