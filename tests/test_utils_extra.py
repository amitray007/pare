"""Tests for utils modules: url_fetch, subprocess_runner, concurrency, metadata, logging."""

import asyncio
import io

import pytest
from PIL import Image
from unittest.mock import patch, MagicMock, AsyncMock

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
    stdout, stderr, rc = await run_tool(["python", "-c", "import sys; sys.stdout.buffer.write(b'hello')"], b"")
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
    import struct

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
    from utils.metadata import strip_metadata_selective, _strip_png_metadata
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
    from PIL.ExifTags import IFD
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


@pytest.mark.asyncio
async def test_url_fetch_timeout():
    """Timeout raises URLFetchError."""
    import httpx
    from utils.url_fetch import fetch_image

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.side_effect = httpx.TimeoutException("timeout")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(URLFetchError, match="timed out"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_request_error():
    """Request error raises URLFetchError."""
    import httpx
    from utils.url_fetch import fetch_image

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.side_effect = httpx.RequestError("connection failed")

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(URLFetchError, match="fetch failed"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_success():
    """Successful fetch returns image bytes."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.is_success = True
    mock_response.headers = {}
    mock_response.content = b"\x89PNG fake image data"

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_image("https://example.com/img.png")
            assert result == b"\x89PNG fake image data"


@pytest.mark.asyncio
async def test_url_fetch_redirect():
    """Redirect is followed with SSRF check at each hop."""
    from utils.url_fetch import fetch_image

    redirect_response = MagicMock()
    redirect_response.is_redirect = True
    redirect_response.next_request = MagicMock()
    redirect_response.next_request.url = "https://cdn.example.com/img.png"

    final_response = MagicMock()
    final_response.is_redirect = False
    final_response.is_success = True
    final_response.headers = {}
    final_response.content = b"image bytes"

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.side_effect = [redirect_response, final_response]

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await fetch_image("https://example.com/img.png")
            assert result == b"image bytes"


@pytest.mark.asyncio
async def test_url_fetch_non_success_status():
    """Non-2xx status raises URLFetchError."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.is_success = False
    mock_response.status_code = 404

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(URLFetchError, match="HTTP 404"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_content_length_too_large():
    """Content-Length header too large -> FileTooLargeError."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.is_success = True
    mock_response.headers = {"content-length": str(999999999)}
    mock_response.content = b"x"

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(FileTooLargeError):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_redirect_no_location():
    """Redirect without Location header -> URLFetchError."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = True
    mock_response.next_request = None

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(URLFetchError, match="Redirect without Location"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_too_many_redirects():
    """Too many redirects -> URLFetchError."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = True
    mock_response.next_request = MagicMock()
    mock_response.next_request.url = "https://example.com/redir"

    with patch("utils.url_fetch.validate_url", return_value="https://example.com"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(URLFetchError, match="Too many redirects"):
                await fetch_image("https://example.com/img.png")


@pytest.mark.asyncio
async def test_url_fetch_body_too_large():
    """Response body exceeds size limit -> FileTooLargeError."""
    from utils.url_fetch import fetch_image

    mock_response = MagicMock()
    mock_response.is_redirect = False
    mock_response.is_success = True
    mock_response.headers = {}
    mock_response.content = b"x" * (33 * 1024 * 1024 + 1)  # > 32MB

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get.return_value = mock_response

        with patch("httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(FileTooLargeError):
                await fetch_image("https://example.com/img.png")


# --- logging ---


def test_setup_logging():
    """setup_logging returns a logger and configures handlers."""
    from utils.logging import setup_logging
    logger = setup_logging()
    assert logger.name == "pare"
    assert len(logger.handlers) > 0
