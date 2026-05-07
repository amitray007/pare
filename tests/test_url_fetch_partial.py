"""Tests for utils/url_fetch.py — fetch_partial() Range-request function."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from exceptions import URLFetchError

# ---------------------------------------------------------------------------
# Shared mock builders (mirror test_utils_extra.py patterns)
# ---------------------------------------------------------------------------


def _make_partial_stream_client(responses):
    """Build a mock client where client.stream() yields responses in order."""
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


def _make_response(
    status_code: int,
    body: bytes,
    headers: dict | None = None,
) -> MagicMock:
    """Return a mock response with the given status code and body."""

    async def _aiter():
        yield body

    resp = MagicMock()
    resp.status_code = status_code
    resp.is_redirect = False
    resp.headers = headers or {}
    resp.aiter_bytes = _aiter
    return resp


# ---------------------------------------------------------------------------
# 206 happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_206_happy_path():
    """206 with Content-Range header returns correct bytes and total_size."""
    from utils.url_fetch import fetch_partial

    body = b"A" * 100
    resp = _make_response(
        206,
        body,
        headers={"content-range": "bytes 0-99/8192"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 99))

    assert len(data) == 100
    assert total == 8192


@pytest.mark.asyncio
async def test_fetch_partial_206_correct_body():
    """206 response body is returned correctly."""
    from utils.url_fetch import fetch_partial

    body = bytes(range(50))  # 50 distinct bytes
    resp = _make_response(
        206,
        body,
        headers={"content-range": "bytes 0-49/4096"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 49))

    assert data == body
    assert total == 4096


@pytest.mark.asyncio
async def test_fetch_partial_206_no_content_range_header():
    """206 without Content-Range header returns total_size=None."""
    from utils.url_fetch import fetch_partial

    body = b"x" * 50
    resp = _make_response(206, body, headers={})
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 49))

    assert total is None


@pytest.mark.asyncio
async def test_fetch_partial_206_wildcard_total():
    """Content-Range: bytes 0-49/* → total_size is None."""
    from utils.url_fetch import fetch_partial

    body = b"y" * 50
    resp = _make_response(
        206,
        body,
        headers={"content-range": "bytes 0-49/*"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 49))

    assert total is None


# ---------------------------------------------------------------------------
# 200 fallback (origin ignored Range)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_200_fallback_truncates():
    """200 response (origin ignored Range): body truncated to byte_range end, total from CL."""
    from utils.url_fetch import fetch_partial

    full_body = b"Z" * 16384  # Much larger than requested range
    resp = _make_response(
        200,
        full_body,
        headers={"content-length": "16384"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 8191))

    # Hard cap: at most 8192 bytes (end - start + 1)
    assert len(data) <= 8192
    assert total == 16384


@pytest.mark.asyncio
async def test_fetch_partial_200_fallback_no_content_length():
    """200 without Content-Length → total_size=None, body still truncated."""
    from utils.url_fetch import fetch_partial

    full_body = b"Q" * 4096
    resp = _make_response(200, full_body, headers={})
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 99))

    assert len(data) <= 100
    assert total is None


# ---------------------------------------------------------------------------
# 416 Range Not Satisfiable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_416_returns_empty():
    """416 response returns (b'', None) — caller handles."""
    from utils.url_fetch import fetch_partial

    resp = _make_response(416, b"", headers={})
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 8191))

    assert data == b""
    assert total is None


# ---------------------------------------------------------------------------
# Server sends more bytes than requested → reader truncates
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_truncates_oversized_server_response():
    """Server returns more bytes than requested — reader never holds more than max_bytes."""
    from utils.url_fetch import fetch_partial

    # Server streams 64KB even though we asked for 100 bytes
    huge_body = b"X" * 65536
    resp = _make_response(
        206,
        huge_body,
        headers={"content-range": "bytes 0-99/65536"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 99))

    # Must never exceed byte_range end - start + 1
    assert len(data) <= 100
    assert total == 65536


# ---------------------------------------------------------------------------
# Redirect — SSRF validated at each hop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_redirect_ssrf_validated():
    """Redirect triggers SSRF validation at each hop."""
    from utils.url_fetch import fetch_partial

    redirect_resp = MagicMock()
    redirect_resp.status_code = 301
    redirect_resp.is_redirect = True
    redirect_resp.next_request = MagicMock()
    redirect_resp.next_request.url = "https://cdn.example.com/img.png"

    body = b"B" * 100
    final_resp = _make_response(
        206,
        body,
        headers={"content-range": "bytes 0-99/4096"},
    )

    mock_client = _make_partial_stream_client([redirect_resp, final_resp])

    validate_calls = []

    def _mock_validate(url: str) -> str:
        validate_calls.append(url)
        return url

    with patch("utils.url_fetch.validate_url", side_effect=_mock_validate):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 99))

    # validate_url must be called at least twice: once for the initial URL,
    # once for the redirect target.
    assert len(validate_calls) >= 2
    assert "https://example.com/img.png" in validate_calls
    assert "https://cdn.example.com/img.png" in validate_calls
    assert len(data) == 100
    assert total == 4096


# ---------------------------------------------------------------------------
# Timeout and request errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_timeout_raises_url_fetch_error():
    """TimeoutException is surfaced as URLFetchError."""
    from utils.url_fetch import fetch_partial

    def _stream_timeout(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_timeout

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="timed out"):
                await fetch_partial("https://example.com/img.png", byte_range=(0, 8191))


@pytest.mark.asyncio
async def test_fetch_partial_request_error_raises_url_fetch_error():
    """RequestError is surfaced as URLFetchError."""
    from utils.url_fetch import fetch_partial

    def _stream_error(*args, **kwargs):
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(side_effect=httpx.RequestError("connection refused"))
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_error

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="fetch failed"):
                await fetch_partial("https://example.com/img.png", byte_range=(0, 8191))


# ---------------------------------------------------------------------------
# Non-2xx non-416 error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_404_raises_url_fetch_error():
    """404 response raises URLFetchError."""
    from utils.url_fetch import fetch_partial

    resp = _make_response(404, b"not found", headers={})
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value="https://example.com/img.png"):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="HTTP 404"):
                await fetch_partial("https://example.com/img.png", byte_range=(0, 8191))


# ---------------------------------------------------------------------------
# SSRF blocks initial URL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_ssrf_blocked():
    """Private IP URL is blocked by SSRF validation before any request."""
    from exceptions import SSRFError
    from utils.url_fetch import fetch_partial

    with pytest.raises(SSRFError):
        await fetch_partial("https://127.0.0.1/img.png", byte_range=(0, 8191))


# ---------------------------------------------------------------------------
# Redirect without Location header → URLFetchError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_redirect_no_next_request_raises():
    """Redirect response with no next_request (no Location header) raises URLFetchError."""
    from utils.url_fetch import fetch_partial

    redirect_resp = MagicMock()
    redirect_resp.status_code = 301
    redirect_resp.is_redirect = True
    redirect_resp.next_request = None  # no Location header

    mock_client = _make_partial_stream_client([redirect_resp])

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="Redirect without Location header"):
                await fetch_partial("https://example.com/img.png", byte_range=(0, 99))


# ---------------------------------------------------------------------------
# Too many redirects → URLFetchError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_too_many_redirects_raises():
    """Redirect loop exhausted → URLFetchError('Too many redirects')."""
    from utils.url_fetch import fetch_partial

    # Build max_redirects + 1 redirect responses, all pointing elsewhere
    def _make_redirect(target: str) -> MagicMock:
        r = MagicMock()
        r.status_code = 301
        r.is_redirect = True
        r.next_request = MagicMock()
        r.next_request.url = target
        return r

    # 6 redirects (default max is 5, so this exhausts the loop)
    redirects = [_make_redirect(f"https://cdn{i}.example.com/img.png") for i in range(6)]
    mock_client = _make_partial_stream_client(redirects)

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            with pytest.raises(URLFetchError, match="Too many redirects"):
                await fetch_partial("https://example.com/img.png", byte_range=(0, 99))


# ---------------------------------------------------------------------------
# 206 with malformed Content-Range header → total_size=None (no crash)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_206_malformed_content_range():
    """206 with non-parseable Content-Range → total_size=None (ValueError branch)."""
    from utils.url_fetch import fetch_partial

    body = b"A" * 50
    resp = _make_response(
        206,
        body,
        headers={"content-range": "bytes 0-49/notanumber"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 49))

    assert total is None
    assert len(data) == 50


# ---------------------------------------------------------------------------
# 206 — reader loop hits remaining <= 0 guard before last chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_206_remaining_zero_guard():
    """206 reader: when buf fills exactly, remaining<=0 guard fires on next iteration."""
    from utils.url_fetch import fetch_partial

    # Simulate server streaming 3 chunks: first fills the buffer exactly, then more
    chunk1 = b"X" * 10
    chunk2 = b"Y" * 10  # this chunk would overflow; guard must prevent it

    async def _aiter_multi():
        yield chunk1
        yield chunk2

    resp = MagicMock()
    resp.status_code = 206
    resp.is_redirect = False
    resp.headers = {"content-range": "bytes 0-9/100"}
    resp.aiter_bytes = _aiter_multi

    def _stream_ctx(*args, **kwargs):
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 9))

    # Must be capped at max_bytes = 10
    assert len(data) <= 10
    assert total == 100


# ---------------------------------------------------------------------------
# 200 with malformed Content-Length → total_size=None (ValueError branch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_200_malformed_content_length():
    """200 with non-integer Content-Length → total_size=None (ValueError branch)."""
    from utils.url_fetch import fetch_partial

    body = b"Q" * 50
    resp = _make_response(
        200,
        body,
        headers={"content-length": "notanumber"},
    )
    mock_client = _make_partial_stream_client([resp])

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 99))

    assert total is None
    assert len(data) <= 100


# ---------------------------------------------------------------------------
# 200 — reader loop hits remaining <= 0 guard before last chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_200_remaining_zero_guard():
    """200 reader: when buf fills exactly, remaining<=0 guard fires on next chunk."""
    from utils.url_fetch import fetch_partial

    chunk1 = b"Z" * 10
    chunk2 = b"W" * 10  # overflow chunk — guard must prevent read

    async def _aiter_multi():
        yield chunk1
        yield chunk2

    resp = MagicMock()
    resp.status_code = 200
    resp.is_redirect = False
    resp.headers = {}
    resp.aiter_bytes = _aiter_multi

    def _stream_ctx(*args, **kwargs):
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            data, total = await fetch_partial("https://example.com/img.png", byte_range=(0, 9))

    assert len(data) <= 10


# ---------------------------------------------------------------------------
# Input validation: invalid byte_range raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_partial_negative_start_raises_value_error():
    """byte_range with negative start raises ValueError."""
    from utils.url_fetch import fetch_partial

    with patch("utils.url_fetch.validate_url", return_value=None):
        with pytest.raises(ValueError, match="Invalid byte_range"):
            await fetch_partial("https://example.com/img.png", byte_range=(-1, 100))


@pytest.mark.asyncio
async def test_fetch_partial_end_before_start_raises_value_error():
    """byte_range where end < start raises ValueError."""
    from utils.url_fetch import fetch_partial

    with patch("utils.url_fetch.validate_url", return_value=None):
        with pytest.raises(ValueError, match="Invalid byte_range"):
            await fetch_partial("https://example.com/img.png", byte_range=(100, 50))


@pytest.mark.asyncio
async def test_fetch_partial_negative_end_raises_value_error():
    """byte_range where end is negative (end < start=0) raises ValueError."""
    from utils.url_fetch import fetch_partial

    with patch("utils.url_fetch.validate_url", return_value=None):
        with pytest.raises(ValueError, match="Invalid byte_range"):
            await fetch_partial("https://example.com/img.png", byte_range=(0, -1))
