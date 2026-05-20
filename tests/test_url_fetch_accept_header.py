"""Regression tests: fetch_image() and fetch_partial() send a browser-style Accept header.

Shopify's CDN (and similar origins) content-negotiate on Accept and return
HTTP 404 for the httpx default Accept: */*. This verifies that both fetch
functions include image/* in the outbound Accept header.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_stream_client(status_code: int, body: bytes, headers: dict | None = None):
    """Build a mock client that captures kwargs passed to client.stream()."""
    captured = {}

    async def _aiter():
        yield body

    resp = MagicMock()
    resp.status_code = status_code
    resp.is_redirect = False
    resp.is_success = status_code < 400
    resp.headers = headers or {}
    resp.aiter_bytes = _aiter

    def _stream_ctx(*args, **kwargs):
        captured["kwargs"] = kwargs
        ctx = AsyncMock()
        ctx.__aenter__ = AsyncMock(return_value=resp)
        ctx.__aexit__ = AsyncMock(return_value=None)
        return ctx

    mock_client = MagicMock()
    mock_client.stream = _stream_ctx
    return mock_client, captured


@pytest.mark.asyncio
async def test_fetch_image_sends_image_accept_header():
    """fetch_image() must include image/* in the outbound Accept header."""
    from utils.url_fetch import fetch_image

    mock_client, captured = _make_stream_client(200, b"\xff\xd8\xff" + b"\x00" * 100)

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            await fetch_image("https://example.com/img.jpg")

    accept = captured["kwargs"].get("headers", {}).get("Accept", "")
    assert "image/*" in accept, f"Expected 'image/*' in Accept header, got: {accept!r}"


@pytest.mark.asyncio
async def test_fetch_partial_sends_image_accept_header():
    """fetch_partial() must include image/* in the outbound Accept header."""
    from utils.url_fetch import fetch_partial

    body = b"A" * 100
    mock_client, captured = _make_stream_client(
        206, body, headers={"content-range": "bytes 0-99/8192"}
    )

    with patch("utils.url_fetch.validate_url", return_value=None):
        with patch("utils.url_fetch._get_client", new=AsyncMock(return_value=mock_client)):
            await fetch_partial("https://example.com/img.jpg", byte_range=(0, 99))

    accept = captured["kwargs"].get("headers", {}).get("Accept", "")
    assert "image/*" in accept, f"Expected 'image/*' in Accept header, got: {accept!r}"
