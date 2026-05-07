"""Tests for the HTTP fetcher in bench/corpus/fetchers/http.py."""

from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest

from bench.corpus.fetchers import (
    FetchHTTPError,
    FetchIntegrityError,
    FetchTooLargeError,
    fetch,
)
from bench.corpus.manifest import SourceSpec

# ---------------------------------------------------------------------------
# Minimal test HTTP server (no new deps)
# ---------------------------------------------------------------------------


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    """Serves files registered on the handler class itself."""

    routes: dict[str, tuple[int, bytes]] = {}  # path -> (status, body)

    def do_GET(self) -> None:  # noqa: N802
        entry = self.routes.get(self.path)
        if entry is None:
            self.send_response(404)
            self.end_headers()
            return
        status, body = entry
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args, **_kwargs) -> None:  # suppress noise in test output
        pass


def _make_handler(routes: dict[str, tuple[int, bytes]]):
    """Return a fresh handler class with the given route table."""

    class Handler(_RequestHandler):
        pass

    Handler.routes = dict(routes)
    return Handler


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def _run_server(routes: dict[str, tuple[int, bytes]]) -> Generator[str, None, None]:
    """Context manager: start a ThreadingHTTPServer, yield base URL, then shut down."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        handler = _make_handler(routes)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            server.shutdown()

    return _ctx()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_fetch_success_then_cache_hit(tmp_cache: Path) -> None:
    """First fetch downloads the file; second fetch hits the cache (no server call)."""
    body = b"hello corpus"
    sha = hashlib.sha256(body).hexdigest()
    routes = {"/img/hello.png": (200, body)}

    with _run_server(routes) as base_url:
        spec = SourceSpec(
            url=f"{base_url}/img/hello.png",
            sha256=sha,
            license="test",
            attribution="test",
        )

        # First call — downloads and caches
        path1 = fetch(spec, tmp_cache)
        assert path1.exists()
        assert path1.read_bytes() == body

        # Poison the route so a second network hit would return garbage
        _make_handler(routes).routes["/img/hello.png"] = (200, b"POISON")

        # Second call — must return same path from cache, no network
        path2 = fetch(spec, tmp_cache)
        assert path2 == path1
        assert path2.read_bytes() == body  # still original bytes


def test_fetch_sha_mismatch_raises_and_no_cache(tmp_cache: Path) -> None:
    """A body that does not match the declared SHA must raise FetchIntegrityError
    and must NOT leave a file at the expected cache path."""
    body = b"real bytes"
    wrong_sha = "a" * 64  # deliberately wrong
    routes = {"/img/bad.png": (200, body)}

    with _run_server(routes) as base_url:
        spec = SourceSpec(
            url=f"{base_url}/img/bad.png",
            sha256=wrong_sha,
            license="test",
            attribution="test",
        )

        with pytest.raises(FetchIntegrityError) as exc_info:
            fetch(spec, tmp_cache)

        assert exc_info.value.expected == wrong_sha
        assert exc_info.value.url == spec.url

        # No corrupt cache entry should exist
        from bench.corpus.fetchers.http import _cache_path

        assert not _cache_path(spec, tmp_cache).exists()


def test_fetch_http_error_raises(tmp_cache: Path) -> None:
    """A 404 from the server must raise FetchHTTPError."""
    routes = {"/img/missing.png": (404, b"not found")}

    with _run_server(routes) as base_url:
        spec = SourceSpec(
            url=f"{base_url}/img/missing.png",
            sha256="a" * 64,
            license="test",
            attribution="test",
        )

        with pytest.raises(FetchHTTPError) as exc_info:
            fetch(spec, tmp_cache)

        assert "404" in str(exc_info.value)


def test_fetch_sends_default_user_agent(tmp_cache: Path) -> None:
    """The fetcher must send a User-Agent header — some origins (e.g. Wikimedia)
    reject requests without one with HTTP 403."""
    from bench.corpus.fetchers.http import DEFAULT_USER_AGENT

    body = b"ua probe"
    sha = hashlib.sha256(body).hexdigest()
    captured: dict[str, str] = {}

    class _UAHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            captured["user_agent"] = self.headers.get("User-Agent", "")
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args, **_kwargs) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _UAHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = SourceSpec(
            url=f"http://127.0.0.1:{port}/probe",
            sha256=sha,
            license="test",
            attribution="test",
        )
        fetch(spec, tmp_cache)
    finally:
        server.shutdown()

    assert captured["user_agent"] == DEFAULT_USER_AGENT


def test_fetch_too_large_raises_and_no_temp_file(tmp_cache: Path) -> None:
    """A response exceeding MAX_DOWNLOAD_BYTES mid-stream must raise FetchTooLargeError
    and must NOT leave any temp file behind."""
    body = b"X" * 4096  # 4 KB body
    sha = hashlib.sha256(body).hexdigest()
    routes = {"/img/big.png": (200, body)}

    with _run_server(routes) as base_url:
        spec = SourceSpec(
            url=f"{base_url}/img/big.png",
            sha256=sha,
            license="test",
            attribution="test",
        )

        # Override the module-level constant to 1 KB so our 4 KB body triggers it
        with patch("bench.corpus.fetchers.http.MAX_DOWNLOAD_BYTES", 1024):
            with pytest.raises(FetchTooLargeError):
                fetch(spec, tmp_cache)

        # No leftover temp files in the cache tree
        from bench.corpus.fetchers.http import _cache_path

        cache_dir = _cache_path(spec, tmp_cache).parent
        if cache_dir.exists():
            tmp_files = list(cache_dir.glob("*.tmp"))
            assert tmp_files == [], f"leftover temp files: {tmp_files}"
