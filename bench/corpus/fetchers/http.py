"""HTTP fetcher with SHA-256 integrity verification and local caching.

Cache layout:
    <cache_root>/<sha256[:2]>/<sha256>/<basename(url)>

The cache is content-addressed: if a file with the expected SHA exists at
the expected path, no network call is made.  Corrupt or stale entries are
deleted and re-fetched.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.parse
from pathlib import Path

import httpx

from bench.corpus.manifest import SourceSpec

DEFAULT_CACHE_ROOT = Path("bench/corpus/cache")

# 100 MB guard — large enough for any reasonable source image
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024

# Some origins (notably upload.wikimedia.org) reject requests without a
# User-Agent. A polite identifier with a project URL also makes it easier
# for upstream maintainers to reach out if our fetcher misbehaves.
DEFAULT_USER_AGENT = (
    "pare-bench-corpus/1.0 (+https://github.com/amitray007/pare; benchmark fetcher)"
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FetchError(Exception):
    """Base class for all fetcher errors."""


class FetchIntegrityError(FetchError):
    """Downloaded bytes do not match the declared SHA-256."""

    def __init__(self, url: str, expected: str, actual: str) -> None:
        super().__init__(
            f"SHA-256 mismatch for {url!r}: expected {expected[:16]}… got {actual[:16]}…"
        )
        self.url = url
        self.expected = expected
        self.actual = actual


class FetchHTTPError(FetchError):
    """HTTP-level error (non-2xx status or connection failure)."""

    def __init__(self, url: str, detail: str) -> None:
        super().__init__(f"HTTP error fetching {url!r}: {detail}")
        self.url = url
        self.detail = detail


class FetchTooLargeError(FetchError):
    """Response exceeded MAX_DOWNLOAD_BYTES and was aborted mid-stream."""

    def __init__(self, url: str, limit: int) -> None:
        super().__init__(f"Response from {url!r} exceeded {limit // (1024 * 1024)} MB limit")
        self.url = url
        self.limit = limit


# ---------------------------------------------------------------------------
# Core fetch logic
# ---------------------------------------------------------------------------


def _cache_path(spec: SourceSpec, cache_root: Path) -> Path:
    """Compute the deterministic cache path for a SourceSpec."""
    parsed = urllib.parse.urlparse(spec.url)
    basename = Path(parsed.path).name
    if not basename:
        # URL ends with '/' or has no path component — fall back to the hash
        basename = spec.sha256
    return cache_root / spec.sha256[:2] / spec.sha256 / basename


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(
    spec: SourceSpec,
    cache_root: Path = DEFAULT_CACHE_ROOT,
    *,
    timeout_s: float = 60.0,
) -> Path:
    """Fetch `spec.url`, verify SHA-256, and return the cached file path.

    Idempotent: if the cached file exists and its SHA matches `spec.sha256`,
    no network call is made.  If the cached file exists but is corrupt
    (SHA mismatch), it is deleted and re-fetched.

    Raises:
        FetchIntegrityError: downloaded SHA does not match spec.sha256.
        FetchHTTPError: HTTP non-2xx status or connection failure.
        FetchTooLargeError: response body exceeded MAX_DOWNLOAD_BYTES.
    """
    final_path = _cache_path(spec, cache_root)

    # Cache hit: verify and return without hitting the network
    if final_path.exists():
        actual = _sha256_of_file(final_path)
        if actual == spec.sha256:
            return final_path
        # Stale / corrupt cache entry — delete and re-fetch
        final_path.unlink()

    # Prepare destination directory
    final_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream into a temp file on the same filesystem (enables atomic rename)
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=final_path.name + ".", suffix=".tmp", dir=final_path.parent
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(fd, "wb") as tmp_file:
            h = hashlib.sha256()
            total = 0

            try:
                with httpx.Client(
                    timeout=timeout_s,
                    follow_redirects=True,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                ) as client:
                    with client.stream("GET", spec.url) as response:
                        try:
                            response.raise_for_status()
                        except httpx.HTTPStatusError as exc:
                            raise FetchHTTPError(
                                spec.url,
                                f"status {exc.response.status_code}",
                            ) from exc

                        for chunk in response.iter_bytes(chunk_size=64 * 1024):
                            total += len(chunk)
                            if total > MAX_DOWNLOAD_BYTES:
                                raise FetchTooLargeError(spec.url, MAX_DOWNLOAD_BYTES)
                            h.update(chunk)
                            tmp_file.write(chunk)

            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                if not isinstance(exc, FetchHTTPError):
                    raise FetchHTTPError(spec.url, str(exc)) from exc
                raise

        # Verify integrity before committing the cache entry
        actual = h.hexdigest()
        if actual != spec.sha256:
            tmp_path.unlink(missing_ok=True)
            raise FetchIntegrityError(spec.url, spec.sha256, actual)

        # Atomic rename: on the same filesystem, so this is safe
        os.replace(tmp_path, final_path)
        return final_path

    except Exception:
        # Clean up the temp file on any failure
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise
