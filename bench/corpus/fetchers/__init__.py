"""Corpus fetchers — download and cache real-world images by hash-pinned URL."""

from bench.corpus.fetchers.http import (
    DEFAULT_CACHE_ROOT,
    MAX_DOWNLOAD_BYTES,
    FetchError,
    FetchHTTPError,
    FetchIntegrityError,
    FetchTooLargeError,
    fetch,
)
from bench.corpus.manifest import SourceSpec

__all__ = [
    "DEFAULT_CACHE_ROOT",
    "MAX_DOWNLOAD_BYTES",
    "FetchError",
    "FetchHTTPError",
    "FetchIntegrityError",
    "FetchTooLargeError",
    "SourceSpec",
    "fetch",
]
