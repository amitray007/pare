"""Fitted BPP model loader.

Exposes ``load_png_model() -> Loaded | LoadFailed``, backed by ``functools.lru_cache(maxsize=1)``
so the JSON artifact is parsed exactly once per process.

CPython issue #103475 caveat
----------------------------
``lru_cache(maxsize=1)`` is thread-safe at the dict level but offers **no call-once guarantee**
under truly concurrent first-request bursts — the loader may execute 2+ times before the cache
entry is written.  This is acceptable here because the JSON is small (<50 KB) and the result is
deterministic (identical JSON → identical ``Loaded`` value).  For strict call-once semantics, wrap
the body in ``threading.Lock``.

``lru_cache`` does **not** cache exceptions (CPython issue #103475).  A loader that raises would
re-run on every call, defeating the cache.  This module avoids that by always returning a value
(``Loaded`` or ``LoadFailed``), never raising.

Multi-process invariant
-----------------------
Each uvicorn worker has its own ``lru_cache`` entry.  Artifact updates require a Cloud Run
revision deploy — in-place mutation of the JSON file on a running instance is ineffective.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path

from estimation.models._artifact import (
    JpegHeaderModel,
    Loaded,
    LoadedHeader,
    LoadedJpeg,
    LoadFailed,
    PngHeaderModel,
    PngModel,
)

logger = logging.getLogger("pare.estimation.models")

# Co-locate artifacts next to the consumer module.
_MODELS_DIR = Path(__file__).parent


@functools.lru_cache(maxsize=1)
def load_png_model() -> Loaded | LoadFailed:
    """Load the PNG fitted-BPP model artifact.

    Returns ``Loaded(model)`` on success or ``LoadFailed(reason)`` on any failure.
    Never raises — all exceptions are converted to ``LoadFailed("other")`` (unexpected)
    or the more specific reason codes handled inside ``PngModel.from_json()``.

    The result is cached for the lifetime of the process (``lru_cache(maxsize=1)``).
    See module docstring for the CPython call-once caveat.
    """
    artifact_path = _MODELS_DIR / "png_v1.json"
    try:
        return PngModel.from_json(artifact_path)
    except Exception as exc:  # defensive — from_json should never raise, but belt-and-suspenders
        logger.warning("load_png_model: unexpected exception: %s", exc)
        return LoadFailed(reason="other")


@functools.lru_cache(maxsize=1)
def load_png_header_model() -> LoadedHeader | LoadFailed:
    """Load the PNG header-only fitted-BPP model artifact.

    Returns ``LoadedHeader(model)`` on success or ``LoadFailed(reason)`` on any failure.
    Never raises — all exceptions are converted to ``LoadFailed("other")`` (unexpected)
    or the more specific reason codes handled inside ``PngHeaderModel.from_json()``.

    Loaded but unused by any caller in Phase 1a — dispatch wiring is Phase 2.

    The result is cached for the lifetime of the process (``lru_cache(maxsize=1)``).
    See module docstring for the CPython call-once caveat.
    """
    artifact_path = _MODELS_DIR / "png_header_v1.json"
    try:
        return PngHeaderModel.from_json(artifact_path)
    except Exception as exc:  # defensive — from_json should never raise, but belt-and-suspenders
        logger.warning("load_png_header_model: unexpected exception: %s", exc)
        return LoadFailed(reason="other")


@functools.lru_cache(maxsize=1)
def load_jpeg_header_model() -> LoadedJpeg | LoadFailed:
    """Load the JPEG header-only fitted-BPP model artifact.

    Returns ``LoadedJpeg(model)`` on success or ``LoadFailed(reason)`` on any failure.
    Never raises — all exceptions are converted to ``LoadFailed("other")`` (unexpected)
    or the more specific reason codes handled inside ``JpegHeaderModel.from_json()``.

    Loaded but unused by any caller in Phase 1b — dispatch wiring is Phase 2.

    The result is cached for the lifetime of the process (``lru_cache(maxsize=1)``).
    See module docstring for the CPython call-once caveat.
    """
    artifact_path = _MODELS_DIR / "jpeg_header_v1.json"
    try:
        return JpegHeaderModel.from_json(artifact_path)
    except Exception as exc:  # defensive — from_json should never raise, but belt-and-suspenders
        logger.warning("load_jpeg_header_model: unexpected exception: %s", exc)
        return LoadFailed(reason="other")


__all__ = [
    "Loaded",
    "LoadedHeader",
    "LoadedJpeg",
    "LoadFailed",
    "JpegHeaderModel",
    "PngHeaderModel",
    "PngModel",
    "load_jpeg_header_model",
    "load_png_header_model",
    "load_png_model",
]
