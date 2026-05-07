"""Visual quality samples generator for the Pare dashboard.

Produces a worst-case sample per lossy format from a pr-mode bench run JSON.
Each record carries SSIM/PSNR, compression stats, and—when the corpus is
available—base64-encoded before/after PNG thumbnails (max 300×300, optimize=True).

Lossless formats (PNG, GIF, APNG, BMP, TIFF, SVG, SVGZ) return records with
``lossless=True`` and no thumbnail fields.

If the corpus files are missing or optimizer errors occur the record is still
returned, just without ``orig_thumb_b64``/``opt_thumb_b64``.  The page
renderer handles both shapes gracefully.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Formats considered lossy — we generate thumbnails for these.
_LOSSY_FORMATS: frozenset[str] = frozenset({"jpeg", "webp", "avif", "heic", "jxl"})
# All lossless formats supported by the corpus.
_LOSSLESS_FORMATS: frozenset[str] = frozenset({"png", "apng", "gif", "bmp", "tiff", "svg", "svgz"})

# Thumbnail max dimension (pixels).
_THUMB_MAX_PX = 300


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64_thumbnail(image_bytes: bytes) -> str:
    """Decode *image_bytes* as an image, resize to fit 300x300, return base64 PNG."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img.thumbnail((_THUMB_MAX_PX, _THUMB_MAX_PX), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _find_worst_ssim_per_format(
    iterations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return {fmt: worst_iteration} for each lossy format.

    Deduplicates by case_id (pr-mode duplicates iterations for timing).
    Only considers iterations that have a valid ``quality.ssim`` value.
    """
    seen: set[str] = set()
    worst: dict[str, dict[str, Any]] = {}

    for it in iterations:
        if it.get("error") is not None:
            continue
        cid = it.get("case_id", "")
        if cid in seen:
            continue
        seen.add(cid)

        fmt = it.get("format", "")
        if fmt not in _LOSSY_FORMATS:
            continue

        quality = it.get("quality")
        if not isinstance(quality, dict):
            continue
        ssim_val = quality.get("ssim")
        if ssim_val is None:
            continue

        ssim_f = float(ssim_val)
        if fmt not in worst or ssim_f < float(worst[fmt]["quality"]["ssim"]):
            worst[fmt] = it

    return worst


def _corpus_file_path(corpus_root: Path, case_id: str) -> Path | None:
    """Reconstruct the on-disk corpus path from a case_id like ``name.fmt@preset``.

    Layout from ``bench.corpus.builder.file_path``:
        <root>/<bucket>/<format>/<name>.<ext>

    We reconstruct by walking the corpus root looking for the file that matches
    the name+format extracted from *case_id*.  Returns None if not found.
    """
    # case_id format: "<name>.<fmt>@<preset>"
    # name itself may contain dots (e.g. "photo_perlin_xlarge_jpeg.jpeg")
    # The "@<preset>" suffix is always at the end.
    if "@" not in case_id:
        return None
    base, _preset = case_id.rsplit("@", 1)
    # The format is the last dot-separated token of base.
    if "." not in base:
        return None
    name_part, fmt = base.rsplit(".", 1)
    ext = "apng" if fmt == "apng" else fmt

    # Search corpus_root/<bucket>/<fmt>/<name>.<ext>
    pattern = f"*/{fmt}/{name_part}.{ext}"
    matches = list(corpus_root.glob(pattern))
    if not matches:
        return None
    return matches[0]


# ---------------------------------------------------------------------------
# Async optimizer wrapper
# ---------------------------------------------------------------------------


async def _optimize_bytes(input_data: bytes, quality: int) -> bytes | None:
    """Run optimize_image() on *input_data*; return optimized bytes or None."""
    try:
        from optimizers.router import optimize_image
        from schemas import OptimizationConfig

        config = OptimizationConfig(quality=quality)
        result = await optimize_image(input_data, config)
        return result.optimized_bytes
    except Exception as exc:
        logger.warning("samples: optimizer error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Per-format record builder
# ---------------------------------------------------------------------------


def _build_lossy_record(
    fmt: str,
    it: dict[str, Any],
    corpus_root: Path | None,
) -> dict[str, Any]:
    """Build a sample record for a lossy format.

    Loads the original file bytes from the corpus (if available), re-optimizes,
    and generates thumbnails.  Any failure is caught and logged; the record is
    returned without thumbnails rather than raising.
    """
    from bench.runner.report.thresholds import SSIM_DEFAULT, SSIM_THRESHOLDS

    quality_block = it.get("quality", {}) or {}
    ssim_val = quality_block.get("ssim")
    psnr_val = quality_block.get("psnr_db")
    preset = it.get("preset", "medium")
    case_id = it.get("case_id", "")

    # Size info from the optimize block written by pr mode.
    opt_block = it.get("optimize", {}) or {}
    size_orig = it.get("input_size", 0)
    size_opt = opt_block.get("actual_size", 0)
    reduction_pct = opt_block.get("actual_reduction_pct") or it.get("reduction_pct", 0.0)

    threshold = SSIM_THRESHOLDS.get(preset, SSIM_DEFAULT)
    if ssim_val is not None:
        ssim_f = float(ssim_val)
        if ssim_f >= threshold:
            status = "ok"
        elif ssim_f >= threshold * 0.98:
            status = "warn"
        else:
            status = "fail"
    else:
        status = "fail"

    record: dict[str, Any] = {
        "format": fmt,
        "case_id": case_id,
        "preset": preset,
        "quality": round(float(ssim_val), 4) if ssim_val is not None else None,
        "ssim": round(float(ssim_val), 4) if ssim_val is not None else None,
        "psnr_db": round(float(psnr_val), 2) if psnr_val is not None else None,
        "ssim_threshold": threshold,
        "status": status,
        "size_orig_kb": round(size_orig / 1024, 1) if size_orig else 0,
        "size_opt_kb": round(size_opt / 1024, 1) if size_opt else 0,
        "reduction_pct": round(float(reduction_pct), 1) if reduction_pct else 0.0,
        "lossless": False,
    }

    # -- Thumbnail generation (best-effort) --
    if corpus_root is None:
        logger.debug("samples: no corpus_root — skipping thumbnails for %s", fmt)
        return record

    corpus_file = _corpus_file_path(corpus_root, case_id)
    if corpus_file is None:
        logger.warning("samples: corpus file not found for case_id=%r (fmt=%s)", case_id, fmt)
        return record

    try:
        input_data = corpus_file.read_bytes()
    except OSError as exc:
        logger.warning("samples: could not read corpus file %s: %s", corpus_file, exc)
        return record

    # Run the optimizer synchronously (we're in a sync caller; wrap in asyncio.run).
    # Use the preset quality mapping.
    from bench.runner.case import PRESET_QUALITY

    quality_int = PRESET_QUALITY.get(preset, 60)

    try:
        opt_data = asyncio.run(_optimize_bytes(input_data, quality_int))
    except RuntimeError:
        # If there's already an event loop running (e.g. in tests), use get_event_loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            opt_data = loop.run_until_complete(_optimize_bytes(input_data, quality_int))
        except Exception as exc2:
            logger.warning("samples: async optimizer fallback failed for %s: %s", fmt, exc2)
            opt_data = None

    if opt_data is None:
        logger.warning("samples: optimizer returned None for %s — skipping thumbnails", fmt)
        return record

    try:
        record["orig_thumb_b64"] = _b64_thumbnail(input_data)
        record["opt_thumb_b64"] = _b64_thumbnail(opt_data)
    except Exception as exc:
        logger.warning("samples: thumbnail generation failed for %s: %s", fmt, exc)
        # Remove any partial keys
        record.pop("orig_thumb_b64", None)
        record.pop("opt_thumb_b64", None)

    return record


def _build_lossless_record(fmt: str) -> dict[str, Any]:
    return {
        "format": fmt,
        "case_id": None,
        "preset": None,
        "ssim": None,
        "psnr_db": None,
        "ssim_threshold": None,
        "status": "ok",
        "size_orig_kb": 0,
        "size_opt_kb": 0,
        "reduction_pct": 0.0,
        "lossless": True,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_sample_records(
    run_data: dict[str, Any],
    corpus_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build one record per format (lossy and lossless).

    Parameters
    ----------
    run_data:
        A parsed pr-mode bench run JSON (schema_version=2).
    corpus_root:
        Path to the on-disk corpus directory (e.g. ``bench/corpus/data``).
        If None, thumbnails are skipped for all lossy formats — the record
        still contains SSIM/PSNR/size data.

    Returns
    -------
    list[dict]
        One record per format, lossy formats first sorted by status (fail,
        warn, ok), then alpha within group, followed by lossless formats alpha.
    """
    iterations: list[dict[str, Any]] = run_data.get("iterations", [])
    worst_per_fmt = _find_worst_ssim_per_format(iterations)

    records: list[dict[str, Any]] = []

    # --- Lossy formats ---
    for fmt in sorted(_LOSSY_FORMATS):
        if fmt in worst_per_fmt:
            try:
                rec = _build_lossy_record(fmt, worst_per_fmt[fmt], corpus_root)
            except Exception as exc:
                logger.warning("samples: unexpected error building record for %s: %s", fmt, exc)
                # Still include a minimal record so the page renders.
                rec = {
                    "format": fmt,
                    "case_id": worst_per_fmt[fmt].get("case_id", ""),
                    "preset": worst_per_fmt[fmt].get("preset", ""),
                    "ssim": None,
                    "psnr_db": None,
                    "ssim_threshold": None,
                    "status": "fail",
                    "size_orig_kb": 0,
                    "size_opt_kb": 0,
                    "reduction_pct": 0.0,
                    "lossless": False,
                }
        else:
            # No pr-mode quality data for this format (quick-mode run, or format
            # not in corpus).  Produce a minimal stub so the page mentions it.
            rec = {
                "format": fmt,
                "case_id": None,
                "preset": None,
                "ssim": None,
                "psnr_db": None,
                "ssim_threshold": None,
                "status": "ok",
                "size_orig_kb": 0,
                "size_opt_kb": 0,
                "reduction_pct": 0.0,
                "lossless": False,
                "no_data": True,
            }
        records.append(rec)

    # Sort lossy records: fail first, then warn, then ok/no_data, alpha within group.
    _order = {"fail": 0, "warn": 1, "ok": 2}
    records.sort(key=lambda r: (_order.get(r["status"], 3), r["format"]))

    # --- Lossless formats ---
    lossless_records = [_build_lossless_record(fmt) for fmt in sorted(_LOSSLESS_FORMATS)]
    records.extend(lossless_records)

    return records
