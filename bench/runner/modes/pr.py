"""PR mode: timing + accuracy + quality in a single pass per case.

Designed as the default mode for manual and automated PR benchmarking.
Each case produces one record with three nested blocks:

  * ``measurement``  — timing stats from N repeated optimize_image() calls
  * ``accuracy``     — estimator prediction vs actual (one call, not N)
  * ``quality``      — SSIM/PSNR for lossy formats; None for lossless

Why a single pass?

  * Avoids running optimize_image() three separate times for the same input.
  * Produces one self-contained JSON document that answers speed, accuracy,
    and quality questions without reconciling separate report files.
  * The PR scorecard can show all three signals on one page.

Pass ordering per case:
  1. **Accuracy step (once)**: run estimate() then optimize_image().
     Capture predicted/actual size and reduction; derive error metrics.
  2. **Quality step (once, lossy only)**: decode input and optimized bytes
     from the accuracy step; compute SSIM/PSNR (and ssimulacra2/butteraugli
     if available). No re-optimization.
  3. **Timing step (N iterations)**: re-run optimize_image() with measure()
     for wall_ms and CPU attribution. Warmup iterations are discarded.

The output schema for each case is a superset of the quick-mode per-case
dict so that existing stats roll-up (json_writer._roll_up_stats) and the
timing reporters work without changes.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from bench.runner.case import Case
from bench.runner.measure import Measurement, measure
from bench.runner.modes.accuracy import _compute_error
from bench.runner.modes.quick import _run_one_case, measurement_to_dict
from bench.runner.probe import collect_tool_invocations
from bench.runner.quality import butteraugli_scores, psnr_db, ssim, ssimulacra2_score
from estimation.estimator import estimate
from optimizers.router import optimize_image
from schemas import OptimizationConfig

logger = logging.getLogger(__name__)

# Formats that are always lossless — quality scoring skipped for these.
_LOSSLESS_FORMATS: frozenset[str] = frozenset({"png", "apng", "gif", "bmp", "tiff", "svg", "svgz"})
_LOSSY_FORMATS: frozenset[str] = frozenset({"jpeg", "webp", "avif", "heic", "jxl"})


def _decode_to_array(data: bytes) -> np.ndarray:
    """Decode image bytes to a float32 numpy array in [0, 1], RGB."""
    import io

    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


async def _quality_block_for(
    input_data: bytes,
    optimized_bytes: bytes,
    case: Case,
    *,
    fast: bool = False,
) -> dict[str, Any] | None:
    """Compute quality metrics for a lossy case using already-optimized bytes.

    Returns None for lossless formats. Returns a dict with quality metrics
    (matching the quality-mode schema) for lossy formats.
    """
    if case.fmt not in _LOSSY_FORMATS:
        return None

    q_start = time.perf_counter()

    try:
        ref_arr = await asyncio.to_thread(_decode_to_array, input_data)
        dist_arr = await asyncio.to_thread(_decode_to_array, optimized_bytes)
    except Exception as exc:
        logger.warning("case %s decode failed for quality scoring: %s", case.case_id, exc)
        return {
            "ssim": None,
            "psnr_db": None,
            "ssimulacra2": None,
            "butteraugli_max": None,
            "butteraugli_3norm": None,
            "wall_ms": 0.0,
            "decode_error": f"{type(exc).__name__}: {exc}",
        }

    shapes_match = ref_arr.shape == dist_arr.shape
    if not shapes_match:
        logger.warning(
            "case %s: ref shape %s != dist shape %s; null quality metrics",
            case.case_id,
            ref_arr.shape,
            dist_arr.shape,
        )

    ssim_val: float | None = None
    psnr_val: float | None = None
    perfect_match = False

    if shapes_match:
        ssim_val = ssim(ref_arr, dist_arr)
        psnr_raw = psnr_db(ref_arr, dist_arr)
        if psnr_raw is None:
            perfect_match = True
        else:
            psnr_val = psnr_raw

    ss2_val: float | None = None
    ba_max: float | None = None
    ba_norm: float | None = None

    if not fast and shapes_match:
        with tempfile.TemporaryDirectory(prefix="pare_pr_quality_") as tmpdir:
            tmp = Path(tmpdir)
            ref_png = tmp / "ref.png"
            dist_png = tmp / "dist.png"
            try:
                ref_img = Image.fromarray((ref_arr * 255).astype(np.uint8), mode="RGB")
                dist_img = Image.fromarray((dist_arr * 255).astype(np.uint8), mode="RGB")
                ref_img.save(str(ref_png), format="PNG", compress_level=1)
                dist_img.save(str(dist_png), format="PNG", compress_level=1)
                ss2_val = ssimulacra2_score(ref_png, dist_png)
                ba_max, ba_norm = butteraugli_scores(ref_png, dist_png)
            except Exception as exc:
                logger.warning("case %s quality scoring error: %s", case.case_id, exc)

    q_wall_ms = (time.perf_counter() - q_start) * 1000.0

    block: dict[str, Any] = {
        "ssim": round(ssim_val, 6) if ssim_val is not None else None,
        "psnr_db": round(psnr_val, 3) if psnr_val is not None else None,
        "ssimulacra2": round(ss2_val, 3) if ss2_val is not None else None,
        "butteraugli_max": round(ba_max, 4) if ba_max is not None else None,
        "butteraugli_3norm": round(ba_norm, 4) if ba_norm is not None else None,
        "wall_ms": round(q_wall_ms, 1),
    }
    if perfect_match:
        block["perfect_match"] = True
    return block


async def _run_one_pr_case(
    case: Case,
    *,
    warmup: int,
    repeat: int,
    fast_quality: bool = False,
) -> list[dict[str, Any]]:
    """Run the full PR pipeline for one case.

    Returns a list of per-iteration dicts (one per timing iteration).
    The accuracy and quality blocks appear on every iteration row with
    the same values (computed once), so that json_writer._roll_up_stats
    still works correctly (it reads only ``measurement`` fields for rollup).
    """
    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    base: dict[str, Any] = {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "format": case.fmt,
        "preset": case.preset,
        "input_size": case.input_size,
    }

    # -----------------------------------------------------------------------
    # Step 1: Accuracy — run estimate() + optimize_image() once
    # -----------------------------------------------------------------------
    est_m = Measurement()
    try:
        with measure() as est_m:
            est_result = await estimate(input_data, config)
    except Exception as exc:
        logger.warning("case %s estimate failed: %s", case.case_id, exc)
        return [
            {
                **base,
                "iteration": 0,
                "error": {
                    "phase": "estimate",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            }
        ]

    estimate_block: dict[str, Any] = {
        "wall_ms": est_m.wall_ms,
        "measurement": measurement_to_dict(est_m),
        "predicted_size": est_result.estimated_optimized_size,
        "predicted_reduction_pct": est_result.estimated_reduction_percent,
        "method": est_result.method,
        "confidence": est_result.confidence,
        "already_optimized": est_result.already_optimized,
    }

    opt_m = Measurement()
    try:
        with measure() as opt_m:
            with collect_tool_invocations() as invocations:
                opt_result = await optimize_image(input_data, config)
    except Exception as exc:
        logger.warning("case %s optimize (accuracy) failed: %s", case.case_id, exc)
        return [
            {
                **base,
                "iteration": 0,
                "estimate": estimate_block,
                "error": {
                    "phase": "optimize",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            }
        ]

    optimize_block: dict[str, Any] = {
        "wall_ms": opt_m.wall_ms,
        "measurement": measurement_to_dict(opt_m),
        "tool_invocations": [
            {"tool": inv.tool, "wall_ms": inv.wall_ms, "exit_code": inv.exit_code}
            for inv in invocations
        ],
        "actual_size": opt_result.optimized_size,
        "actual_reduction_pct": opt_result.reduction_percent,
        "method": opt_result.method,
    }

    accuracy_block = _compute_error(
        predicted_size=est_result.estimated_optimized_size,
        actual_size=opt_result.optimized_size,
        predicted_reduction_pct=est_result.estimated_reduction_percent,
        actual_reduction_pct=opt_result.reduction_percent,
    )

    # -----------------------------------------------------------------------
    # Step 2: Quality — decode + score (uses opt_result.optimized_bytes)
    # -----------------------------------------------------------------------
    quality_block = await _quality_block_for(
        input_data,
        opt_result.optimized_bytes,
        case,
        fast=fast_quality,
    )

    # -----------------------------------------------------------------------
    # Step 3: Timing — warmup + repeat iterations
    # -----------------------------------------------------------------------
    for _ in range(warmup):
        try:
            await optimize_image(input_data, config)
        except Exception:
            pass

    timing_iterations: list[dict[str, Any]] = []
    for i in range(repeat):
        try:
            iter_result = await _run_one_case(case)
            iter_result["iteration"] = i
        except Exception as exc:
            logger.warning("case %s timing iter %d failed: %s", case.case_id, i, exc)
            iter_result = {
                **base,
                "iteration": i,
                "error": f"{type(exc).__name__}: {exc}",
            }
        timing_iterations.append(iter_result)

    # Attach accuracy and quality to every iteration row so that the JSON
    # schema is self-contained and per-case detail can be rendered from a
    # single row. The stats roll-up (json_writer._roll_up_stats) only reads
    # ``measurement`` fields, so the extra blocks don't interfere.
    results: list[dict[str, Any]] = []
    for it in timing_iterations:
        row = dict(it)
        row["estimate"] = estimate_block
        row["optimize"] = optimize_block
        row["accuracy"] = accuracy_block
        row["quality"] = quality_block
        # Mirror accuracy-mode top-level shims for backwards-compatible rollup
        row.setdefault("reduction_pct", opt_result.reduction_percent)
        row.setdefault("method", opt_result.method)
        row.setdefault("optimized_size", opt_result.optimized_size)
        results.append(row)

    return results


async def run_pr(
    cases: list[Case],
    *,
    warmup: int = 1,
    repeat: int = 3,
    fast_quality: bool = False,
) -> list[dict[str, Any]]:
    """Run the full PR pipeline for all cases, sequentially.

    Sequential by design: clean wall-time isolation per case.
    Returns a flat list of per-iteration dicts (repeat dicts per case).
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            case_results = await _run_one_pr_case(
                case,
                warmup=warmup,
                repeat=repeat,
                fast_quality=fast_quality,
            )
            results.extend(case_results)
        except Exception as exc:
            logger.warning("case %s unexpected failure: %s", case.case_id, exc)
            results.append(
                {
                    "case_id": case.case_id,
                    "name": case.name,
                    "bucket": case.bucket,
                    "format": case.fmt,
                    "preset": case.preset,
                    "iteration": 0,
                    "error": {
                        "phase": "load",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                }
            )
    return results


def run_pr_sync(
    cases: list[Case],
    *,
    warmup: int = 1,
    repeat: int = 3,
    fast_quality: bool = False,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for use from ``bench.runner.cli``."""
    return asyncio.run(
        run_pr(
            cases,
            warmup=warmup,
            repeat=repeat,
            fast_quality=fast_quality,
        )
    )
