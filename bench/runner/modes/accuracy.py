"""Accuracy mode: measure estimator prediction vs actual optimization.

For every case in the corpus, runs both:
1. ``estimate(data, config)`` — sample-based prediction (~50–500 ms).
2. ``optimize_image(data, config)`` — real optimization (~100 ms – seconds).

Records predicted-vs-actual error metrics so estimator accuracy regressions
can be caught by the bench. Sequential, one case at a time, matching quick
mode's rationale: clean wall-time isolation per case.

Output schema is a superset of quick mode's per-case dict so existing
reporters can ingest both (the top-level ``measurement`` key comes from
the optimize side for backwards-compatible stats roll-up).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bench.runner.case import Case
from bench.runner.measure import Measurement, measure
from bench.runner.modes.quick import measurement_to_dict
from bench.runner.probe import collect_tool_invocations
from estimation.estimator import estimate
from optimizers.router import optimize_image
from schemas import OptimizationConfig

logger = logging.getLogger(__name__)


def _compute_error(
    predicted_size: int,
    actual_size: int,
    predicted_reduction_pct: float,
    actual_reduction_pct: float,
) -> dict[str, Any]:
    """Compute signed and absolute error metrics."""
    size_abs_error_bytes = predicted_size - actual_size
    if actual_size > 0:
        size_rel_error_pct = round(100.0 * size_abs_error_bytes / actual_size, 3)
    else:
        size_rel_error_pct = 0.0
    reduction_abs_error_pct = round(predicted_reduction_pct - actual_reduction_pct, 3)
    return {
        "size_abs_error_bytes": size_abs_error_bytes,
        "size_rel_error_pct": size_rel_error_pct,
        "reduction_abs_error_pct": reduction_abs_error_pct,
        "reduction_abs_error_pct_abs": round(abs(reduction_abs_error_pct), 3),
    }


async def _run_one_accuracy_case(case: Case) -> dict[str, Any]:
    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    base = {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "format": case.fmt,
        "preset": case.preset,
        "input_size": case.input_size,
        "iteration": 0,
    }

    # --- Estimate phase ---
    est_m = Measurement()
    try:
        with measure() as est_m:
            est_result = await estimate(input_data, config)
    except Exception as exc:
        logger.warning("case %s estimate failed: %s", case.case_id, exc)
        return {
            **base,
            "error": {
                "phase": "estimate",
                "message": f"{type(exc).__name__}: {exc}",
            },
        }

    estimate_block: dict[str, Any] = {
        "wall_ms": est_m.wall_ms,
        "measurement": measurement_to_dict(est_m),
        "predicted_size": est_result.estimated_optimized_size,
        "predicted_reduction_pct": est_result.estimated_reduction_percent,
        "method": est_result.method,
        "confidence": est_result.confidence,
        "already_optimized": est_result.already_optimized,
        "path": est_result.path,
    }

    # --- Optimize phase ---
    opt_m = Measurement()
    try:
        with measure() as opt_m:
            with collect_tool_invocations() as invocations:
                opt_result = await optimize_image(input_data, config)
    except Exception as exc:
        logger.warning("case %s optimize failed: %s", case.case_id, exc)
        return {
            **base,
            "estimate": estimate_block,
            "error": {
                "phase": "optimize",
                "message": f"{type(exc).__name__}: {exc}",
            },
        }

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

    # Expose top-level fields that the existing stats roll-up and reporters
    # expect (measurement, reduction_pct, method, optimized_size).  These
    # come from the optimize side so "stats" in the JSON report reflects
    # actual compression performance.
    return {
        **base,
        # Top-level shims for backwards-compatible stats roll-up
        "measurement": measurement_to_dict(opt_m),
        "tool_invocations": optimize_block["tool_invocations"],
        "reduction_pct": opt_result.reduction_percent,
        "method": opt_result.method,
        "optimized_size": opt_result.optimized_size,
        # Accuracy-specific nested blocks. ``accuracy`` holds prediction-vs-
        # actual error metrics on success; ``error`` is only present when
        # estimate or optimize raised, mirroring quick mode's failure shape.
        "estimate": estimate_block,
        "optimize": optimize_block,
        "accuracy": accuracy_block,
    }


async def run_accuracy(cases: list[Case]) -> list[dict[str, Any]]:
    """Sequentially run one estimate + optimize per case.

    Sequential by design: clean wall-time isolation per case, matching
    quick mode's rationale. Returns one dict per case; failures have
    ``error.phase`` set instead of the normal per-stage data.
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = await _run_one_accuracy_case(case)
            results.append(result)
        except Exception as exc:
            # Outer guard: should not normally fire since _run_one_accuracy_case
            # catches its own exceptions, but keeps the runner alive under any
            # unexpected error (e.g. case.load() fails).
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


def run_accuracy_sync(cases: list[Case]) -> list[dict[str, Any]]:
    """Synchronous wrapper for use from ``bench.runner.cli``."""
    return asyncio.run(run_accuracy(cases))
