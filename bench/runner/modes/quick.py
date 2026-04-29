"""Quick mode: 1 iteration per case, sequential, no warmup.

Used for PR sanity checks. Output schema matches `timing` so reports
can ingest both interchangeably.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bench.runner.case import Case
from bench.runner.measure import Measurement, measure
from bench.runner.probe import collect_tool_invocations
from optimizers.router import optimize_image
from schemas import OptimizationConfig

logger = logging.getLogger(__name__)


def measurement_to_dict(m: Measurement) -> dict[str, Any]:
    return {
        "wall_ms": m.wall_ms,
        "parent_user_ms": m.parent_user_ms,
        "parent_sys_ms": m.parent_sys_ms,
        "children_user_ms": m.children_user_ms,
        "children_sys_ms": m.children_sys_ms,
        "total_cpu_ms": m.total_cpu_ms,
        "parallelism": m.parallelism,
        "parent_peak_rss_kb": m.parent_peak_rss_kb,
        "children_peak_rss_kb": m.children_peak_rss_kb,
        # Capacity-planning headline: max(parent, children) peak RSS.
        # This is what determines the Cloud Run instance size needed.
        "peak_rss_kb": m.peak_rss_kb,
        "py_peak_alloc_kb": m.py_peak_alloc_kb,
        "phases": dict(m.phases),
    }


async def _run_one_case(case: Case, *, track_python_allocs: bool = False) -> dict[str, Any]:
    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    with measure(track_python_allocs=track_python_allocs) as m:
        with collect_tool_invocations() as invocations:
            result = await optimize_image(input_data, config)

    return {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "format": case.fmt,
        "preset": case.preset,
        "input_size": case.input_size,
        "iteration": 0,
        "measurement": measurement_to_dict(m),
        "tool_invocations": [
            {"tool": inv.tool, "wall_ms": inv.wall_ms, "exit_code": inv.exit_code}
            for inv in invocations
        ],
        "reduction_pct": result.reduction_percent,
        "method": result.method,
        "optimized_size": result.optimized_size,
    }


async def run_quick(
    cases: list[Case],
    *,
    track_python_allocs: bool = False,
) -> list[dict[str, Any]]:
    """Sequentially run one iteration per case.

    Sequential by design: `asyncio.gather` would entangle wall-times
    across cases (the legacy bench's bug). One case at a time gives
    each case the whole machine for its window.
    """
    results: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = await _run_one_case(case, track_python_allocs=track_python_allocs)
            results.append(result)
        except Exception as e:
            logger.warning("case %s failed: %s", case.case_id, e)
            results.append(
                {
                    "case_id": case.case_id,
                    "name": case.name,
                    "bucket": case.bucket,
                    "format": case.fmt,
                    "preset": case.preset,
                    "iteration": 0,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return results


def run_quick_sync(
    cases: list[Case],
    *,
    track_python_allocs: bool = False,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for use from `bench.runner.cli`."""
    return asyncio.run(run_quick(cases, track_python_allocs=track_python_allocs))
