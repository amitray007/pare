"""Isolated iteration runner — spawn a fresh Python subprocess per iteration.

Public entry point:

    run_iteration_in_worker(case, *, track_python_allocs=False,
                            track_rss_curve=False) -> dict[str, Any]

Each call spawns a brand-new Python interpreter via
``multiprocessing.get_context("spawn")``, runs one ``measure() +
optimize_image()`` iteration inside it, and returns the per-iteration dict
(same shape as ``bench.runner.modes.quick._run_one_case``).

**Why spawn?**  ``os.fork`` on macOS leaks Pillow's plugin registration state and
open file descriptors in unpredictable ways. ``spawn`` is the safe choice and is
identical to Linux's default for ``ProcessPoolExecutor`` since Python 3.12.

**Why sequential?**  Parallel workers would entangle wall-times across cases —
exactly the bug the rest of the bench avoids by running one case at a time.

**Why a Pool(maxtasksperchild=1)?**  Each ``pool.apply`` call with
``maxtasksperchild=1`` guarantees a fresh OS process per call. The pool's
bookkeeping is lighter than constructing a new ``Process`` + ``Queue`` pair
every iteration, but the behaviour is identical: after the first task the
worker is retired and a new one is spawned for the next call.

**Overhead:** Python startup + plugin registration on macOS takes 200–400 ms per
worker. For a 3-case × 2-repeat run that's 6 extra seconds vs the in-process
variant. This is the documented cost of clean-RSS measurement.
"""

from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

from bench.runner.case import Case

# ---------------------------------------------------------------------------
# Worker function — must be top-level (no closures) so pickle works across
# the spawn boundary.
# ---------------------------------------------------------------------------


def _worker_main(
    case_dict: dict[str, Any],
    *,
    track_python_allocs: bool,
    track_rss_curve: bool,
) -> dict[str, Any]:
    """Runs inside the spawned subprocess.

    Imports happen here, so plugin registration (pillow_heif, jxlpy, etc.)
    occurs inside the worker's fresh interpreter — that's the whole point of
    isolation. The ``RUSAGE_SELF`` numbers recorded by ``measure()`` reflect
    only this worker process, not any previous case's allocations.
    """
    import asyncio

    # These imports trigger pillow plugin registration in the fresh worker.
    from bench.runner.case import Case
    from bench.runner.measure import measure
    from bench.runner.modes.quick import measurement_to_dict
    from bench.runner.probe import collect_tool_invocations
    from optimizers.router import optimize_image
    from schemas import OptimizationConfig

    case = Case(
        case_id=case_dict["case_id"],
        name=case_dict["name"],
        bucket=case_dict["bucket"],
        fmt=case_dict["fmt"],
        preset=case_dict["preset"],
        quality=case_dict["quality"],
        file_path=Path(case_dict["file_path"]),
        input_size=case_dict["input_size"],
    )

    input_data = case.load()
    config = OptimizationConfig(quality=case.quality)

    async def _run() -> dict[str, Any]:
        with measure(
            track_python_allocs=track_python_allocs,
            track_rss_curve=track_rss_curve,
        ) as m:
            with collect_tool_invocations() as invocations:
                result = await optimize_image(input_data, config)

        return {
            "case_id": case.case_id,
            "name": case.name,
            "bucket": case.bucket,
            "format": case.fmt,
            "preset": case.preset,
            "input_size": case.input_size,
            # iteration index is injected by the caller
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

    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_iteration_in_worker(
    case: Case,
    *,
    track_python_allocs: bool = False,
    track_rss_curve: bool = False,
) -> dict[str, Any]:
    """Spawn a fresh Python process, run one iteration of ``case``, return its
    measurement dict (same shape as quick mode's per-iteration dict).

    On worker error, returns the standard ``{error: "..."}`` failure shape so
    callers don't need to special-case it.

    Sequential by design; never call this concurrently with itself.
    """
    case_dict: dict[str, Any] = {
        "case_id": case.case_id,
        "name": case.name,
        "bucket": case.bucket,
        "fmt": case.fmt,
        "preset": case.preset,
        "quality": case.quality,
        "file_path": str(case.file_path),
        "input_size": case.input_size,
    }

    ctx = multiprocessing.get_context("spawn")
    # maxtasksperchild=1 guarantees a fresh OS process for every pool.apply call.
    with ctx.Pool(processes=1, maxtasksperchild=1) as pool:
        try:
            result = pool.apply(
                _worker_main,
                args=(case_dict,),
                kwds={
                    "track_python_allocs": track_python_allocs,
                    "track_rss_curve": track_rss_curve,
                },
            )
            return result
        except Exception as e:
            return {
                "case_id": case.case_id,
                "name": case.name,
                "bucket": case.bucket,
                "format": case.fmt,
                "preset": case.preset,
                "error": f"{type(e).__name__}: {e}",
            }
