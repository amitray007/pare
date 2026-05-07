"""Timing mode: multi-iteration latency benchmark with warmup.

For each case, run `warmup` warmup iterations (discarded) followed by
`repeat` measured iterations. Output a flat list where each entry
includes its iteration index — the reporting layer rolls these up into
percentile / median / MAD via `bench.runner.stats.summarize_iterations`.

Cases run sequentially. `--isolate` reruns each iteration in a fresh Python
subprocess so that ``parent_peak_rss_kb`` reflects per-case allocation rather
than the cumulative high-water mark that builds up across cases in the same
process. Without ``--isolate``, ``parent_peak_rss_kb`` is a monotonic
high-water mark across cases — interpret it as "cumulative parent footprint"
and use ``children_peak_rss_kb`` as the per-case headline. See bench/CLAUDE.md.

Overhead of ``--isolate``: Python startup + plugin registration takes
200–400 ms per worker on macOS. For N cases × repeat iterations that's
N × repeat extra seconds compared to the in-process variant.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from bench.runner.case import Case
from bench.runner.modes.quick import _run_one_case

logger = logging.getLogger(__name__)


async def _run_iterations(
    case: Case,
    *,
    warmup: int,
    repeat: int,
    track_python_allocs: bool,
    isolate: bool,
) -> list[dict[str, Any]]:
    """Run `warmup + repeat` total iterations; return only the measured ones."""
    if isolate:
        return await _run_iterations_isolated(
            case,
            warmup=warmup,
            repeat=repeat,
            track_python_allocs=track_python_allocs,
        )

    for _ in range(warmup):
        try:
            await _run_one_case(case, track_python_allocs=False)
        except Exception:
            # If warmup fails, the timed iterations will fail too — record that
            # below rather than aborting here.
            pass

    measured: list[dict[str, Any]] = []
    for i in range(repeat):
        try:
            result = await _run_one_case(case, track_python_allocs=track_python_allocs)
            result["iteration"] = i
            measured.append(result)
        except Exception as e:
            logger.warning("case %s iter %d failed: %s", case.case_id, i, e)
            measured.append(
                {
                    "case_id": case.case_id,
                    "name": case.name,
                    "bucket": case.bucket,
                    "format": case.fmt,
                    "preset": case.preset,
                    "iteration": i,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
    return measured


async def _run_iterations_isolated(
    case: Case,
    *,
    warmup: int,
    repeat: int,
    track_python_allocs: bool,
) -> list[dict[str, Any]]:
    """Like ``_run_iterations`` but each call goes to a fresh subprocess.

    Warmup iterations also use fresh workers (clean cold-start is part of the
    headline). Workers are spawned sequentially — parallel workers would
    entangle wall-times, defeating the purpose.
    """
    # Import here to avoid circular imports at module level; isolate.py imports
    # Case, and timing.py imports Case as well, but they don't form a cycle.
    from bench.runner.isolate import run_iteration_in_worker

    for _ in range(warmup):
        # Warmup result discarded; still spawn fresh so caches are primed.
        run_iteration_in_worker(case, track_python_allocs=False)

    measured: list[dict[str, Any]] = []
    for i in range(repeat):
        result = run_iteration_in_worker(
            case,
            track_python_allocs=track_python_allocs,
        )
        result["iteration"] = i
        if "error" in result:
            logger.warning("case %s iter %d failed: %s", case.case_id, i, result["error"])
        measured.append(result)
    return measured


async def run_timing(
    cases: list[Case],
    *,
    warmup: int = 1,
    repeat: int = 5,
    seed: int = 42,
    shuffle: bool = True,
    track_python_allocs: bool = False,
    isolate: bool = False,
) -> list[dict[str, Any]]:
    """Run all cases sequentially with warmup + repeat per case.

    Order shuffling defends against systematic first-case-is-cold bias
    (PIL plugin lazy-load, OS page cache populating). Set ``shuffle=False``
    for reproducible debugging.

    When ``isolate=True`` each (case × iteration) tuple runs in a fresh
    Python subprocess so ``parent_peak_rss_kb`` is clean per-case rather
    than cumulative. Each spawn adds ~200–400 ms cold-start overhead.
    """
    ordered: list[Case] = list(cases)
    if shuffle:
        random.Random(seed).shuffle(ordered)

    results: list[dict[str, Any]] = []
    for case in ordered:
        iter_results = await _run_iterations(
            case,
            warmup=warmup,
            repeat=repeat,
            track_python_allocs=track_python_allocs,
            isolate=isolate,
        )
        results.extend(iter_results)
    return results


def run_timing_sync(
    cases: list[Case],
    *,
    warmup: int = 1,
    repeat: int = 5,
    seed: int = 42,
    shuffle: bool = True,
    track_python_allocs: bool = False,
    isolate: bool = False,
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_timing(
            cases,
            warmup=warmup,
            repeat=repeat,
            seed=seed,
            shuffle=shuffle,
            track_python_allocs=track_python_allocs,
            isolate=isolate,
        )
    )
