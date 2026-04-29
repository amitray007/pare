"""Timing mode: multi-iteration latency benchmark with warmup.

For each case, run `warmup` warmup iterations (discarded) followed by
`repeat` measured iterations. Output a flat list where each entry
includes its iteration index — the reporting layer rolls these up into
percentile / median / MAD via `bench.runner.stats.summarize_iterations`.

Cases run sequentially. `--isolate` (re-exec each case in a fresh Python
subprocess) is intentionally deferred to v1: it doubles runtime for
modest variance reduction, and on Pare's typical workloads (CPU-bound
subprocesses dominating wall time), the heap-state contamination
addressed by isolation is small. Without isolate, `parent_peak_rss_kb`
is a monotonic high-water mark across cases — interpret it as
"cumulative parent footprint" and use `children_peak_rss_kb` as the
per-case headline. See bench/CLAUDE.md.
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
) -> list[dict[str, Any]]:
    """Run `warmup + repeat` total iterations; return only the measured ones."""
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


async def run_timing(
    cases: list[Case],
    *,
    warmup: int = 1,
    repeat: int = 5,
    seed: int = 42,
    shuffle: bool = True,
    track_python_allocs: bool = False,
) -> list[dict[str, Any]]:
    """Run all cases sequentially with warmup + repeat per case.

    Order shuffling defends against systematic first-case-is-cold bias
    (PIL plugin lazy-load, OS page cache populating). Set `shuffle=False`
    for reproducible debugging.
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
) -> list[dict[str, Any]]:
    return asyncio.run(
        run_timing(
            cases,
            warmup=warmup,
            repeat=repeat,
            seed=seed,
            shuffle=shuffle,
            track_python_allocs=track_python_allocs,
        )
    )
