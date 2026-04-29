"""Memory mode: capacity-planning measurement.

For each case, run a single iteration with `tracemalloc` active and
record `max(parent_peak_rss, children_peak_rss)` as the headline. The
RSS numbers come from `RUSAGE_*.ru_maxrss`, which is a high-water mark
over the process lifetime, not a delta — so multiple iterations in the
same parent give the same number. v0 runs one iteration per case
sequentially; isolation (one fresh subprocess per case for a clean
parent baseline) is deferred to v1.

Tracemalloc adds 30-50 % overhead and is therefore *off* in `timing`
mode. The numbers it produces (Python heap allocations) are
supplementary signal — `peak_rss_kb` is the primary capacity metric.
"""

from __future__ import annotations

import asyncio
from typing import Any

from bench.runner.case import Case
from bench.runner.modes.quick import run_quick


async def run_memory(cases: list[Case]) -> list[dict[str, Any]]:
    """1 iteration per case with tracemalloc enabled."""
    return await run_quick(cases, track_python_allocs=True)


def run_memory_sync(cases: list[Case]) -> list[dict[str, Any]]:
    return asyncio.run(run_memory(cases))
