"""Memory mode: capacity-planning measurement.

For each case, run a single iteration with `tracemalloc` active, record
`max(parent_peak_rss, children_peak_rss)` from `RUSAGE_*.ru_maxrss` as
the canonical headline, and additionally collect a 50 ms-cadence RSS
curve (parent + children sum from psutil) for visualization.

The two memory signals serve different purposes:

- **Headline (RUSAGE_*.ru_maxrss)** — sample-rate-independent peak.
  Cannot miss spikes. Use this for capacity planning / Cloud Run sizing.
- **Curve (psutil sampler)** — timeline showing how RSS evolves through
  decode → encode → strip phases. Can miss sub-50ms subprocess bursts
  (which is why it's not the primary signal). Useful for comparing
  shapes across runs.

Tracemalloc only sees Python heap allocations and adds 30-50% overhead;
it's enabled here as a third diagnostic signal (e.g. detecting a runaway
buffer copy) but is not the primary metric.

Memory mode runs one iteration per case sequentially; isolation (fresh
subprocess per case for a clean parent baseline) is deferred to v1.
"""

from __future__ import annotations

import asyncio
from typing import Any

from bench.runner.case import Case
from bench.runner.modes.quick import run_quick


async def run_memory(cases: list[Case]) -> list[dict[str, Any]]:
    """1 iteration per case with tracemalloc + 50ms RSS sampling."""
    return await run_quick(cases, track_python_allocs=True, track_rss_curve=True)


def run_memory_sync(cases: list[Case]) -> list[dict[str, Any]]:
    return asyncio.run(run_memory(cases))
