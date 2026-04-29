"""Subprocess-aware measurement primitive.

This is the bedrock the rest of `bench.runner` builds on. The legacy
`benchmarks/` runner used `time.process_time()` and `tracemalloc`, both
of which see only the parent Python process — but Pare's optimizers do
80–95 % of their CPU work inside subprocesses (mozjpeg, pngquant, oxipng,
cjxl, gifsicle, ...), and `tracemalloc` does not see subprocess RSS at
all. Numbers from the legacy bench are therefore off by an order of
magnitude.

This module captures the honest totals via `resource.getrusage()`:

- `RUSAGE_SELF` — parent process user/system CPU + peak RSS.
- `RUSAGE_CHILDREN` — aggregate over **reaped** children. For Pare's
  optimizers, every subprocess is awaited via `proc.communicate()` /
  `proc.wait()` before the optimizer returns, so reaping happens inside
  the timed window. If your code spawns a child that lives past the
  context manager exit, its CPU is *not* attributed.

`ru_maxrss` units differ across platforms (bytes on Darwin/BSD, KB on
Linux); they're normalized to KB before being stored.

Tracemalloc is opt-in via `track_python_allocs=True` because it adds
30–50 % overhead; the timing-mode bench leaves it off and reads peak
RSS from `RUSAGE_*` instead.
"""

from __future__ import annotations

import contextvars
import gc
import platform
import resource
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

# Per-context phase recorder. `phase()` writes into this dict while a
# `measure()` block holds the contextvar; outside that block the var is
# None and `phase()` is a no-op.
_phase_recorder: contextvars.ContextVar[dict[str, float] | None] = contextvars.ContextVar(
    "bench_phase_recorder", default=None
)


@dataclass
class Measurement:
    """A single observation from one wrapped block of work."""

    wall_ms: float = 0.0

    # Parent process CPU (the Python process running the bench).
    parent_user_ms: float = 0.0
    parent_sys_ms: float = 0.0

    # Aggregated CPU from all reaped child processes during the window.
    children_user_ms: float = 0.0
    children_sys_ms: float = 0.0

    # Peak resident set size, normalized to KB. Note: `RUSAGE_*.ru_maxrss`
    # is a *high-water mark* over the lifetime of the process(es), not a
    # delta — a fresh subprocess gives a clean baseline; reusing a
    # parent across iterations means the parent's number monotonically
    # grows. Use `--isolate` modes for clean parent_peak_rss_kb numbers.
    parent_peak_rss_kb: int = 0
    children_peak_rss_kb: int = 0

    # Set only when `track_python_allocs=True`.
    py_peak_alloc_kb: int | None = None

    # Optional per-phase wall-time breakdown, populated by `phase()` calls
    # inside the measured block.
    phases: dict[str, float] = field(default_factory=dict)

    @property
    def total_cpu_ms(self) -> float:
        return (
            self.parent_user_ms + self.parent_sys_ms + self.children_user_ms + self.children_sys_ms
        )

    @property
    def parallelism(self) -> float:
        """Effective CPU parallelism: total_cpu / wall.

        Values > 1.0 mean the case used multiple cores during the
        window; e.g. PNG (pngquant + oxipng under `asyncio.gather`)
        or JPEG (jpegli + jpegtran) routinely exceed 1.5×.
        """
        return self.total_cpu_ms / self.wall_ms if self.wall_ms > 0 else 0.0

    @property
    def peak_rss_kb(self) -> int:
        """Capacity-planning headline: max(parent, children) peak RSS."""
        return max(self.parent_peak_rss_kb, self.children_peak_rss_kb)


_BSD_LIKE = {"Darwin", "FreeBSD", "OpenBSD", "NetBSD"}


def _maxrss_to_kb(maxrss: int) -> int:
    """`ru_maxrss` is bytes on Darwin/BSD, KB on Linux. Normalize."""
    if platform.system() in _BSD_LIKE:
        return maxrss // 1024
    return maxrss


@contextmanager
def measure(*, track_python_allocs: bool = False) -> Iterator[Measurement]:
    """Wrap a block of work; populate a `Measurement` on exit.

    GC is collected once before the timer starts and disabled inside the
    block to keep collection-related noise out of the measurement. It is
    re-enabled in `finally` so an exception cannot leave the interpreter
    in a no-GC state.
    """
    m = Measurement()

    if track_python_allocs:
        tracemalloc.start()

    gc.collect()
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()

    ru0_self = resource.getrusage(resource.RUSAGE_SELF)
    ru0_children = resource.getrusage(resource.RUSAGE_CHILDREN)
    token = _phase_recorder.set(m.phases)
    t0 = time.perf_counter_ns()

    try:
        yield m
    finally:
        t1 = time.perf_counter_ns()
        ru1_children = resource.getrusage(resource.RUSAGE_CHILDREN)
        ru1_self = resource.getrusage(resource.RUSAGE_SELF)
        _phase_recorder.reset(token)

        if gc_was_enabled:
            gc.enable()

        if track_python_allocs:
            try:
                _current, peak = tracemalloc.get_traced_memory()
                m.py_peak_alloc_kb = peak // 1024
            finally:
                tracemalloc.stop()

        m.wall_ms = (t1 - t0) / 1e6
        m.parent_user_ms = (ru1_self.ru_utime - ru0_self.ru_utime) * 1000
        m.parent_sys_ms = (ru1_self.ru_stime - ru0_self.ru_stime) * 1000
        m.children_user_ms = (ru1_children.ru_utime - ru0_children.ru_utime) * 1000
        m.children_sys_ms = (ru1_children.ru_stime - ru0_children.ru_stime) * 1000
        m.parent_peak_rss_kb = _maxrss_to_kb(ru1_self.ru_maxrss)
        m.children_peak_rss_kb = _maxrss_to_kb(ru1_children.ru_maxrss)


@contextmanager
def phase(name: str) -> Iterator[None]:
    """Record per-phase wall time inside an active `measure()` block.

    No-op if there's no active recorder (production code paths can leave
    `phase()` calls in place at zero cost).
    """
    recorder = _phase_recorder.get()
    if recorder is None:
        yield
        return
    t0 = time.perf_counter_ns()
    try:
        yield
    finally:
        t1 = time.perf_counter_ns()
        recorder[name] = recorder.get(name, 0.0) + (t1 - t0) / 1e6
