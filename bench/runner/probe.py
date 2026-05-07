"""Per-CLI-tool attribution for `run_tool()` invocations.

Pare's optimizers route every CLI call through `utils.subprocess_runner.run_tool`.
That module exposes a `run_tool_probe` contextvar; when this module's
`collect_tool_invocations()` is active, every call gets recorded with
its wall time and exit code. Outside the context, the probe is None and
adds no overhead.

This complements `bench.runner.measure.Measurement`. `Measurement` gives
aggregate CPU and RSS; `ToolInvocation` answers "where did the time go?"
— e.g. "the JPEG case spent 38 ms in mozjpeg, 12 ms in jpegtran."
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator

from utils.subprocess_runner import run_tool_probe


@dataclass
class ToolInvocation:
    tool: str
    wall_ms: float
    exit_code: int


@dataclass
class ToolStats:
    """Per-tool aggregate over a `collect_tool_invocations` window."""

    tool: str
    invocations: int = 0
    total_wall_ms: float = 0.0
    failures: int = 0
    invocation_wall_ms: list[float] = field(default_factory=list)


def aggregate(invocations: list[ToolInvocation]) -> dict[str, ToolStats]:
    """Group invocations by tool name and tally wall time + failures."""
    stats: dict[str, ToolStats] = {}
    for inv in invocations:
        s = stats.get(inv.tool)
        if s is None:
            s = ToolStats(tool=inv.tool)
            stats[inv.tool] = s
        s.invocations += 1
        s.total_wall_ms += inv.wall_ms
        s.invocation_wall_ms.append(inv.wall_ms)
        if inv.exit_code != 0:
            s.failures += 1
    return stats


@contextmanager
def collect_tool_invocations() -> Iterator[list[ToolInvocation]]:
    """Capture every `run_tool()` call inside the block.

    Yields a list that the probe appends to. Use `aggregate()` afterwards
    if you want per-tool rollups; the raw list is also useful for
    timeline analysis.
    """
    invocations: list[ToolInvocation] = []

    def probe(tool: str, wall_ms: float, exit_code: int) -> None:
        invocations.append(ToolInvocation(tool=tool, wall_ms=wall_ms, exit_code=exit_code))

    token = run_tool_probe.set(probe)
    try:
        yield invocations
    finally:
        run_tool_probe.reset(token)
