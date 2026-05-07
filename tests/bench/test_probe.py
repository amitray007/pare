"""Tests for the run_tool probe + bench-side collector."""

from __future__ import annotations

import asyncio
import sys

import pytest

from bench.runner.probe import (
    ToolInvocation,
    aggregate,
    collect_tool_invocations,
)
from utils.subprocess_runner import run_tool, run_tool_probe


def _run(coro):
    return asyncio.run(coro)


def test_probe_default_is_none_in_production():
    """The probe must be None outside an active collector — no overhead."""
    assert run_tool_probe.get() is None


def test_collect_records_tool_name_and_exit_code():
    async def go():
        with collect_tool_invocations() as invocations:
            await run_tool([sys.executable, "-c", "print('hi')"], b"")
        return invocations

    invocations = _run(go())
    assert len(invocations) == 1
    assert invocations[0].tool == sys.executable
    assert invocations[0].exit_code == 0
    assert invocations[0].wall_ms > 0


def test_collect_captures_multiple_invocations():
    async def go():
        with collect_tool_invocations() as invocations:
            for _ in range(3):
                await run_tool([sys.executable, "-c", "pass"], b"")
        return invocations

    assert len(_run(go())) == 3


def test_collect_captures_nonzero_exit_via_allowed_codes():
    """Even when the called tool exits non-zero, the probe still fires."""

    async def go():
        with collect_tool_invocations() as invocations:
            await run_tool(
                [sys.executable, "-c", "import sys; sys.exit(99)"],
                b"",
                allowed_exit_codes={99},
            )
        return invocations

    invocations = _run(go())
    assert invocations[0].exit_code == 99


def test_collect_records_invocation_when_run_tool_raises():
    """A failed (non-allowed) exit raises OptimizationError, but the
    probe should already have recorded the invocation so the bench can
    see what failed."""
    from exceptions import OptimizationError

    async def go():
        with collect_tool_invocations() as invocations:
            with pytest.raises(OptimizationError):
                await run_tool(
                    [sys.executable, "-c", "import sys; sys.exit(7)"],
                    b"",
                )
        return invocations

    invocations = _run(go())
    assert len(invocations) == 1
    assert invocations[0].exit_code == 7


def test_collect_resets_probe_on_exit():
    async def go():
        with collect_tool_invocations():
            assert run_tool_probe.get() is not None
        assert run_tool_probe.get() is None

    _run(go())


def test_collect_resets_probe_even_if_block_raises():
    async def go():
        try:
            with collect_tool_invocations():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert run_tool_probe.get() is None

    _run(go())


def test_aggregate_groups_by_tool_name():
    invocations = [
        ToolInvocation(tool="mozjpeg", wall_ms=10.0, exit_code=0),
        ToolInvocation(tool="mozjpeg", wall_ms=20.0, exit_code=0),
        ToolInvocation(tool="jpegtran", wall_ms=5.0, exit_code=0),
    ]
    stats = aggregate(invocations)
    assert set(stats) == {"mozjpeg", "jpegtran"}
    assert stats["mozjpeg"].invocations == 2
    assert stats["mozjpeg"].total_wall_ms == 30.0
    assert stats["mozjpeg"].invocation_wall_ms == [10.0, 20.0]
    assert stats["jpegtran"].invocations == 1


def test_aggregate_counts_failures():
    invocations = [
        ToolInvocation(tool="pngquant", wall_ms=5.0, exit_code=0),
        ToolInvocation(tool="pngquant", wall_ms=4.0, exit_code=99),
    ]
    stats = aggregate(invocations)
    assert stats["pngquant"].failures == 1
    assert stats["pngquant"].invocations == 2


def test_aggregate_returns_empty_for_no_invocations():
    assert aggregate([]) == {}


def test_concurrent_run_tools_attribute_correctly():
    """Multiple async run_tool() calls in flight must all be captured.
    contextvars propagate to spawned tasks, so this should Just Work."""

    async def go():
        with collect_tool_invocations() as invocations:
            await asyncio.gather(
                run_tool([sys.executable, "-c", "pass"], b""),
                run_tool([sys.executable, "-c", "pass"], b""),
                run_tool([sys.executable, "-c", "pass"], b""),
            )
        return invocations

    assert len(_run(go())) == 3


def test_probe_exception_does_not_break_run_tool():
    """A buggy probe must not crash the optimizer."""

    def crashing_probe(tool: str, wall_ms: float, exit_code: int) -> None:
        raise RuntimeError("probe is buggy")

    async def go():
        token = run_tool_probe.set(crashing_probe)
        try:
            stdout, _, _ = await run_tool([sys.executable, "-c", "print('ok')"], b"")
        finally:
            run_tool_probe.reset(token)
        return stdout

    out = _run(go())
    assert b"ok" in out
