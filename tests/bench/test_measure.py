"""Subprocess-aware measurement tests.

These tests verify that `Measurement` actually captures subprocess CPU
and RSS — the very thing the legacy bench was missing. Since the test
runs CPU-bound work in a child process and asserts non-zero
`children_user_ms`, a regression to the old `process_time()` approach
would fail the suite immediately.
"""

from __future__ import annotations

import asyncio
import gc
import subprocess
import sys
import time

from bench.runner.measure import Measurement, measure, phase


def test_measure_captures_wall_time():
    with measure() as m:
        time.sleep(0.05)
    assert m.wall_ms >= 40
    assert m.wall_ms < 200


def test_measure_captures_parent_user_cpu():
    """Busy-loop in the parent — parent_user_ms should be > 0."""
    with measure() as m:
        # ~50ms of CPU-bound work in the parent
        end = time.perf_counter() + 0.05
        x = 0
        while time.perf_counter() < end:
            x += 1
    assert m.parent_user_ms > 5, f"parent_user_ms={m.parent_user_ms}"


def test_measure_captures_subprocess_cpu():
    """The headline test: subprocess CPU must show up in
    children_user_ms. Legacy bench's `time.process_time()` would report 0
    here — proving the new measurement actually counts subprocess work.
    """
    busy_loop = (
        "x = 0\n"
        "import time\n"
        "end = time.perf_counter() + 0.10\n"
        "while time.perf_counter() < end:\n"
        "    x += 1\n"
    )
    with measure() as m:
        subprocess.run([sys.executable, "-c", busy_loop], check=True)

    assert m.children_user_ms > 30, (
        f"children_user_ms={m.children_user_ms} — subprocess CPU not captured. "
        "This is the bug the legacy bench had."
    )


def test_measure_captures_async_subprocess_cpu():
    """Async subprocesses (the path Pare's run_tool() uses) must also
    surface in children CPU."""
    busy_loop = (
        "import time\n"
        "end = time.perf_counter() + 0.08\n"
        "x = 0\n"
        "while time.perf_counter() < end:\n"
        "    x += 1\n"
    )

    async def run():
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            busy_loop,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    with measure() as m:
        asyncio.run(run())

    assert m.children_user_ms > 20, f"children_user_ms={m.children_user_ms}"


def test_measure_total_cpu_combines_parent_and_children():
    busy_loop = (
        "import time\n"
        "end = time.perf_counter() + 0.05\n"
        "x = 0\n"
        "while time.perf_counter() < end:\n"
        "    x += 1\n"
    )
    with measure() as m:
        subprocess.run([sys.executable, "-c", busy_loop], check=True)
        end = time.perf_counter() + 0.05
        y = 0
        while time.perf_counter() < end:
            y += 1

    expected_floor = m.parent_user_ms + m.children_user_ms
    assert m.total_cpu_ms >= expected_floor - 0.01


def test_measure_parallelism_above_one_when_subprocess_runs_concurrently():
    """If the parent and a subprocess both burn CPU at the same time,
    parallelism should exceed 1.0."""
    busy_loop = (
        "import time\n"
        "end = time.perf_counter() + 0.10\n"
        "x = 0\n"
        "while time.perf_counter() < end:\n"
        "    x += 1\n"
    )
    with measure() as m:
        proc = subprocess.Popen([sys.executable, "-c", busy_loop])
        end = time.perf_counter() + 0.10
        y = 0
        while time.perf_counter() < end:
            y += 1
        proc.wait()  # reap so RUSAGE_CHILDREN sees this child

    assert m.parallelism > 1.0, f"parallelism={m.parallelism:.2f}"


def test_measure_captures_peak_rss():
    with measure() as m:
        time.sleep(0.01)
    assert m.parent_peak_rss_kb > 0
    # children RSS may be 0 if no subprocess was reaped during the window
    assert m.peak_rss_kb >= m.parent_peak_rss_kb


def test_track_python_allocs_records_peak():
    with measure(track_python_allocs=True) as m:
        big = [bytearray(64 * 1024) for _ in range(8)]
        del big
    assert m.py_peak_alloc_kb is not None
    assert m.py_peak_alloc_kb > 0


def test_track_python_allocs_default_off():
    with measure() as m:
        time.sleep(0.005)
    assert m.py_peak_alloc_kb is None


def test_phase_records_wall_time():
    with measure() as m:
        with phase("decode"):
            time.sleep(0.02)
        with phase("encode"):
            time.sleep(0.03)
    assert "decode" in m.phases
    assert "encode" in m.phases
    assert m.phases["decode"] > 10
    assert m.phases["encode"] > 20


def test_phase_outside_measure_is_noop():
    """Calling phase() outside a measure() block must not raise."""
    with phase("nothing"):
        pass


def test_phase_can_be_called_repeatedly_under_same_label():
    """Repeated phase() calls accumulate."""
    with measure() as m:
        for _ in range(3):
            with phase("strip"):
                time.sleep(0.005)
    assert m.phases["strip"] > 12


def test_gc_re_enabled_on_exception():
    """If the measured block raises, gc must still be re-enabled."""
    gc.enable()
    try:
        with measure():
            assert gc.isenabled() is False
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert gc.isenabled()


def test_tracemalloc_stopped_on_exception():
    import tracemalloc

    try:
        with measure(track_python_allocs=True):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not tracemalloc.is_tracing()


def test_measurement_default_total_cpu_zero():
    m = Measurement()
    assert m.total_cpu_ms == 0.0
    assert m.parallelism == 0.0


def test_peak_rss_takes_max_of_parent_and_children():
    m = Measurement(parent_peak_rss_kb=100, children_peak_rss_kb=300)
    assert m.peak_rss_kb == 300


def test_two_nested_measure_blocks_do_not_leak_phases():
    """Phase recorder is per-context; nested measures stay independent."""
    with measure() as outer:
        with phase("outer_phase"):
            time.sleep(0.005)
        with measure() as inner:
            with phase("inner_phase"):
                time.sleep(0.005)
        with phase("outer_again"):
            time.sleep(0.005)

    assert "outer_phase" in outer.phases
    assert "outer_again" in outer.phases
    assert "inner_phase" in inner.phases
    assert "inner_phase" not in outer.phases
    assert "outer_phase" not in inner.phases
