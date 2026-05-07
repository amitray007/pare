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
    parallelism should exceed 1.0.

    The subprocess writes a "ready" line to stdout *after* paying its
    Python startup cost, then burns CPU. The parent blocks on that line
    before starting its own busy loop, so the two CPU-burning windows
    reliably overlap regardless of interpreter cold-start variance.
    """
    busy_loop = (
        "import sys, time\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "end = time.perf_counter() + 0.20\n"
        "x = 0\n"
        "while time.perf_counter() < end:\n"
        "    x += 1\n"
    )
    with measure() as m:
        proc = subprocess.Popen(
            [sys.executable, "-c", busy_loop], stdout=subprocess.PIPE, text=True
        )
        assert proc.stdout is not None
        proc.stdout.readline()  # block until subprocess is in its busy loop
        end = time.perf_counter() + 0.20
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


def test_rss_curve_collects_samples_when_enabled():
    """Memory mode opts in to a 50 ms-cadence RSS curve via psutil."""
    from bench.runner.measure import measure as _measure

    with _measure(track_rss_curve=True, rss_sample_interval_ms=20) as m:
        # Run for ~120ms so we get >=4 samples at 20ms cadence.
        time.sleep(0.12)

    assert len(m.rss_samples) >= 3, f"got {len(m.rss_samples)} samples"
    for offset_ms, rss_kb in m.rss_samples:
        assert offset_ms >= 0
        assert rss_kb > 0


def test_rss_curve_off_by_default():
    """Default measure() does not spawn a sampler thread or collect samples."""
    with measure() as m:
        time.sleep(0.05)
    assert m.rss_samples == []


def test_rss_curve_includes_subprocess_rss():
    """Curve sums parent + children RSS, so a memory-hungry subprocess
    should bump the values during its lifetime."""
    import subprocess
    import sys

    # Allocate ~50 MB in the child and hold it for ~150ms.
    hog = (
        "import time\n"
        "buf = bytearray(50 * 1024 * 1024)\n"
        "for i in range(len(buf)):\n"
        "    if i % 4096 == 0:\n"
        "        buf[i] = 1\n"
        "time.sleep(0.15)\n"
    )
    with measure(track_rss_curve=True, rss_sample_interval_ms=20) as m:
        proc = subprocess.Popen([sys.executable, "-c", hog])
        time.sleep(0.18)
        proc.wait()

    assert m.rss_samples
    peak_kb = max(s[1] for s in m.rss_samples)
    # Expect at least 30 MB above the bare parent. The child was ~50 MB,
    # but psutil sampling can miss the exact peak — so be lenient.
    parent_baseline_kb = m.rss_samples[0][1]
    assert peak_kb - parent_baseline_kb > 30 * 1024, (
        f"expected child to inflate curve by >30MB, got "
        f"baseline={parent_baseline_kb}KB peak={peak_kb}KB"
    )


def test_rss_curve_warns_when_psutil_missing(monkeypatch):
    """If psutil isn't installed, the flag is a warn-and-no-op rather
    than a hard failure."""
    import warnings as _warnings

    from bench.runner import measure as measure_module

    monkeypatch.setattr(measure_module, "_PSUTIL_AVAILABLE", False)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        with measure(track_rss_curve=True) as m:
            time.sleep(0.01)
    assert m.rss_samples == []
    assert any("psutil" in str(w.message) for w in caught)


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
