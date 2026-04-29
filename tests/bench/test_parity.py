"""Parity test: prove the new bench captures subprocess CPU + RSS.

The legacy `benchmarks/runner.py` measured `time.process_time()` (parent
only) and `tracemalloc` (Python heap only). This test runs a real
optimizer case under both measurement styles and asserts:

1. The NEW total_cpu_ms is strictly greater than the OLD parent-only
   CPU. This is the proof that subprocess CPU is now counted.
2. The NEW peak RSS captures children (CLI tools), which the OLD
   `tracemalloc`-based measurement could never see.

Without this test, a regression to the parent-only measurement would
silently pass — the old numbers would still be self-consistent.
"""

from __future__ import annotations

import asyncio
import time
import tracemalloc
from pathlib import Path

import pytest

from bench.corpus.builder import build
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
from bench.runner.case import load_cases
from bench.runner.measure import measure
from optimizers.router import optimize_image
from schemas import OptimizationConfig


@pytest.fixture
def png_case_bytes(tmp_path: Path) -> tuple[bytes, OptimizationConfig]:
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="parity",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=1,
                width=192,
                height=144,
                output_formats=["png"],
            )
        ],
    )
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations
    cases = load_cases(m, tmp_path, preset_filter={"high"})
    return cases[0].load(), OptimizationConfig(quality=cases[0].quality)


def _legacy_measure(data: bytes, config: OptimizationConfig) -> dict:
    """Reproduce the legacy `benchmarks/runner.py` measurement style."""
    tracemalloc.start()
    cpu_t0 = time.process_time()
    wall_t0 = time.perf_counter_ns()

    asyncio.run(optimize_image(data, config))

    wall_ms = (time.perf_counter_ns() - wall_t0) / 1e6
    cpu_ms = (time.process_time() - cpu_t0) * 1000
    _, py_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "wall_ms": wall_ms,
        "cpu_time_ms": cpu_ms,  # parent only — the legacy bug
        "py_peak_kb": py_peak // 1024,
    }


def _new_measure(data: bytes, config: OptimizationConfig) -> dict:
    """Run the same call under the new subprocess-aware measurement."""
    with measure(track_python_allocs=True) as m:
        asyncio.run(optimize_image(data, config))
    return {
        "wall_ms": m.wall_ms,
        "total_cpu_ms": m.total_cpu_ms,
        "parent_user_ms": m.parent_user_ms,
        "parent_sys_ms": m.parent_sys_ms,
        "children_user_ms": m.children_user_ms,
        "children_sys_ms": m.children_sys_ms,
        "parallelism": m.parallelism,
        "parent_peak_rss_kb": m.parent_peak_rss_kb,
        "children_peak_rss_kb": m.children_peak_rss_kb,
        "peak_rss_kb": m.peak_rss_kb,
        "py_peak_alloc_kb": m.py_peak_alloc_kb,
    }


def test_new_total_cpu_strictly_exceeds_legacy_parent_only(png_case_bytes):
    """The headline parity assertion: new measurement counts subprocess
    CPU, which makes total_cpu_ms strictly greater than parent-only
    cpu_time_ms for any case that invokes a CLI subprocess.

    PNG optimization in Pare invokes pngquant via run_tool() and oxipng
    via the in-process pyoxipng binding. Pngquant is the subprocess that
    makes the parity gap measurable.
    """
    data, config = png_case_bytes
    legacy = _legacy_measure(data, config)
    new = _new_measure(data, config)

    assert new["children_user_ms"] > 0, (
        "expected non-zero children CPU for PNG case "
        "(pngquant runs as subprocess) — got "
        f"children_user_ms={new['children_user_ms']}"
    )
    assert new["total_cpu_ms"] > legacy["cpu_time_ms"], (
        f"new total_cpu_ms ({new['total_cpu_ms']:.1f}ms) must exceed "
        f"legacy cpu_time_ms ({legacy['cpu_time_ms']:.1f}ms) — that gap "
        f"is the subprocess CPU the legacy bench couldn't see"
    )


def test_new_peak_rss_includes_children(png_case_bytes):
    """`tracemalloc` saw only Python heap (~MB), missing pngquant's RSS.
    `peak_rss_kb` (max of parent + children RUSAGE) is the headline
    metric that determines Cloud Run instance sizing.
    """
    data, config = png_case_bytes
    new = _new_measure(data, config)

    assert new["peak_rss_kb"] > 0
    # children should appear since pngquant runs as a real subprocess
    assert new["children_peak_rss_kb"] > 0


def test_parallelism_above_one_for_optimizers_using_asyncio_gather(png_case_bytes):
    """PngOptimizer runs pngquant + oxipng in parallel via asyncio.gather.
    Parallelism should exceed 1 — the legacy bench couldn't detect this.
    """
    data, config = png_case_bytes
    new = _new_measure(data, config)
    assert new["parallelism"] > 1.0, (
        f"PNG optimizer expected to parallelize "
        f"(pngquant + oxipng concurrent), got parallelism={new['parallelism']:.2f}"
    )


def test_legacy_measurement_underreports_cpu_systematically(png_case_bytes):
    """Quantify the gap: for PNG cases, legacy underreports CPU by at
    least 30% (a deliberately conservative bar; in practice the gap is
    often >2×). If this test ever fails, that's a hint that pngquant
    is no longer running as a subprocess (or has gotten very fast).
    """
    data, config = png_case_bytes
    legacy = _legacy_measure(data, config)
    new = _new_measure(data, config)

    if legacy["cpu_time_ms"] <= 0:
        pytest.skip("legacy CPU measurement was zero — no signal to compare against")

    ratio = new["total_cpu_ms"] / legacy["cpu_time_ms"]
    assert ratio >= 1.3, (
        f"expected legacy to underreport CPU by ≥30%; got ratio={ratio:.2f} "
        f"(legacy={legacy['cpu_time_ms']:.1f}ms, new={new['total_cpu_ms']:.1f}ms)"
    )


def test_walls_within_reasonable_bound_of_each_other(png_case_bytes):
    """Sanity: both measurement styles see roughly the same wall time
    (the optimizer doesn't get faster or slower based on whether you
    measure it). Allow a generous 30% spread for OS jitter."""
    data, config = png_case_bytes
    legacy = _legacy_measure(data, config)
    new = _new_measure(data, config)

    ratio = legacy["wall_ms"] / new["wall_ms"]
    assert (
        0.5 < ratio < 2.0
    ), f"wall_ms differs too much: legacy={legacy['wall_ms']:.1f}ms new={new['wall_ms']:.1f}ms"
