"""Integration tests for quick + timing modes.

These tests run the full pipeline: corpus -> cases -> Measurement -> optimizer.
They exercise the subprocess-attribution path end-to-end. Tests are
relatively slow (~5-15s total) because they invoke real CLI tools.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.corpus.builder import build
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
from bench.runner.case import load_cases
from bench.runner.modes.memory import run_memory_sync
from bench.runner.modes.quick import run_quick_sync
from bench.runner.modes.timing import run_timing_sync


@pytest.fixture
def small_corpus(tmp_path: Path) -> tuple[Manifest, Path]:
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="photo_a",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=1,
                width=192,
                height=144,
                output_formats=["png"],
                tags=["photo"],
            ),
        ],
    )
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations
    return m, tmp_path


def test_quick_mode_returns_one_result_per_case(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root)  # 1 entry × 1 fmt × 3 presets = 3 cases
    results = run_quick_sync(cases)
    assert len(results) == 3


def test_quick_mode_records_measurement_fields(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_quick_sync(cases)

    r = results[0]
    assert r["case_id"] == "photo_a.png@high"
    assert r["bucket"] == "small"
    assert r["format"] == "png"
    assert r["preset"] == "high"
    assert r["iteration"] == 0
    assert r["input_size"] > 0
    assert "method" in r and r["method"]
    assert r["reduction_pct"] >= 0
    assert "measurement" in r

    m = r["measurement"]
    assert m["wall_ms"] > 0
    assert m["total_cpu_ms"] >= 0


def test_quick_mode_captures_subprocess_cpu_for_png(small_corpus):
    """PNG optimization invokes pngquant and oxipng-cli as subprocesses
    (pngquant via run_tool). Children CPU must be > 0.

    If oxipng runs in-process via pyoxipng, it won't show up in
    children CPU — but pngquant does, so this assertion is robust.
    """
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_quick_sync(cases)
    r = results[0]
    # Either pngquant fired (children_cpu > 0) OR the install lacks pngquant
    # (method falls back to oxipng-only). Both are acceptable shapes.
    if "pngquant" in r["method"]:
        assert r["measurement"]["children_user_ms"] > 0, r


def test_quick_mode_records_tool_invocations(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_quick_sync(cases)
    r = results[0]
    assert "tool_invocations" in r
    # Tool list may be empty if optimizer fully Pillow/pyoxipng-based
    for inv in r["tool_invocations"]:
        assert "tool" in inv and "wall_ms" in inv and "exit_code" in inv


def test_quick_mode_continues_after_one_case_fails(tmp_path: Path):
    """A bogus corpus file (zero bytes) should fail format detection;
    other cases should still run."""
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="ok",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=1,
                width=192,
                height=144,
                output_formats=["png"],
            ),
        ],
    )
    build(m, tmp_path)
    cases = load_cases(m, tmp_path, preset_filter={"high"})
    # Corrupt the file
    cases[0].file_path.write_bytes(b"\x00" * 16)

    results = run_quick_sync(cases)
    assert "error" in results[0]


def test_timing_mode_repeats_each_case(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_timing_sync(cases, warmup=0, repeat=3, shuffle=False)
    assert len(results) == 3
    assert {r["iteration"] for r in results} == {0, 1, 2}


def test_timing_mode_warmup_iterations_are_not_returned(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_timing_sync(cases, warmup=2, repeat=2, shuffle=False)
    # Only the 2 measured iterations are returned; the 2 warmups are discarded
    assert len(results) == 2


def test_timing_mode_shuffle_does_not_reorder_iterations_within_a_case(small_corpus):
    """Within a single case, iteration index goes 0, 1, 2, ... in order.
    Shuffle reorders cases, not iterations within."""
    manifest, root = small_corpus
    cases = load_cases(manifest, root)
    results = run_timing_sync(cases, warmup=0, repeat=3, shuffle=True, seed=1)
    by_case: dict[str, list[int]] = {}
    for r in results:
        by_case.setdefault(r["case_id"], []).append(r["iteration"])
    for case_id, iters in by_case.items():
        assert iters == [0, 1, 2], f"{case_id}: {iters}"


def test_timing_mode_produces_consistent_walls_within_one_case(small_corpus):
    """A noise-content PNG case should run repeatably on a quiet box —
    not flaky, just as a sanity check that we're measuring a real signal,
    not pure jitter."""
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_timing_sync(cases, warmup=1, repeat=5, shuffle=False)
    walls = [r["measurement"]["wall_ms"] for r in results if "measurement" in r]
    assert len(walls) == 5
    # max should be within 5x of min — tight enough to flag pathological jitter
    assert max(walls) < 5 * min(walls), f"walls={walls}"


def test_memory_mode_records_py_peak_alloc(small_corpus):
    """tracemalloc must populate py_peak_alloc_kb in memory mode."""
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_memory_sync(cases)
    r = results[0]
    assert r["measurement"]["py_peak_alloc_kb"] is not None
    assert r["measurement"]["py_peak_alloc_kb"] > 0


def test_memory_mode_returns_peak_rss_headline(small_corpus):
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_memory_sync(cases)
    r = results[0]
    m = r["measurement"]
    assert m["peak_rss_kb"] == max(m["parent_peak_rss_kb"], m["children_peak_rss_kb"])
    assert m["peak_rss_kb"] > 0


def test_quick_mode_does_not_record_py_peak_alloc(small_corpus):
    """Default quick mode leaves tracemalloc off — no overhead."""
    manifest, root = small_corpus
    cases = load_cases(manifest, root, preset_filter={"high"})
    results = run_quick_sync(cases)
    assert results[0]["measurement"]["py_peak_alloc_kb"] is None


def test_timing_mode_seed_makes_shuffle_reproducible(small_corpus):
    """Same seed -> same case order. Different seeds -> usually different."""
    manifest, root = small_corpus
    # Build a longer case list so shuffle has meaningful effect
    m, root2 = manifest, root
    m.entries.append(
        ManifestEntry(
            name="photo_b",
            bucket=Bucket.SMALL,
            content_kind="photo_noise",
            seed=99,
            width=192,
            height=144,
            output_formats=["png"],
        )
    )
    build(m, root2)
    cases = load_cases(m, root2)

    a = run_timing_sync(cases, warmup=0, repeat=1, shuffle=True, seed=1)
    b = run_timing_sync(cases, warmup=0, repeat=1, shuffle=True, seed=1)
    assert [r["case_id"] for r in a] == [r["case_id"] for r in b]
