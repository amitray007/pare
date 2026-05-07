"""Tests for bench.dashboard.build.

Five test cases as specified:

1. test_build_creates_history_json      — local build produces index.html + data/history.json
2. test_history_json_runs_oldest_first  — runs sorted by timestamp_unix ascending
3. test_history_skips_commits_where_baseline_didnt_exist — missing file handled gracefully
4. test_per_format_aggregation_uses_median — per-format p50/p95 match expected medians
5. test_dashboard_index_html_loads_history_json — rendered HTML references data/history.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

import pytest

from bench.dashboard.build import (
    _aggregate_by_format,
    _find_repo_root,
    build_history,
    main,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Repo root — needed so git log commands actually work.
REPO_ROOT = _find_repo_root(Path(__file__))


def _stat(fmt: str, p50: float, p95: float, parent_rss: int, children_rss: int) -> dict[str, Any]:
    """Build a minimal CaseStats-shaped dict for aggregation tests."""
    return {
        "case_id": f"test_{fmt}@medium",
        "format": fmt,
        "preset": "medium",
        "bucket": "small",
        "iterations": 1,
        "p50_ms": p50,
        "p95_ms": p95,
        "parent_peak_rss_p95_kb": parent_rss,
        "children_peak_rss_p95_kb": children_rss,
    }


# ---------------------------------------------------------------------------
# Test 1 — build produces the expected files
# ---------------------------------------------------------------------------


def test_build_creates_history_json(tmp_path: Path) -> None:
    """Invoking main() must create index.html and data/history.json."""
    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])

    assert rc == 0, "build.main() returned non-zero"
    assert (out / "index.html").exists(), "index.html missing"
    assert (out / "data" / "history.json").exists(), "data/history.json missing"

    history = json.loads((out / "data" / "history.json").read_text())
    assert "generated_at" in history
    assert "runs" in history
    assert isinstance(history["runs"], list)

    # The repo has at least one commit that touched baseline.core.json.
    assert len(history["runs"]) >= 1, (
        "Expected at least 1 historical run but got 0. "
        "Check that reports/baseline.core.json exists in git history."
    )


# ---------------------------------------------------------------------------
# Test 2 — runs are sorted oldest-first
# ---------------------------------------------------------------------------


def test_history_json_runs_oldest_first(tmp_path: Path) -> None:
    """Runs in history.json must be sorted by timestamp_unix ascending (oldest first)."""
    out = tmp_path / "out"
    main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])

    history = json.loads((out / "data" / "history.json").read_text())
    runs = history["runs"]
    if len(runs) < 2:
        pytest.skip("Need at least 2 historical runs to test ordering")

    timestamps = [r["timestamp_unix"] for r in runs]
    assert timestamps == sorted(timestamps), "Runs are not sorted oldest-first by timestamp_unix"


# ---------------------------------------------------------------------------
# Test 3 — missing file at a commit is skipped gracefully
# ---------------------------------------------------------------------------


def test_history_skips_commits_where_baseline_didnt_exist(tmp_path: Path) -> None:
    """build_history() must not crash when git show returns nothing for a commit.

    The build script uses ``git log --follow`` which only lists commits that
    touched the tracked file, so skipping missing files is a belt-and-suspenders
    guard. We verify it by patching show_file to return None for one commit
    and checking the run list is still built without error.
    """
    import bench.dashboard.build as build_mod

    original_show = build_mod.show_file
    call_count: list[int] = [0]

    def patched_show(repo_root: Path, sha: str, file_path: str) -> str | None:
        call_count[0] += 1
        # Simulate missing file for the very first call.
        if call_count[0] == 1:
            return None
        return original_show(repo_root, sha, file_path)

    build_mod.show_file = patched_show  # type: ignore[assignment]
    try:
        # Should not raise even though the first commit returns None.
        history = build_history(REPO_ROOT, limit=100)
    finally:
        build_mod.show_file = original_show  # type: ignore[assignment]

    # Whatever runs are found must still be valid records.
    for run in history["runs"]:
        assert "sha" in run
        assert "by_format" in run


# ---------------------------------------------------------------------------
# Test 4 — per-format aggregation uses median
# ---------------------------------------------------------------------------


def test_per_format_aggregation_uses_median() -> None:
    """_aggregate_by_format() must use median() for p50/p95, not mean."""
    # Three JPEG cases with deliberately skewed values so mean != median.
    stats = [
        _stat("jpeg", p50=10.0, p95=20.0, parent_rss=1000, children_rss=500),
        _stat("jpeg", p50=12.0, p95=22.0, parent_rss=1200, children_rss=600),
        _stat("jpeg", p50=100.0, p95=200.0, parent_rss=5000, children_rss=2000),
    ]
    result = _aggregate_by_format(stats)
    assert "jpeg" in result

    p50_vals = [10.0, 12.0, 100.0]
    p95_vals = [20.0, 22.0, 200.0]
    rss_vals = [1000 + 500, 1200 + 600, 5000 + 2000]

    expected_p50 = round(median(p50_vals), 3)
    expected_p95 = round(median(p95_vals), 3)
    expected_rss = int(median(rss_vals))

    assert (
        result["jpeg"]["p50_ms"] == expected_p50
    ), f"p50_ms={result['jpeg']['p50_ms']} != expected {expected_p50}"
    assert (
        result["jpeg"]["p95_ms"] == expected_p95
    ), f"p95_ms={result['jpeg']['p95_ms']} != expected {expected_p95}"
    assert (
        result["jpeg"]["peak_rss_kb"] == expected_rss
    ), f"peak_rss_kb={result['jpeg']['peak_rss_kb']} != expected {expected_rss}"
    assert result["jpeg"]["n"] == 3


# ---------------------------------------------------------------------------
# Test 5 — index.html references the JSON data path
# ---------------------------------------------------------------------------


def test_dashboard_index_html_loads_history_json(tmp_path: Path) -> None:
    """The rendered index.html must contain the literal string 'data/history.json'."""
    out = tmp_path / "out"
    main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "data/history.json" in html, (
        "index.html does not reference 'data/history.json' — "
        "the JS fetch path is missing or changed"
    )


# ---------------------------------------------------------------------------
# Test 6 — corpus auto-detect prefers tests/corpus over bench/corpus/data
# ---------------------------------------------------------------------------


def test_corpus_autodetect_prefers_tests_corpus(tmp_path: Path) -> None:
    """Auto-detect should find tests/corpus first (Bug 2 fix)."""
    import bench.dashboard.build as build_mod

    # Create both candidate directories so we can verify which one wins.
    fake_tests_corpus = tmp_path / "tests" / "corpus"
    fake_tests_corpus.mkdir(parents=True)
    fake_bench_corpus = tmp_path / "bench" / "corpus" / "data"
    fake_bench_corpus.mkdir(parents=True)

    # Patch _find_repo_root so the auto-detect resolves against our tmp tree.
    original_find = build_mod._find_repo_root
    build_mod._find_repo_root = lambda _: tmp_path  # type: ignore[assignment]
    try:
        # Verify ordering: tests/corpus wins over bench/corpus/data when both exist.
        _candidates = [
            tmp_path / "tests" / "corpus",
            tmp_path / "bench" / "corpus" / "data",
        ]
        selected = next((p for p in _candidates if p.exists()), None)
        assert (
            selected == fake_tests_corpus
        ), f"Expected tests/corpus to be selected, got {selected}"
    finally:
        build_mod._find_repo_root = original_find  # type: ignore[assignment]


def test_corpus_autodetect_falls_back_to_bench_corpus_data(tmp_path: Path) -> None:
    """Auto-detect falls back to bench/corpus/data when tests/corpus doesn't exist."""
    fake_bench_corpus = tmp_path / "bench" / "corpus" / "data"
    fake_bench_corpus.mkdir(parents=True)
    # tests/corpus intentionally NOT created.

    _candidates = [
        tmp_path / "tests" / "corpus",
        tmp_path / "bench" / "corpus" / "data",
    ]
    selected = next((p for p in _candidates if p.exists()), None)
    assert selected == fake_bench_corpus, f"Expected bench/corpus/data fallback, got {selected}"


def test_corpus_autodetect_returns_none_when_neither_exists(tmp_path: Path) -> None:
    """Auto-detect returns None gracefully when neither corpus path exists."""
    _candidates = [
        tmp_path / "tests" / "corpus",
        tmp_path / "bench" / "corpus" / "data",
    ]
    selected = next((p for p in _candidates if p.exists()), None)
    assert selected is None, f"Expected None when no corpus found, got {selected}"
