"""Report writer + compare tests."""

from __future__ import annotations

from pathlib import Path

from bench.runner.compare import compare, render_compare_markdown
from bench.runner.report.json_writer import (
    SCHEMA_VERSION,
    RunMetadata,
    load_run,
    write_run,
)
from bench.runner.report.markdown import render_run


def _iter(case_id: str, wall_ms: float, iteration: int = 0, fmt: str = "png") -> dict:
    return {
        "case_id": case_id,
        "name": case_id.split(".")[0],
        "bucket": "small",
        "format": fmt,
        "preset": case_id.split("@")[1],
        "input_size": 65536,
        "iteration": iteration,
        "measurement": {
            "wall_ms": wall_ms,
            "parent_user_ms": wall_ms * 0.3,
            "parent_sys_ms": wall_ms * 0.05,
            "children_user_ms": wall_ms * 0.6,
            "children_sys_ms": wall_ms * 0.05,
            "total_cpu_ms": wall_ms,
            "parallelism": 1.0,
            "parent_peak_rss_kb": 4096,
            "children_peak_rss_kb": 8192,
            "peak_rss_kb": 8192,
            "py_peak_alloc_kb": None,
            "phases": {},
        },
        "tool_invocations": [],
        "reduction_pct": 65.0,
        "method": "pngquant + oxipng",
        "optimized_size": 22937,
    }


def _metadata(mode: str = "timing") -> RunMetadata:
    return RunMetadata(
        mode=mode,
        config={"warmup": 1, "repeat": 5, "seed": 42, "shuffle": True},
        manifest_name="core",
        manifest_sha256="abc123" * 10,
    )


def test_write_and_load_round_trip(tmp_path: Path):
    iterations = [_iter("a.png@high", 10.0), _iter("a.png@high", 11.0, iteration=1)]
    out = tmp_path / "run.json"
    write_run(_metadata(), iterations, out)

    loaded = load_run(out)
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["mode"] == "timing"
    assert len(loaded["iterations"]) == 2
    assert len(loaded["stats"]) == 1


def test_load_rejects_unsupported_schema_version(tmp_path: Path):
    import pytest

    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 99, "iterations": [], "stats": []}')
    with pytest.raises(ValueError, match="schema_version"):
        load_run(bad)


def test_stats_rolled_up_from_iterations(tmp_path: Path):
    iterations = [
        _iter("a.png@high", 10.0, iteration=0),
        _iter("a.png@high", 11.0, iteration=1),
        _iter("a.png@high", 9.0, iteration=2),
    ]
    out = tmp_path / "run.json"
    write_run(_metadata(), iterations, out)
    loaded = load_run(out)
    assert len(loaded["stats"]) == 1
    s = loaded["stats"][0]
    assert s["case_id"] == "a.png@high"
    assert s["iterations"] == 3
    assert 9 <= s["median_ms"] <= 11


def test_errored_iterations_excluded_from_stats(tmp_path: Path):
    iterations = [
        _iter("a.png@high", 10.0),
        {
            "case_id": "b.png@high",
            "name": "b",
            "bucket": "small",
            "format": "png",
            "preset": "high",
            "iteration": 0,
            "error": "boom",
        },
    ]
    out = tmp_path / "run.json"
    write_run(_metadata(), iterations, out)
    loaded = load_run(out)
    # Only the successful case has stats; errors persist in iterations
    assert {s["case_id"] for s in loaded["stats"]} == {"a.png@high"}


def test_render_run_emits_markdown_with_table(tmp_path: Path):
    iterations = [_iter("a.png@high", 10.0), _iter("b.jpeg@medium", 25.0, fmt="jpeg")]
    out = tmp_path / "run.json"
    write_run(_metadata(), iterations, out)
    md = render_run(load_run(out))
    assert "Pare bench" in md
    assert "a.png@high" in md
    assert "b.jpeg@medium" in md
    assert "| case_id |" in md  # table header


def test_render_run_includes_error_section_when_failures_exist(tmp_path: Path):
    iterations = [
        {
            "case_id": "fail.png@high",
            "name": "fail",
            "bucket": "small",
            "format": "png",
            "preset": "high",
            "iteration": 0,
            "error": "OptimizationError: bad input",
        },
    ]
    out = tmp_path / "run.json"
    write_run(_metadata(), iterations, out)
    md = render_run(load_run(out))
    assert "Errors" in md
    assert "OptimizationError" in md


def test_compare_flags_regression(tmp_path: Path):
    """Head is significantly slower than baseline for a common case."""
    a_iters = [_iter("c.png@high", 10.0 + i * 0.1, iteration=i) for i in range(5)]
    b_iters = [_iter("c.png@high", 15.0 + i * 0.1, iteration=i) for i in range(5)]

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    write_run(_metadata(), a_iters, a_path)
    write_run(_metadata(), b_iters, b_path)

    result = compare(a_path, b_path, threshold_pct=10.0)
    assert len(result.regressions) == 1
    assert result.exit_code == 1
    assert result.regressions[0].case_id == "c.png@high"


def test_compare_no_regression_for_overlapping_distributions(tmp_path: Path):
    a_iters = [_iter("c.png@high", 10.0 + (i % 3) * 0.5, iteration=i) for i in range(5)]
    b_iters = [_iter("c.png@high", 10.2 + (i % 3) * 0.5, iteration=i) for i in range(5)]

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    write_run(_metadata(), a_iters, a_path)
    write_run(_metadata(), b_iters, b_path)

    result = compare(a_path, b_path, threshold_pct=10.0)
    assert result.regressions == []
    assert result.exit_code == 0


def test_compare_lists_only_in_a_and_only_in_b(tmp_path: Path):
    a_iters = [_iter("a_only.png@high", 10.0), _iter("shared.png@high", 11.0)]
    b_iters = [_iter("b_only.png@high", 10.0), _iter("shared.png@high", 11.0)]

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    write_run(_metadata(), a_iters, a_path)
    write_run(_metadata(), b_iters, b_path)

    result = compare(a_path, b_path)
    assert result.only_in_a == ["a_only.png@high"]
    assert result.only_in_b == ["b_only.png@high"]


def test_render_compare_markdown_contains_threshold_and_alpha(tmp_path: Path):
    a_iters = [_iter("c.png@high", 10.0, iteration=i) for i in range(3)]
    b_iters = [_iter("c.png@high", 12.0, iteration=i) for i in range(3)]

    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    write_run(_metadata(), a_iters, a_path)
    write_run(_metadata(), b_iters, b_path)

    md = render_compare_markdown(compare(a_path, b_path))
    assert "threshold=" in md
    assert "α=" in md


def test_compare_handles_empty_iterations(tmp_path: Path):
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    write_run(_metadata(), [], a_path)
    write_run(_metadata(), [], b_path)

    result = compare(a_path, b_path)
    assert result.diffs == []
    assert result.exit_code == 0


def test_runmetadata_default_timestamp_is_iso():
    md = RunMetadata(mode="timing", config={})
    assert "T" in md.timestamp  # ISO-8601 has T separator
