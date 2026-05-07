"""Unit tests for the noise-floor fallback in bench/runner/compare.py.

Covers:
  - 1-iter each, +30% delta  → noise_floor_regression (exit 1)
  - 1-iter each, -50% delta  → improvement (exit 0)  [Bug 1 fix]
  - 1-iter each, +5% delta   → noise_floor_ok (exit 0)
  - 5-iter each, +30% delta, low variance  → significant regression (exit 1)
  - 5-iter each, -50% delta, low variance  → improvement (exit 0)
  - 5-iter each, +5% delta,  low variance  → below_threshold (exit 0)
  - Custom noise_floor_pct=50 suppresses a +30% flag until +50%+
  - Label display strings via format_compare_label
  - Per-format rollup: low-power improvements counted under Improvements, not Regressions
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.runner.compare import compare
from bench.runner.report.json_writer import RunMetadata, write_run
from bench.runner.report.markdown import build_format_rollup, format_compare_label  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers (mirrors test_report.py)
# ---------------------------------------------------------------------------


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
        "reduction_pct": 60.0,
        "method": "pngquant",
        "optimized_size": 26214,
    }


def _metadata() -> RunMetadata:
    return RunMetadata(
        mode="quick",
        config={"warmup": 0, "repeat": 1},
        manifest_name="core",
        manifest_sha256="abc123" * 10,
    )


def _write(tmp_path: Path, name: str, iters: list) -> Path:
    p = tmp_path / name
    write_run(_metadata(), iters, p)
    return p


# ---------------------------------------------------------------------------
# Noise-floor path (< 3 iters on either side)
# ---------------------------------------------------------------------------


def test_noise_floor_regression_1iter_30pct(tmp_path: Path):
    """1 iteration each, +30% delta → noise_floor_regression, exit 1."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 130.0)])

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is True
    assert d.label == "noise_floor_regression"
    assert len(result.noise_floor_flags) == 1
    assert result.regressions == []
    assert result.exit_code == 1


def test_noise_floor_ok_1iter_5pct(tmp_path: Path):
    """1 iteration each, +5% delta → noise_floor_ok, exit 0."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 105.0)])

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is False
    assert d.label == "noise_floor_ok"
    assert result.noise_floor_flags == []
    assert result.regressions == []
    assert result.exit_code == 0


def test_noise_floor_boundary_exactly_at_threshold(tmp_path: Path):
    """Exactly at noise_floor_pct — must be flagged (>= is inclusive)."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 125.0)])  # +25%

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is True
    assert result.exit_code == 1


def test_noise_floor_improvement_1iter_50pct(tmp_path: Path):
    """1 iteration each, -50% delta (head much faster) → improvement, NOT noise_floor_regression, exit 0."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 50.0)])  # -50%

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is True
    assert d.delta_pct < 0, "delta_pct should be negative (head faster)"
    # Bug 1 fix: negative delta past noise floor → improvement, not noise_floor_regression
    assert d.label == "improvement", f"Expected 'improvement', got '{d.label}'"
    assert result.noise_floor_flags == [], "Improvements must NOT appear in noise_floor_flags"
    assert result.regressions == []
    assert result.exit_code == 0, "Improvements must not fail CI"


# ---------------------------------------------------------------------------
# Stats path (>= 3 iters on both sides)
# ---------------------------------------------------------------------------


def test_stats_significant_regression_5iter_30pct(tmp_path: Path):
    """5 iterations each with tight spread, +30% delta → significant regression, exit 1."""
    a_iters = [_iter("img.png@high", 100.0 + i * 0.1, iteration=i) for i in range(5)]
    b_iters = [_iter("img.png@high", 130.0 + i * 0.1, iteration=i) for i in range(5)]

    a = _write(tmp_path, "a.json", a_iters)
    b = _write(tmp_path, "b.json", b_iters)

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is False
    assert d.significant is True
    assert d.threshold_breach is True
    assert d.label == "significant"
    assert len(result.regressions) == 1
    assert result.noise_floor_flags == []
    assert result.exit_code == 1


def test_stats_below_threshold_5iter_5pct(tmp_path: Path):
    """5 iterations each with tight spread, +5% delta → below_threshold, exit 0."""
    a_iters = [_iter("img.png@high", 100.0 + i * 0.05, iteration=i) for i in range(5)]
    b_iters = [_iter("img.png@high", 105.0 + i * 0.05, iteration=i) for i in range(5)]

    a = _write(tmp_path, "a.json", a_iters)
    b = _write(tmp_path, "b.json", b_iters)

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is False
    # +5% is below the 10% threshold, so no regression regardless of significance
    assert d.threshold_breach is False
    assert d.label in ("below_threshold", "ok")
    assert result.regressions == []
    assert result.exit_code == 0


def test_stats_improvement_5iter_50pct(tmp_path: Path):
    """5 iterations each with tight spread, -50% delta → improvement, exit 0."""
    a_iters = [_iter("img.png@high", 100.0 + i * 0.1, iteration=i) for i in range(5)]
    b_iters = [_iter("img.png@high", 50.0 + i * 0.1, iteration=i) for i in range(5)]

    a = _write(tmp_path, "a.json", a_iters)
    b = _write(tmp_path, "b.json", b_iters)

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)

    assert len(result.diffs) == 1
    d = result.diffs[0]
    assert d.iters_low_power is False
    assert d.label == "improvement"
    assert len(result.improvements) == 1
    assert result.regressions == []
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Custom noise_floor_pct
# ---------------------------------------------------------------------------


def test_custom_noise_floor_pct_50_suppresses_30pct(tmp_path: Path):
    """noise_floor_pct=50 — a +30% delta on 1 iter is below the floor, not flagged."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 130.0)])  # +30%

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=50.0)

    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is False
    assert d.label == "noise_floor_ok"
    assert result.exit_code == 0


def test_custom_noise_floor_pct_50_flags_60pct(tmp_path: Path):
    """noise_floor_pct=50 — a +60% delta on 1 iter IS above the floor, flagged."""
    a = _write(tmp_path, "a.json", [_iter("img.png@high", 100.0)])
    b = _write(tmp_path, "b.json", [_iter("img.png@high", 160.0)])  # +60%

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=50.0)

    d = result.diffs[0]
    assert d.iters_low_power is True
    assert d.threshold_breach is True
    assert d.label == "noise_floor_regression"
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# format_compare_label display mapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, expected",
    [
        ("significant", "❌ regression"),
        ("noise_floor_regression", "⚠ noise-floor"),
        ("improvement", "✅ improvement"),
        ("below_threshold", "~"),
        ("noise_floor_ok", "~"),
        ("ok", "~"),
        ("unknown_label", "unknown_label"),  # passthrough for unknown strings
    ],
)
def test_format_compare_label(label: str, expected: str):
    assert format_compare_label(label) == expected


# ---------------------------------------------------------------------------
# Per-format rollup: low-power improvements counted correctly (Bug 1 fix)
# ---------------------------------------------------------------------------


def test_rollup_low_power_improvement_counted_as_improvement(tmp_path: Path):
    """1-iter each, -50% delta → rollup shows Improvements=1, Regressions=0."""
    a = _write(tmp_path, "a.json", [_iter("img.jpeg@high", 100.0, fmt="jpeg")])
    b = _write(tmp_path, "b.json", [_iter("img.jpeg@high", 50.0, fmt="jpeg")])

    result = compare(a, b, threshold_pct=10.0, noise_floor_pct=25.0)
    assert result.exit_code == 0

    rollups = build_format_rollup(result.diffs)
    assert len(rollups) == 1
    r = rollups[0]
    assert r.fmt == "jpeg"
    assert r.n_improvements == 1, f"Expected 1 improvement, got {r.n_improvements}"
    assert r.n_regressions == 0, f"Expected 0 regressions, got {r.n_regressions}"
    # Status glyph should be ✅ (improvement) not ⚠ (noise_floor_regression)
    assert r.status == "✅", f"Expected '✅' status, got '{r.status}'"
