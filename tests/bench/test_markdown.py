"""Unit tests for the per-format rollup helpers in bench/runner/report/markdown.py.

Covers:
  - Format extraction from representative case_ids.
  - Grouping across multiple formats.
  - Status derivation (❌ / ⚠ / ✅ / ~) and the precedence rules.
  - Sort order: ❌ before ~; within ❌, higher worst_delta first.
  - render_compare_markdown emits rollup table and <details> block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.runner.compare import CaseDiff, CompareResult, render_compare_markdown  # noqa: E402
from bench.runner.report.markdown import (
    _extract_format,
    build_format_rollup,
    render_format_rollup_table,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _diff(
    case_id: str,
    delta_pct: float,
    *,
    significant: bool = False,
    threshold_breach: bool = False,
    iters_low_power: bool = False,
    cohens_d: float = 0.0,
    p_value: float = 1.0,
) -> CaseDiff:
    """Build a minimal CaseDiff for testing rollup logic."""
    baseline = 100.0
    head = baseline * (1 + delta_pct / 100)
    return CaseDiff(
        case_id=case_id,
        baseline_median_ms=baseline,
        head_median_ms=head,
        delta_pct=delta_pct,
        p_value=p_value,
        cohens_d=cohens_d,
        significant=significant,
        threshold_breach=threshold_breach,
        iters_low_power=iters_low_power,
    )


def _sig_regression(case_id: str, delta_pct: float) -> CaseDiff:
    """Significant regression (label == 'significant')."""
    return _diff(case_id, delta_pct, significant=True, threshold_breach=True, p_value=0.01)


def _noise_regression(case_id: str, delta_pct: float) -> CaseDiff:
    """Noise-floor regression (label == 'noise_floor_regression')."""
    return _diff(case_id, delta_pct, iters_low_power=True, threshold_breach=True)


def _improvement(case_id: str, delta_pct: float) -> CaseDiff:
    """Improvement (label == 'improvement')."""
    assert delta_pct < 0
    return _diff(case_id, delta_pct, significant=True, threshold_breach=True, p_value=0.01)


def _flat(case_id: str, delta_pct: float = 1.0) -> CaseDiff:
    """Below-threshold / ok (label == 'ok' or 'below_threshold')."""
    return _diff(case_id, delta_pct)


# ---------------------------------------------------------------------------
# _extract_format
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case_id, expected",
    [
        ("photo_perlin_tiny_heic.heic@high", "heic"),
        ("path_text_on_flat_small_jpeg.jpeg@medium", "jpeg"),
        ("animated_translation_medium_apng.apng@high", "apng"),
        ("graphic_palette_tiny_tiff.tiff@low", "tiff"),
        ("simple.png@low", "png"),
        ("no_at_sign.png", "png"),  # no @ separator — still works on the left part
        ("broken", "unknown"),  # no dot at all → unknown
        ("", "unknown"),  # empty string → unknown
    ],
)
def test_extract_format(case_id: str, expected: str):
    assert _extract_format(case_id) == expected


# ---------------------------------------------------------------------------
# build_format_rollup — grouping and field values
# ---------------------------------------------------------------------------


def test_rollup_groups_by_format():
    diffs = [
        _flat("a.jpeg@high"),
        _flat("b.jpeg@medium"),
        _flat("c.png@high"),
    ]
    rollups = build_format_rollup(diffs)
    fmts = {r.fmt for r in rollups}
    assert fmts == {"jpeg", "png"}
    jpeg_r = next(r for r in rollups if r.fmt == "jpeg")
    assert jpeg_r.n_cases == 2
    png_r = next(r for r in rollups if r.fmt == "png")
    assert png_r.n_cases == 1


def test_rollup_counts_regressions_and_improvements():
    diffs = [
        _sig_regression("a.jpeg@high", 30.0),
        _improvement("b.jpeg@medium", -15.0),
        _flat("c.jpeg@low", 2.0),
    ]
    rollups = build_format_rollup(diffs)
    assert len(rollups) == 1
    r = rollups[0]
    assert r.n_regressions == 1
    assert r.n_improvements == 1
    assert r.n_cases == 3


def test_rollup_median_and_worst():
    diffs = [
        _flat("a.jpeg@high", 5.0),
        _flat("b.jpeg@medium", 10.0),
        _flat("c.jpeg@low", 15.0),
    ]
    rollups = build_format_rollup(diffs)
    r = rollups[0]
    assert r.median_delta_pct == pytest.approx(10.0)
    assert r.worst_delta_pct == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# Status derivation
# ---------------------------------------------------------------------------


def test_status_significant_is_red_x():
    """Any significant regression → ❌, even when mixed with improvements."""
    diffs = [
        _sig_regression("a.jpeg@high", 30.0),
        _improvement("b.jpeg@medium", -15.0),
        _flat("c.jpeg@low"),
    ]
    r = build_format_rollup(diffs)[0]
    assert r.status == "❌"
    assert r.worst_label == "significant"


def test_status_noise_floor_only_is_warning():
    """Only noise_floor_regression labels → ⚠."""
    diffs = [
        _noise_regression("a.png@high", 30.0),
        _noise_regression("b.png@medium", 28.0),
    ]
    r = build_format_rollup(diffs)[0]
    assert r.status == "⚠"
    assert r.worst_label == "noise_floor_regression"


def test_status_improvement_only_is_checkmark():
    """Only improvement labels → ✅."""
    diffs = [
        _improvement("a.png@high", -20.0),
        _improvement("b.png@medium", -5.0),
    ]
    r = build_format_rollup(diffs)[0]
    assert r.status == "✅"
    assert r.worst_label == "improvement"


def test_status_flat_only_is_tilde():
    """All ok / below_threshold → ~."""
    diffs = [
        _flat("a.png@high", 1.0),
        _flat("b.png@medium", -0.5),
    ]
    r = build_format_rollup(diffs)[0]
    assert r.status == "~"


# ---------------------------------------------------------------------------
# Sort order
# ---------------------------------------------------------------------------


def test_sort_red_x_before_tilde():
    """❌ formats appear before ~ formats in the rollup."""
    diffs = [
        _flat("a.png@high"),
        _sig_regression("b.jpeg@high", 40.0),
    ]
    rollups = build_format_rollup(diffs)
    assert rollups[0].fmt == "jpeg"
    assert rollups[0].status == "❌"
    assert rollups[1].fmt == "png"
    assert rollups[1].status == "~"


def test_sort_within_red_x_by_worst_delta_descending():
    """Within ❌, the format with the larger worst_delta_pct comes first."""
    diffs = [
        _sig_regression("a.jpeg@high", 20.0),
        _sig_regression("b.jpeg@medium", 60.0),  # jpeg worst = 60
        _sig_regression("c.webp@high", 35.0),  # webp worst = 35
    ]
    rollups = build_format_rollup(diffs)
    assert rollups[0].fmt == "jpeg"  # worst = 60
    assert rollups[1].fmt == "webp"  # worst = 35


# ---------------------------------------------------------------------------
# render_compare_markdown integration
# ---------------------------------------------------------------------------


def _make_result(diffs: list[CaseDiff]) -> CompareResult:
    return CompareResult(
        a_path=Path("baseline.json"),
        b_path=Path("head.json"),
        diffs=diffs,
    )


def test_render_compare_markdown_contains_rollup_table():
    diffs = [
        _sig_regression("a.jpeg@high", 30.0),
        _flat("b.png@high"),
    ]
    md = render_compare_markdown(_make_result(diffs))
    assert "## Per-format summary" in md
    assert "| Format |" in md
    assert "`jpeg`" in md
    assert "`png`" in md


def test_render_compare_markdown_wraps_detail_in_details_block():
    diffs = [
        _flat("a.jpeg@high", 2.0),
        _flat("b.jpeg@medium", 3.0),
    ]
    md = render_compare_markdown(_make_result(diffs))
    assert "<details>" in md
    assert "<summary>Per-case detail (2 cases)</summary>" in md
    assert "</details>" in md


def test_render_compare_markdown_detail_count_matches_diffs():
    diffs = [_flat(f"a_{i}.jpeg@high", float(i)) for i in range(7)]
    md = render_compare_markdown(_make_result(diffs))
    assert "Per-case detail (7 cases)" in md


def test_render_compare_markdown_rollup_before_details():
    """The rollup table must appear before the <details> block."""
    diffs = [_flat("a.jpeg@high", 2.0)]
    md = render_compare_markdown(_make_result(diffs))
    rollup_pos = md.index("## Per-format summary")
    details_pos = md.index("<details>")
    assert rollup_pos < details_pos


def test_render_compare_markdown_no_diffs_skips_rollup():
    """With no common diffs, neither rollup nor details should appear."""
    md = render_compare_markdown(_make_result([]))
    assert "No common cases to compare" in md
    assert "## Per-format summary" not in md
    assert "<details>" not in md


def test_render_format_rollup_table_delta_formatting():
    """Deltas should render with explicit sign and one decimal place."""
    from bench.runner.report.markdown import FormatRollup

    rollup = FormatRollup(
        fmt="heic",
        n_cases=9,
        median_delta_pct=35.4,
        worst_delta_pct=67.5,
        n_regressions=9,
        n_improvements=0,
        worst_label="significant",
        status="❌",
    )
    table = render_format_rollup_table([rollup])
    assert "+35.4%" in table
    assert "+67.5%" in table
    assert "❌" in table


def test_render_format_rollup_table_negative_delta():
    from bench.runner.report.markdown import FormatRollup

    rollup = FormatRollup(
        fmt="png",
        n_cases=3,
        median_delta_pct=-2.3,
        worst_delta_pct=0.5,
        n_regressions=0,
        n_improvements=2,
        worst_label="improvement",
        status="✅",
    )
    table = render_format_rollup_table([rollup])
    assert "-2.3%" in table
    assert "+0.5%" in table
