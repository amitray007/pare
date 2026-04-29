"""Stats layer tests."""

from __future__ import annotations

import math

import pytest

from bench.runner.stats import (
    CaseStats,
    cohens_d,
    differs_significantly,
    mad,
    mean,
    median,
    percentile,
    stdev,
    summarize_iterations,
    welch_t_test,
)

# --- percentile / median --------------------------------------------------


def test_percentile_at_endpoints():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 5.0


def test_percentile_at_50_is_median():
    assert percentile([1.0, 3.0, 5.0], 50) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5


def test_percentile_uses_linear_interpolation():
    """Matches numpy's default (linear interp between order statistics)."""
    data = [10.0, 20.0, 30.0, 40.0, 50.0]
    # rank = 0.95 * 4 = 3.8  -> between data[3]=40 and data[4]=50
    # 40 * (1-0.8) + 50 * 0.8 = 8 + 40 = 48
    assert math.isclose(percentile(data, 95), 48.0, abs_tol=1e-6)


def test_percentile_empty_returns_zero():
    assert percentile([], 50) == 0.0


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0], -10.0)
    with pytest.raises(ValueError):
        percentile([1.0], 101.0)


def test_median_matches_p50():
    data = [1.5, 2.5, 4.0, 8.0, 16.0]
    assert median(data) == percentile(data, 50)


# --- MAD ------------------------------------------------------------------


def test_mad_zero_for_constant_data():
    assert mad([5.0, 5.0, 5.0]) == 0.0


def test_mad_robust_to_outlier():
    """One huge outlier shouldn't hugely inflate MAD (unlike stdev)."""
    clean = [10.0, 11.0, 12.0, 13.0, 14.0]
    with_outlier = clean + [10000.0]
    mad_clean = mad(clean)
    mad_outlier = mad(with_outlier)
    # MAD should change by less than a factor of 3 even with a 1000x outlier
    assert mad_outlier < mad_clean * 3
    # stdev would explode in the same scenario
    assert stdev(with_outlier) > stdev(clean) * 100


def test_mad_empty_returns_zero():
    assert mad([]) == 0.0


# --- mean / stdev ---------------------------------------------------------


def test_mean_empty_returns_zero():
    assert mean([]) == 0.0


def test_stdev_below_two_samples_returns_zero():
    assert stdev([]) == 0.0
    assert stdev([5.0]) == 0.0


def test_stdev_matches_known_value():
    # population variance computed by hand: data = [2,4,4,4,5,5,7,9]
    # mean=5, sample variance = 32/7 ≈ 4.571, stdev ≈ 2.138
    assert math.isclose(stdev([2, 4, 4, 4, 5, 5, 7, 9]), 2.13808993529, abs_tol=1e-6)


# --- Cohen's d ------------------------------------------------------------


def test_cohens_d_zero_for_identical_distributions():
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = list(a)
    assert cohens_d(a, b) == 0.0


def test_cohens_d_large_for_well_separated_means():
    a = [10.0, 10.5, 9.5, 10.2, 10.1]
    b = [20.0, 20.5, 19.5, 20.2, 20.1]
    d = cohens_d(a, b)
    assert d < -2.0, f"expected large negative effect size, got {d}"


def test_cohens_d_returns_zero_for_too_few_samples():
    assert cohens_d([1.0], [2.0]) == 0.0
    assert cohens_d([], [1.0, 2.0]) == 0.0


# --- Welch's t-test -------------------------------------------------------


def test_welch_returns_high_p_for_overlapping_distributions():
    a = [10.0, 11.0, 9.0, 10.5, 9.5]
    b = [10.2, 10.8, 9.7, 10.1, 9.9]
    _, p, _ = welch_t_test(a, b)
    assert p > 0.3, f"expected high p, got {p}"


def test_welch_returns_low_p_for_well_separated_distributions():
    a = [10.0, 10.5, 9.5, 10.2, 10.1]
    b = [20.0, 20.5, 19.5, 20.2, 20.1]
    _, p, _ = welch_t_test(a, b)
    assert p < 0.001


def test_welch_satterthwaite_df_in_reasonable_range():
    a = [10.0] * 8
    b = [10.0] * 8
    _, _, df = welch_t_test(a, b)
    # With identical inputs, var=0 -> we short-circuit; df=0
    assert df == 0.0

    a = [9.0, 11.0, 10.0, 10.5, 9.5]
    b = [12.0, 13.0, 11.5, 12.5, 12.0]
    _, _, df = welch_t_test(a, b)
    # With n=5 each, df should be ~5..8
    assert 3 <= df <= 9


def test_welch_handles_n_below_two():
    _, p, _ = welch_t_test([1.0], [2.0, 3.0])
    assert p == 1.0


def test_welch_handles_zero_variance():
    """Identical samples — welch should return non-significant."""
    _, p, _ = welch_t_test([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
    assert p == 1.0


# --- differs_significantly ------------------------------------------------


def test_differs_significantly_requires_both_p_and_effect_size():
    """Effect size large but p > alpha (small n + small variance)."""
    a = [10.0, 10.0]
    b = [12.0, 12.0]
    # n=2 each gives df near 1, p high; not significant despite huge d
    assert not differs_significantly(a, b)


def test_differs_significantly_yes_for_well_separated():
    a = [10.0, 10.5, 9.5, 10.2, 10.1, 9.9, 10.3]
    b = [15.0, 15.5, 14.5, 15.2, 15.1, 14.9, 15.3]
    assert differs_significantly(a, b)


def test_differs_significantly_no_for_overlapping_distributions():
    a = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 10.8]
    b = [10.2, 10.8, 9.7, 10.1, 9.9, 10.4, 10.0]
    assert not differs_significantly(a, b)


def test_differs_significantly_threshold_is_tunable():
    """A regression with small effect size should be filtered with stricter `min_effect_size`."""
    a = [10.0, 10.1, 9.9, 10.05, 9.95, 10.02, 9.98] * 5
    b = [10.5, 10.6, 10.4, 10.55, 10.45, 10.52, 10.48] * 5
    # default 0.5 threshold: this is around d≈4-5 → significant
    assert differs_significantly(a, b)
    # super-strict threshold rejects it
    assert not differs_significantly(a, b, min_effect_size=10.0)


# --- summarize_iterations -------------------------------------------------


def _iter(wall: float, child: float = 5.0, rss: int = 1024, par: float = 1.2) -> dict:
    return {
        "wall_ms": wall,
        "children_cpu_ms": child,
        "children_peak_rss_kb": rss,
        "parent_peak_rss_kb": rss * 2,
        "parallelism": par,
    }


def test_summarize_rejects_empty_input():
    with pytest.raises(ValueError):
        summarize_iterations("c", "small", "png", "high", [])


def test_summarize_aggregates_walls_and_percentiles():
    walls = [10.0, 12.0, 11.0, 13.0, 14.0]
    iters = [_iter(w) for w in walls]
    s = summarize_iterations("c", "small", "png", "high", iters)
    assert s.iterations == 5
    assert math.isclose(s.median_ms, 12.0, abs_tol=1e-6)
    assert math.isclose(s.mean_ms, 12.0, abs_tol=1e-6)
    assert s.min_ms == 10.0
    assert s.max_ms == 14.0
    assert s.raw_wall_ms == walls


def test_summarize_carries_through_metadata():
    s = summarize_iterations(
        "case_a@high",
        "medium",
        "jpeg",
        "high",
        [_iter(10.0), _iter(11.0)],
        reduction_pct=64.5,
        method="mozjpeg+jpegli",
    )
    assert s.case_id == "case_a@high"
    assert s.bucket == "medium"
    assert s.format == "jpeg"
    assert s.preset == "high"
    assert s.reduction_pct == 64.5
    assert s.method == "mozjpeg+jpegli"


def test_summarize_handles_optional_py_alloc():
    iters = [
        {**_iter(10.0), "py_peak_alloc_kb": 4096},
        {**_iter(11.0), "py_peak_alloc_kb": 5120},
    ]
    s = summarize_iterations("c", "small", "png", "high", iters)
    assert s.py_peak_alloc_p95_kb is not None
    assert s.py_peak_alloc_p95_kb >= 4096


def test_summarize_omits_py_alloc_when_absent():
    s = summarize_iterations("c", "small", "png", "high", [_iter(10.0), _iter(11.0)])
    assert s.py_peak_alloc_p95_kb is None


def test_case_stats_dataclass_round_trip():
    s = CaseStats(
        case_id="x",
        bucket="small",
        format="png",
        preset="high",
        iterations=1,
        p50_ms=1.0,
        p95_ms=1.0,
        p99_ms=1.0,
        median_ms=1.0,
        mad_ms=0.0,
        mean_ms=1.0,
        stdev_ms=0.0,
        min_ms=1.0,
        max_ms=1.0,
        children_cpu_p50_ms=1.0,
        children_cpu_p95_ms=1.0,
        children_peak_rss_p95_kb=1024,
        parent_peak_rss_p95_kb=2048,
        parallelism_p50=1.2,
    )
    assert s.case_id == "x"
    assert s.iterations == 1
