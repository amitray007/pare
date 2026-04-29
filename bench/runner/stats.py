"""Statistical primitives for benchmark analysis.

Pure-Python implementations to avoid pulling scipy into the bench
runtime — bench is a dev tool, but minimizing dependencies keeps it
fast to set up on a fresh machine.

What's here:

- `percentile`, `median`, `mad` (median absolute deviation), `mean`, `stdev`
- `welch_t_test` — Welch's t-test for unequal-variance comparison.
  P-values use a normal-distribution approximation, which is accurate
  for df >= 30 and slightly liberal for smaller df. For n=5 iterations
  (df ~8), a confirmed regression should show effect size > 0.5 in
  addition to p < alpha — `differs_significantly` combines both.
- `cohens_d` — effect-size measurement; robust for small n.
- `CaseStats` — per-case aggregate consumed by reporters and `compare`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy's default.

    `p` is in [0, 100]. Empty input returns 0.0.
    """
    if not data:
        return 0.0
    if not 0 <= p <= 100:
        raise ValueError(f"percentile p={p} out of range [0, 100]")

    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    rank = (p / 100.0) * (n - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_data[lo]
    weight = rank - lo
    return sorted_data[lo] * (1 - weight) + sorted_data[hi] * weight


def median(data: list[float]) -> float:
    return percentile(data, 50.0)


def mad(data: list[float]) -> float:
    """Median absolute deviation — robust scale estimator.

    Less sensitive to outliers than stdev. pyperf recommends MAD as the
    primary spread metric when results aren't stable, which is common
    for subprocess-heavy workloads with OS scheduler jitter.
    """
    if not data:
        return 0.0
    m = median(data)
    return median([abs(x - m) for x in data])


def mean(data: list[float]) -> float:
    return statistics.fmean(data) if data else 0.0


def stdev(data: list[float]) -> float:
    """Sample standard deviation. Returns 0 for n < 2."""
    return statistics.stdev(data) if len(data) >= 2 else 0.0


def cohens_d(a: list[float], b: list[float]) -> float:
    """Effect size with pooled standard deviation.

    |d| < 0.2  — negligible
    |d| < 0.5  — small
    |d| < 0.8  — moderate
    |d| >= 0.8 — large

    For benchmark regression detection, |d| >= 0.5 plus p < 0.05 is a
    reasonable bar; either alone produces too many false alarms.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a, mean_b = mean(a), mean(b)
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled_var <= 0:
        return 0.0
    return (mean_a - mean_b) / math.sqrt(pooled_var)


def welch_t_test(
    a: list[float],
    b: list[float],
) -> tuple[float, float, float]:
    """Welch's t-test for two samples with unequal variances.

    Returns (t_statistic, p_value, df_welch_satterthwaite).

    The p-value uses a normal-distribution approximation, accurate for
    df >= 30 and slightly liberal below that. Pair this with `cohens_d`
    via `differs_significantly` to avoid declaring noise as signal at
    small n.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0, 1.0, 0.0

    mean_a, mean_b = mean(a), mean(b)
    var_a = stdev(a) ** 2
    var_b = stdev(b) ** 2

    se_squared = var_a / n_a + var_b / n_b
    if se_squared <= 0:
        return 0.0, 1.0, 0.0

    se = math.sqrt(se_squared)
    t_stat = (mean_a - mean_b) / se

    df_num = se_squared**2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else max(n_a + n_b - 2, 1)

    # Normal-CDF approximation: 2 * (1 - Phi(|t|)).
    # statistics.NormalDist().cdf is in stdlib (Python >= 3.8).
    p = 2.0 * (1.0 - statistics.NormalDist().cdf(abs(t_stat)))
    p = max(0.0, min(1.0, p))
    return t_stat, p, df


def differs_significantly(
    a: list[float],
    b: list[float],
    *,
    alpha: float = 0.05,
    min_effect_size: float = 0.5,
) -> bool:
    """Combine Welch's t-test and Cohen's d to flag real regressions.

    Both conditions must hold:
    1. Welch's p-value < alpha (the means differ beyond noise), AND
    2. |Cohen's d| >= min_effect_size (the difference is meaningful).

    The conjunction prevents false alarms from large p but trivial
    effect (huge n inflates significance) and from large effect but
    huge variance (one outlier moving the mean).
    """
    if len(a) < 2 or len(b) < 2:
        return False
    _, p, _ = welch_t_test(a, b)
    if p >= alpha:
        return False
    return abs(cohens_d(a, b)) >= min_effect_size


@dataclass
class CaseStats:
    """Aggregate over multiple iterations of one (case, preset) cell."""

    case_id: str
    bucket: str
    format: str
    preset: str
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    median_ms: float
    mad_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    children_cpu_p50_ms: float
    children_cpu_p95_ms: float
    children_peak_rss_p95_kb: int
    parent_peak_rss_p95_kb: int
    parallelism_p50: float
    reduction_pct: float = 0.0
    method: str = ""
    py_peak_alloc_p95_kb: int | None = None
    raw_wall_ms: list[float] = field(default_factory=list)


def summarize_iterations(
    case_id: str,
    bucket: str,
    fmt: str,
    preset: str,
    iter_data: list[dict[str, float]],
    *,
    reduction_pct: float = 0.0,
    method: str = "",
) -> CaseStats:
    """Roll up a list of per-iteration measurement dicts into CaseStats.

    `iter_data` is a list where each element has at least `wall_ms`,
    `children_cpu_ms`, `children_peak_rss_kb`, `parent_peak_rss_kb`,
    `parallelism`. Optional: `py_peak_alloc_kb`.
    """
    if not iter_data:
        raise ValueError("cannot summarize zero iterations")

    walls = [it["wall_ms"] for it in iter_data]
    child_cpu = [it["children_cpu_ms"] for it in iter_data]
    child_rss = [it["children_peak_rss_kb"] for it in iter_data]
    parent_rss = [it["parent_peak_rss_kb"] for it in iter_data]
    parallelisms = [it["parallelism"] for it in iter_data]
    py_alloc = [it.get("py_peak_alloc_kb") for it in iter_data]
    py_alloc_present = [v for v in py_alloc if v is not None]

    return CaseStats(
        case_id=case_id,
        bucket=bucket,
        format=fmt,
        preset=preset,
        iterations=len(iter_data),
        p50_ms=percentile(walls, 50),
        p95_ms=percentile(walls, 95),
        p99_ms=percentile(walls, 99),
        median_ms=median(walls),
        mad_ms=mad(walls),
        mean_ms=mean(walls),
        stdev_ms=stdev(walls),
        min_ms=min(walls),
        max_ms=max(walls),
        children_cpu_p50_ms=percentile(child_cpu, 50),
        children_cpu_p95_ms=percentile(child_cpu, 95),
        children_peak_rss_p95_kb=int(percentile(child_rss, 95)),
        parent_peak_rss_p95_kb=int(percentile(parent_rss, 95)),
        parallelism_p50=percentile(parallelisms, 50),
        reduction_pct=reduction_pct,
        method=method,
        py_peak_alloc_p95_kb=int(percentile(py_alloc_present, 95)) if py_alloc_present else None,
        raw_wall_ms=list(walls),
    )
