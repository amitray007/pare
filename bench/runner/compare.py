"""Diff two benchmark runs with statistical significance.

Pairs cases by `case_id` (e.g. `photo_perlin_medium_001.jpeg@high`).
For each pair, applies Welch's t-test on the raw wall_ms iterations
and Cohen's d on the same — both must clear thresholds for a regression
to be flagged. This combination defends against false alarms from
either large-n inflated significance or single-outlier-driven means.

Exit code semantics:

    0  no significant regression
    1  regression flagged in at least one case
    2  schema error (mismatched schema_version, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.runner.report.json_writer import load_run
from bench.runner.stats import (
    cohens_d,
    differs_significantly,
    median,
    welch_t_test,
)


@dataclass
class CaseDiff:
    case_id: str
    baseline_median_ms: float
    head_median_ms: float
    delta_pct: float
    p_value: float
    cohens_d: float
    significant: bool
    threshold_breach: bool

    @property
    def label(self) -> str:
        if not self.significant:
            return "~"
        if self.threshold_breach:
            return "REGRESSION" if self.delta_pct > 0 else "IMPROVEMENT"
        return "drift"


@dataclass
class CompareResult:
    a_path: Path
    b_path: Path
    diffs: list[CaseDiff] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    threshold_pct: float = 10.0
    alpha: float = 0.05

    @property
    def regressions(self) -> list[CaseDiff]:
        return [d for d in self.diffs if d.significant and d.threshold_breach and d.delta_pct > 0]

    @property
    def improvements(self) -> list[CaseDiff]:
        return [d for d in self.diffs if d.significant and d.threshold_breach and d.delta_pct < 0]

    @property
    def exit_code(self) -> int:
        return 1 if self.regressions else 0


def _wall_iterations_by_case(run: dict[str, Any]) -> dict[str, list[float]]:
    by_case: dict[str, list[float]] = {}
    for it in run["iterations"]:
        if "error" in it:
            continue
        by_case.setdefault(it["case_id"], []).append(it["measurement"]["wall_ms"])
    return by_case


def compare(
    a_path: Path,
    b_path: Path,
    *,
    threshold_pct: float = 10.0,
    alpha: float = 0.05,
    min_effect_size: float = 0.5,
) -> CompareResult:
    """Welch's-t + Cohen's-d diff between two runs."""
    a_run = load_run(a_path)
    b_run = load_run(b_path)

    a_by_case = _wall_iterations_by_case(a_run)
    b_by_case = _wall_iterations_by_case(b_run)

    a_keys = set(a_by_case)
    b_keys = set(b_by_case)
    only_in_a = sorted(a_keys - b_keys)
    only_in_b = sorted(b_keys - a_keys)
    common = sorted(a_keys & b_keys)

    diffs: list[CaseDiff] = []
    for case_id in common:
        a_walls = a_by_case[case_id]
        b_walls = b_by_case[case_id]
        a_med = median(a_walls)
        b_med = median(b_walls)
        delta_pct = ((b_med - a_med) / a_med * 100.0) if a_med > 0 else 0.0
        _, p, _ = welch_t_test(a_walls, b_walls)
        d = cohens_d(a_walls, b_walls)
        significant = differs_significantly(
            a_walls, b_walls, alpha=alpha, min_effect_size=min_effect_size
        )
        threshold_breach = abs(delta_pct) >= threshold_pct
        diffs.append(
            CaseDiff(
                case_id=case_id,
                baseline_median_ms=a_med,
                head_median_ms=b_med,
                delta_pct=delta_pct,
                p_value=p,
                cohens_d=d,
                significant=significant,
                threshold_breach=threshold_breach,
            )
        )

    return CompareResult(
        a_path=a_path,
        b_path=b_path,
        diffs=diffs,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        threshold_pct=threshold_pct,
        alpha=alpha,
    )


def render_compare_markdown(result: CompareResult) -> str:
    lines: list[str] = []
    lines.append(f"# Bench compare: {result.a_path.name} → {result.b_path.name}")
    lines.append("")
    lines.append(
        f"_threshold={result.threshold_pct}%, α={result.alpha}, "
        f"cases compared={len(result.diffs)}, regressions={len(result.regressions)}, "
        f"improvements={len(result.improvements)}_"
    )
    lines.append("")

    if result.only_in_a or result.only_in_b:
        lines.append(
            f"_only in baseline={len(result.only_in_a)}, only in head={len(result.only_in_b)}_"
        )
        lines.append("")

    if not result.diffs:
        lines.append("_No common cases to compare._")
        return "\n".join(lines)

    lines.append("| case_id | baseline | head | Δ% | p | d | label |")
    lines.append("|---|---|---|---|---|---|---|")

    sorted_diffs = sorted(result.diffs, key=lambda d: -abs(d.delta_pct))
    for d in sorted_diffs:
        lines.append(
            f"| `{d.case_id}` | {d.baseline_median_ms:.1f}ms | "
            f"{d.head_median_ms:.1f}ms | {d.delta_pct:+.1f}% | "
            f"{d.p_value:.3f} | {d.cohens_d:+.2f} | **{d.label}** |"
        )
    return "\n".join(lines)
