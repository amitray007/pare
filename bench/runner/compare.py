"""Diff two benchmark runs with statistical significance.

Pairs cases by `case_id` (e.g. `photo_perlin_medium_001.jpeg@high`).
For each pair, applies Welch's t-test on the raw wall_ms iterations
and Cohen's d on the same — both must clear thresholds for a regression
to be flagged. This combination defends against false alarms from
either large-n inflated significance or single-outlier-driven means.

When either side has fewer than 3 iterations (e.g. quick mode with 1
iteration), Welch's t-test is meaningless (p is always 1.0, d is always
0.0). In that case we fall back to a noise-floor check: flag as
regression iff |delta%| >= noise_floor_pct (default 25%).

Exit code semantics:

    0  no significant regression
    1  regression flagged in at least one case
    2  schema error (mismatched schema_version, etc.) or
       comparability error (mode mismatch between runs)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench.runner.report.json_writer import load_run
from bench.runner.report.markdown import (
    build_format_rollup,
    format_compare_label,
    render_format_rollup_table,
)
from bench.runner.stats import (
    cohens_d,
    differs_significantly,
    median,
    welch_t_test,
)

# Minimum iterations on each side before we trust the stats-based gate.
_STATS_MIN_ITERS = 3


@dataclass
class RunConditions:
    """Extracted comparability metadata from a single run."""

    mode: str
    isolate: bool
    platform: str


def _extract_conditions(run: dict[str, Any]) -> RunConditions:
    """Pull the three comparability fields out of a loaded run dict."""
    mode = run.get("mode", "unknown")
    isolate = bool(run.get("config", {}).get("isolate", False))
    platform = run.get("host", {}).get("platform", "unknown")
    return RunConditions(mode=mode, isolate=isolate, platform=platform)


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
    # True when the noise-floor path was used instead of the stats path.
    iters_low_power: bool = False

    @property
    def label(self) -> str:
        if self.iters_low_power:
            # Noise-floor path: no stats, just delta% vs noise_floor_pct.
            # Distinguish direction so improvements are not miscounted as regressions.
            if not self.threshold_breach:
                return "noise_floor_ok"
            return "noise_floor_regression" if self.delta_pct > 0 else "improvement"
        # Stats path: regression/improvement only when BOTH conditions hold.
        if self.significant and self.threshold_breach:
            return "significant" if self.delta_pct > 0 else "improvement"
        # Delta exists but either not statistically significant or below threshold.
        return "below_threshold" if self.threshold_breach else "ok"


@dataclass
class CompareResult:
    a_path: Path
    b_path: Path
    diffs: list[CaseDiff] = field(default_factory=list)
    only_in_a: list[str] = field(default_factory=list)
    only_in_b: list[str] = field(default_factory=list)
    threshold_pct: float = 10.0
    noise_floor_pct: float = 25.0
    alpha: float = 0.05
    # Conditions from each run — populated by compare().
    a_conditions: RunConditions | None = None
    b_conditions: RunConditions | None = None

    @property
    def regressions(self) -> list[CaseDiff]:
        """Cases flagged by the stats gate (significant AND threshold_breach AND positive delta)."""
        return [
            d
            for d in self.diffs
            if not d.iters_low_power and d.significant and d.threshold_breach and d.delta_pct > 0
        ]

    @property
    def noise_floor_flags(self) -> list[CaseDiff]:
        """Cases flagged by the noise-floor gate (low-power path, |delta%| >= noise_floor_pct)."""
        return [
            d for d in self.diffs if d.iters_low_power and d.threshold_breach and d.delta_pct > 0
        ]

    @property
    def improvements(self) -> list[CaseDiff]:
        out: list[CaseDiff] = []
        for d in self.diffs:
            if not d.threshold_breach or d.delta_pct >= 0:
                continue
            if d.iters_low_power or d.significant:
                out.append(d)
        return out

    @property
    def exit_code(self) -> int:
        return 1 if (self.regressions or self.noise_floor_flags) else 0


def _wall_iterations_by_case(run: dict[str, Any]) -> dict[str, list[float]]:
    by_case: dict[str, list[float]] = {}
    for it in run["iterations"]:
        if "error" in it:
            continue
        by_case.setdefault(it["case_id"], []).append(it["measurement"]["wall_ms"])
    return by_case


class ModeMismatchError(ValueError):
    """Raised when two runs use incompatible modes and the caller did not opt in."""

    pass


def compare(
    a_path: Path,
    b_path: Path,
    *,
    threshold_pct: float = 10.0,
    noise_floor_pct: float = 25.0,
    alpha: float = 0.05,
    min_effect_size: float = 0.5,
    allow_mismatched_mode: bool = False,
) -> CompareResult:
    """Welch's-t + Cohen's-d diff between two runs.

    When either side has fewer than 3 iterations, falls back to a pure
    |delta%| check at the higher noise_floor_pct threshold instead of the
    stats-backed gate.

    Before computing diffs, validates that the two runs are comparable:
    - mode must match (e.g. both "quick" or both "timing"); mismatches raise
      ModeMismatchError unless allow_mismatched_mode=True.
    - isolate flag and platform differences are surfaced via result.a_conditions /
      result.b_conditions; callers are responsible for emitting warnings.

    Raises:
        ModeMismatchError: if modes differ and allow_mismatched_mode is False.
    """
    a_run = load_run(a_path)
    b_run = load_run(b_path)

    a_cond = _extract_conditions(a_run)
    b_cond = _extract_conditions(b_run)

    # --- mode check (hard error by default) ---
    if a_cond.mode != b_cond.mode and not allow_mismatched_mode:
        raise ModeMismatchError(
            f"baseline mode={a_cond.mode!r} but head mode={b_cond.mode!r} — "
            f"wall_ms is not comparable across modes. "
            f"Pass --allow-mismatched-mode to override."
        )

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

        low_power = min(len(a_walls), len(b_walls)) < _STATS_MIN_ITERS

        if low_power:
            # Skip stats — they're meaningless with <3 iterations.
            p = 1.0
            d = 0.0
            significant = False
            threshold_breach = abs(delta_pct) >= noise_floor_pct
        else:
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
                iters_low_power=low_power,
            )
        )

    return CompareResult(
        a_path=a_path,
        b_path=b_path,
        diffs=diffs,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        threshold_pct=threshold_pct,
        noise_floor_pct=noise_floor_pct,
        alpha=alpha,
        a_conditions=a_cond,
        b_conditions=b_cond,
    )


def _render_conditions_section(result: CompareResult) -> str:
    """Render a 'compare conditions' header so readers know what they're looking at."""
    a = result.a_conditions
    b = result.b_conditions
    if a is None or b is None:
        return ""

    lines: list[str] = []
    lines.append("## Compare conditions")
    lines.append("")
    lines.append("| | baseline | head |")
    lines.append("|---|---|---|")
    lines.append(f"| **file** | `{result.a_path.name}` | `{result.b_path.name}` |")
    lines.append(f"| **mode** | `{a.mode}` | `{b.mode}` |")
    lines.append(f"| **isolate** | `{a.isolate}` | `{b.isolate}` |")
    lines.append(f"| **platform** | `{a.platform}` | `{b.platform}` |")

    warnings: list[str] = []
    if a.mode != b.mode:
        warnings.append(
            f"Mode mismatch (`{a.mode}` vs `{b.mode}`) — wall_ms is not directly comparable."
        )
    if a.isolate != b.isolate:
        warnings.append(
            f"Isolate mismatch (`{a.isolate}` vs `{b.isolate}`) — "
            f"isolated runs carry ~200-400ms/iter subprocess overhead."
        )
    if a.platform != b.platform:
        warnings.append(
            f"Platform mismatch (`{a.platform}` vs `{b.platform}`) — "
            f"Pillow/zlib version drift across OSes may affect timings."
        )

    if warnings:
        lines.append("")
        for w in warnings:
            lines.append(f"> **WARNING**: {w}")

    return "\n".join(lines)


def render_compare_markdown(result: CompareResult) -> str:
    lines: list[str] = []
    lines.append(f"# Bench compare: {result.a_path.name} → {result.b_path.name}")
    lines.append("")
    lines.append(
        f"_threshold={result.threshold_pct}%, noise-floor={result.noise_floor_pct}%, "
        f"α={result.alpha}, cases compared={len(result.diffs)}, "
        f"regressions={len(result.regressions)}, "
        f"noise_floor_flags={len(result.noise_floor_flags)}, "
        f"improvements={len(result.improvements)}_"
    )
    lines.append("")

    # Always include the conditions section so readers know what they're looking at.
    cond_section = _render_conditions_section(result)
    if cond_section:
        lines.append(cond_section)
        lines.append("")

    if result.only_in_a or result.only_in_b:
        lines.append(
            f"_only in baseline={len(result.only_in_a)}, only in head={len(result.only_in_b)}_"
        )
        lines.append("")

    if not result.diffs:
        lines.append("_No common cases to compare._")
        return "\n".join(lines)

    # Per-format rollup table (scannable summary).
    rollups = build_format_rollup(result.diffs)
    lines.append(render_format_rollup_table(rollups))
    lines.append("")

    # Per-case detail collapsed into a <details> block.
    n = len(result.diffs)
    lines.append("<details>")
    lines.append(f"<summary>Per-case detail ({n} cases)</summary>")
    lines.append("")
    lines.append("| case_id | baseline | head | Δ% | p | d | label |")
    lines.append("|---|---|---|---|---|---|---|")

    sorted_diffs = sorted(result.diffs, key=lambda d: -abs(d.delta_pct))
    for d in sorted_diffs:
        display_label = format_compare_label(d.label)
        lines.append(
            f"| `{d.case_id}` | {d.baseline_median_ms:.1f}ms | "
            f"{d.head_median_ms:.1f}ms | {d.delta_pct:+.1f}% | "
            f"{d.p_value:.3f} | {d.cohens_d:+.2f} | {display_label} |"
        )
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)
