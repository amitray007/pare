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
    _extract_format,
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
class CompressionDiff:
    case_id: str
    baseline_method: str
    head_method: str
    baseline_reduction_pct: float
    head_reduction_pct: float
    reduction_delta_pp: float  # head - baseline (signed)
    baseline_optimized_size: int
    head_optimized_size: int
    size_delta_pct: float  # (head - baseline) / baseline * 100
    method_downgraded_to_none: bool  # baseline had method != "none", head has "none"
    reduction_regressed: bool  # reduction_delta_pp <= -reduction_threshold_pp
    size_regressed: bool  # size_delta_pct >= size_threshold_pct
    threshold_breach: bool  # any of the three above


@dataclass
class EstimationDiff:
    case_id: str
    baseline_path: str  # "exact" | "direct_encode_sample" | "generic_fallback_sample"
    head_path: str
    baseline_size_rel_error_pct: float  # signed
    head_size_rel_error_pct: float
    error_delta_pp: float  # abs(head) - abs(baseline)
    path_shifted: bool  # baseline_path != head_path
    error_regressed: bool  # error_delta_pp >= estimation_threshold_pp
    threshold_breach: bool


@dataclass
class ErrorCountDelta:
    head_only_errors: list[str]  # case_ids errored in head but not baseline
    n_baseline_errors: int
    n_head_errors: int

    @property
    def regressed(self) -> bool:
        return len(self.head_only_errors) > 0


@dataclass
class RunConditions:
    """Extracted comparability metadata from a single run."""

    mode: str
    isolate: bool
    platform: str
    cpu_count: int


def _extract_conditions(run: dict[str, Any]) -> RunConditions:
    """Pull the four comparability fields out of a loaded run dict."""
    mode = run.get("mode", "unknown")
    isolate = bool(run.get("config", {}).get("isolate", False))
    platform = run.get("host", {}).get("platform", "unknown")
    # 0 is a sentinel meaning "not recorded" — skip the cpu_count check for older runs.
    cpu_count = int(run.get("host", {}).get("cpu_count", 0))
    return RunConditions(mode=mode, isolate=isolate, platform=platform, cpu_count=cpu_count)


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
    # New axes — populated by compare() when data is present.
    compression_diffs: list[CompressionDiff] = field(default_factory=list)
    estimation_diffs: list[EstimationDiff] = field(default_factory=list)
    error_count_delta: ErrorCountDelta | None = None
    # Thresholds surfaced in markdown header for transparency.
    reduction_threshold_pp: float = 3.0
    size_threshold_pct: float = 5.0
    estimation_threshold_pp: float = 10.0

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
        if self.regressions or self.noise_floor_flags:
            return 1
        if any(d.threshold_breach for d in self.compression_diffs):
            return 1
        if any(d.error_regressed for d in self.estimation_diffs):
            return 1
        if self.error_count_delta and self.error_count_delta.regressed:
            return 1
        return 0


def _extract_compression(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return first non-errored iteration's compression fields per case_id."""
    seen: dict[str, dict[str, Any]] = {}
    for it in run.get("iterations", []):
        if "error" in it:
            continue
        cid = it.get("case_id", "")
        if cid in seen:
            continue
        method = it.get("method")
        reduction_pct = it.get("reduction_pct")
        optimized_size = it.get("optimized_size")
        if method is None or reduction_pct is None or optimized_size is None:
            continue
        seen[cid] = {
            "method": str(method),
            "reduction_pct": float(reduction_pct),
            "optimized_size": int(optimized_size),
        }
    return seen


def _extract_estimation(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return estimation accuracy fields per case_id (accuracy-mode only).

    Returns an empty dict when the run has no accuracy data (quick-mode JSON).
    """
    seen: dict[str, dict[str, Any]] = {}
    for it in run.get("iterations", []):
        if "error" in it:
            continue
        cid = it.get("case_id", "")
        if cid in seen:
            continue
        acc = it.get("accuracy")
        est = it.get("estimate")
        if not isinstance(acc, dict) or not isinstance(est, dict):
            continue
        size_rel = acc.get("size_rel_error_pct")
        path = est.get("path")
        if size_rel is None or path is None:
            continue
        seen[cid] = {
            "path": str(path),
            "size_rel_error_pct": float(size_rel),
        }
    return seen


def _extract_errors(run: dict[str, Any]) -> dict[str, str]:
    """Return case_ids that have an error field in any of their iterations."""
    errors: dict[str, str] = {}
    for it in run.get("iterations", []):
        if "error" not in it:
            continue
        cid = it.get("case_id", "")
        if cid and cid not in errors:
            err = it["error"]
            if isinstance(err, str):
                errors[cid] = err
            elif isinstance(err, dict):
                phase = err.get("phase", "?")
                msg = err.get("message", "unknown error")
                errors[cid] = f"[{phase}] {msg}"
            else:
                errors[cid] = repr(err)
    return errors


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


class HostMismatchError(ValueError):
    """Raised when two runs have incompatible host CPU counts and the caller did not opt in."""

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
    allow_mismatched_cpu_count: bool = False,
    reduction_threshold_pp: float = 3.0,
    size_threshold_pct: float = 5.0,
    estimation_threshold_pp: float = 10.0,
) -> CompareResult:
    """Welch's-t + Cohen's-d diff between two runs.

    When either side has fewer than 3 iterations, falls back to a pure
    |delta%| check at the higher noise_floor_pct threshold instead of the
    stats-backed gate.

    Before computing diffs, validates that the two runs are comparable:
    - mode must match (e.g. both "quick" or both "timing"); mismatches raise
      ModeMismatchError unless allow_mismatched_mode=True.
    - cpu_count must match; mismatches raise HostMismatchError unless
      allow_mismatched_cpu_count=True. cpu_count=0 (missing data) skips the check.
    - isolate flag and platform differences are surfaced via result.a_conditions /
      result.b_conditions; callers are responsible for emitting warnings.

    Raises:
        ModeMismatchError: if modes differ and allow_mismatched_mode is False.
        HostMismatchError: if cpu_counts differ and allow_mismatched_cpu_count is False.
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

    # --- cpu_count check (hard error by default; 0 = missing data, skip) ---
    if (
        a_cond.cpu_count != 0
        and b_cond.cpu_count != 0
        and a_cond.cpu_count != b_cond.cpu_count
        and not allow_mismatched_cpu_count
    ):
        raise HostMismatchError(
            f"baseline host.cpu_count={a_cond.cpu_count} but head host.cpu_count={b_cond.cpu_count}"
            f" — wall_ms is not comparable across CPU counts (multi-threaded codecs scale inversely"
            f" with cores). Pass --allow-mismatched-cpu-count to override."
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

    # --- Compression axis ---
    a_comp = _extract_compression(a_run)
    b_comp = _extract_compression(b_run)
    compression_diffs: list[CompressionDiff] = []
    for case_id in sorted(set(a_comp) & set(b_comp)):
        ac = a_comp[case_id]
        bc = b_comp[case_id]
        red_delta = bc["reduction_pct"] - ac["reduction_pct"]
        base_size = ac["optimized_size"]
        size_delta = (
            ((bc["optimized_size"] - base_size) / base_size * 100.0) if base_size > 0 else 0.0
        )
        downgraded = ac["method"] != "none" and bc["method"] == "none"
        red_regressed = red_delta <= -reduction_threshold_pp
        size_regressed = size_delta >= size_threshold_pct
        compression_diffs.append(
            CompressionDiff(
                case_id=case_id,
                baseline_method=ac["method"],
                head_method=bc["method"],
                baseline_reduction_pct=ac["reduction_pct"],
                head_reduction_pct=bc["reduction_pct"],
                reduction_delta_pp=red_delta,
                baseline_optimized_size=base_size,
                head_optimized_size=bc["optimized_size"],
                size_delta_pct=size_delta,
                method_downgraded_to_none=downgraded,
                reduction_regressed=red_regressed,
                size_regressed=size_regressed,
                threshold_breach=downgraded or red_regressed or size_regressed,
            )
        )

    # --- Estimation axis ---
    a_est = _extract_estimation(a_run)
    b_est = _extract_estimation(b_run)
    estimation_diffs: list[EstimationDiff] = []
    # Only populate when both sides have estimation data.
    if a_est and b_est:
        for case_id in sorted(set(a_est) & set(b_est)):
            ae = a_est[case_id]
            be = b_est[case_id]
            err_delta = abs(be["size_rel_error_pct"]) - abs(ae["size_rel_error_pct"])
            path_shifted = ae["path"] != be["path"]
            err_regressed = err_delta >= estimation_threshold_pp
            estimation_diffs.append(
                EstimationDiff(
                    case_id=case_id,
                    baseline_path=ae["path"],
                    head_path=be["path"],
                    baseline_size_rel_error_pct=ae["size_rel_error_pct"],
                    head_size_rel_error_pct=be["size_rel_error_pct"],
                    error_delta_pp=err_delta,
                    path_shifted=path_shifted,
                    error_regressed=err_regressed,
                    threshold_breach=err_regressed,
                )
            )

    # --- Error-count axis ---
    a_errors = _extract_errors(a_run)
    b_errors = _extract_errors(b_run)
    head_only = sorted(cid for cid in b_errors if cid not in a_errors)
    error_count_delta = ErrorCountDelta(
        head_only_errors=head_only,
        n_baseline_errors=len(a_errors),
        n_head_errors=len(b_errors),
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
        compression_diffs=compression_diffs,
        estimation_diffs=estimation_diffs,
        error_count_delta=error_count_delta,
        reduction_threshold_pp=reduction_threshold_pp,
        size_threshold_pct=size_threshold_pct,
        estimation_threshold_pp=estimation_threshold_pp,
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
    lines.append(f"| **cpu_count** | `{a.cpu_count}` | `{b.cpu_count}` |")

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
    if a.cpu_count != 0 and b.cpu_count != 0 and a.cpu_count != b.cpu_count:
        warnings.append(
            f"CPU count mismatch (`{a.cpu_count}` vs `{b.cpu_count}`) — "
            f"wall_ms is not comparable across CPU counts (multi-threaded codecs scale inversely"
            f" with cores)."
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

    # --- Timing section ---
    lines.append("## Timing")
    lines.append("")
    if not result.diffs:
        lines.append("_No common cases to compare._")
    else:
        rollups = build_format_rollup(result.diffs)
        lines.append(render_format_rollup_table(rollups))
        lines.append("")

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

    lines.append("")

    # --- Compression section ---
    lines.append(_render_compression_section(result))
    lines.append("")

    # --- Estimation section ---
    est_section = _render_estimation_section(result)
    if est_section:
        lines.append(est_section)
        lines.append("")

    # --- Errors section ---
    err_section = _render_errors_section(result)
    if err_section:
        lines.append(err_section)

    return "\n".join(lines)


def _render_compression_section(result: CompareResult) -> str:
    """Render the Compression axis section."""
    lines: list[str] = []
    lines.append("## Compression")
    lines.append("")
    lines.append(
        f"_reduction_threshold={result.reduction_threshold_pp}pp, "
        f"size_threshold={result.size_threshold_pct}%_"
    )
    lines.append("")

    if not result.compression_diffs:
        lines.append("_No compression data available._")
        return "\n".join(lines)

    # Per-format rollup
    by_fmt: dict[str, list[CompressionDiff]] = {}
    for d in result.compression_diffs:
        fmt = _extract_format(d.case_id)
        by_fmt.setdefault(fmt, []).append(d)

    lines.append(
        "| Format | Cases | Method changes | Median Δreduction (pp) | Worst Δpp | Size regressions | Status |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for fmt in sorted(by_fmt.keys()):
        group = by_fmt[fmt]
        method_changes = sum(1 for d in group if d.baseline_method != d.head_method)
        deltas = [d.reduction_delta_pp for d in group]
        med_delta = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
        worst_delta = min(deltas) if deltas else 0.0
        size_regs = sum(1 for d in group if d.size_regressed)
        has_breach = any(d.threshold_breach for d in group)
        status = (
            "❌"
            if any(d.method_downgraded_to_none for d in group)
            else ("⚠" if has_breach else "~")
        )
        lines.append(
            f"| `{fmt}` | {len(group)} | {method_changes} | {med_delta:+.1f} | "
            f"{worst_delta:+.1f} | {size_regs} | {status} |"
        )

    # Highlight method downgrades to "none" explicitly
    downgrades = [d for d in result.compression_diffs if d.method_downgraded_to_none]
    if downgrades:
        lines.append("")
        lines.append(
            f"❌ regression: head method=`none` on {len(downgrades)} case(s) where baseline had a real optimizer path:"
        )
        for d in downgrades:
            lines.append(
                f"  - `{d.case_id}` (was: {d.baseline_method} → now: none, "
                f"reduction {d.baseline_reduction_pct:.1f}% → {d.head_reduction_pct:.1f}%)"
            )

    # Per-case detail for changed cases only
    changed = [
        d
        for d in result.compression_diffs
        if d.threshold_breach
        or d.baseline_method != d.head_method
        or abs(d.reduction_delta_pp) > 0.01
    ]
    if changed:
        lines.append("")
        lines.append("<details>")
        lines.append(
            "<summary>Per-case detail (only cases with method/reduction/size changes)</summary>"
        )
        lines.append("")
        lines.append(
            "| case_id | base method | head method | base red% | head red% | Δpp | base size | head size | Δsize% | flag |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for d in sorted(changed, key=lambda x: x.reduction_delta_pp):
            flag = "❌" if d.method_downgraded_to_none else ("⚠" if d.threshold_breach else "~")
            lines.append(
                f"| `{d.case_id}` | {d.baseline_method[:20]} | {d.head_method[:20]} "
                f"| {d.baseline_reduction_pct:.1f}% | {d.head_reduction_pct:.1f}% "
                f"| {d.reduction_delta_pp:+.1f} | {d.baseline_optimized_size} "
                f"| {d.head_optimized_size} | {d.size_delta_pct:+.1f}% | {flag} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _render_estimation_section(result: CompareResult) -> str:
    """Render the Estimation axis section. Returns empty string if no data."""
    if not result.estimation_diffs:
        return ""

    lines: list[str] = []
    lines.append("## Estimation")
    lines.append("")
    lines.append(f"_estimation_threshold={result.estimation_threshold_pp}pp_")
    lines.append("")

    # Per-format×path rollup
    by_fmt_path: dict[str, list[EstimationDiff]] = {}
    for d in result.estimation_diffs:
        fmt = _extract_format(d.case_id)
        key = f"{fmt} × {d.baseline_path}"
        by_fmt_path.setdefault(key, []).append(d)

    lines.append("| Format×Path | Cases | Median Δerror (pp) | Worst Δpp | Path shifts | Status |")
    lines.append("|---|---|---|---|---|---|")

    for key in sorted(by_fmt_path.keys()):
        group = by_fmt_path[key]
        deltas = [d.error_delta_pp for d in group]
        med_delta = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
        worst_delta = max(deltas) if deltas else 0.0
        path_shifts = sum(1 for d in group if d.path_shifted)
        has_regression = any(d.error_regressed for d in group)
        status = "⚠" if has_regression else "~"
        lines.append(
            f"| {key} | {len(group)} | {med_delta:+.1f} | {worst_delta:+.1f} | {path_shifts} | {status} |"
        )

    # Per-case detail for shifted or regressed cases
    notable = [d for d in result.estimation_diffs if d.path_shifted or d.error_regressed]
    if notable:
        lines.append("")
        lines.append("<details>")
        lines.append(
            "<summary>Per-case detail (only cases where path shifted or error changed by ≥ threshold)</summary>"
        )
        lines.append("")
        lines.append("| case_id | base path | head path | base err% | head err% | Δpp | flag |")
        lines.append("|---|---|---|---|---|---|---|")
        for d in sorted(notable, key=lambda x: -x.error_delta_pp):
            flag = "⚠" if d.error_regressed else "~"
            lines.append(
                f"| `{d.case_id}` | {d.baseline_path} | {d.head_path} "
                f"| {d.baseline_size_rel_error_pct:+.2f}% | {d.head_size_rel_error_pct:+.2f}% "
                f"| {d.error_delta_pp:+.1f} | {flag} |"
            )
        lines.append("")
        lines.append("</details>")

    return "\n".join(lines)


def _render_errors_section(result: CompareResult) -> str:
    """Render the Errors section. Returns empty string if no new head-only errors."""
    if not result.error_count_delta or not result.error_count_delta.head_only_errors:
        return ""

    ecd = result.error_count_delta
    lines: list[str] = []
    lines.append("## Errors")
    lines.append("")
    lines.append(f"⚠ {len(ecd.head_only_errors)} case(s) errored in head but not in baseline:")
    # We only have the case_ids; the run JSON doesn't carry the error message
    # through ErrorCountDelta (it's a case_id list). Just list the ids.
    for cid in ecd.head_only_errors:
        lines.append(f"- `{cid}`")
    return "\n".join(lines)
