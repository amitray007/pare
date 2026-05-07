"""Scorecard data generator for the Pare dashboard.

Reads a bench run JSON (quick-mode or pr-mode) and produces a list of
per-format records suitable for rendering in the dashboard.

Quick-mode runs have timing + reduction data but no quality or accuracy
blocks. The scorecard gracefully shows "—" for those metrics.

PR-mode runs carry accuracy + quality nested in each iteration row. The
scorecard extracts and aggregates these.

Per-format record shape::

    {
        "format": "jpeg",
        "n_cases": 12,
        "median_reduction_pct": 67.3,
        "speed_by_bucket": {
            "small": {"p95_ms": 23.0, "slo_ms": 500, "status": "ok"},
            ...
        },
        "quality": {          # null if lossless OR no quality data
            "ssim_p50": 0.973,
            "ssim_worst": 0.951,
            "threshold": 0.97,
            "n_below": 0,
            "status": "ok",   # ok | warn | fail
        },
        "accuracy": {         # null if no accuracy data
            "size_rel_error_p95": 4.2,
            "threshold": 15.0,
            "status": "ok",   # ok | warn | fail
        },
        "overall_status": "ok",   # ok | warn | fail
        "summary": "Compresses by 67% on average ...",
    }

Status vocabulary: "ok" = green, "warn" = amber, "fail" = red.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

from bench.runner.report.thresholds import (
    ESTIMATION_SIZE_REL_ERROR,
    LATENCY_FORMAT_RELAX,
    LATENCY_P95_SLOS_MS,
    SSIM_DEFAULT,
    SSIM_THRESHOLDS,
)

# Formats that are always lossless — quality scoring is N/A for them.
_LOSSLESS_FORMATS: frozenset[str] = frozenset({"png", "apng", "gif", "bmp", "tiff", "svg", "svgz"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Simple linear-interpolation percentile (matches bench.runner.stats)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _is_failure(it: dict[str, Any]) -> bool:
    err = it.get("error")
    if err is None:
        return False
    return isinstance(err, (str, dict))


def _speed_status(p95_ms: float, slo_ms: float) -> str:
    if p95_ms <= slo_ms:
        return "ok"
    if p95_ms <= slo_ms * 1.25:
        return "warn"
    return "fail"


def _quality_status(ssim_worst: float, threshold: float) -> str:
    if ssim_worst >= threshold:
        return "ok"
    if ssim_worst >= threshold * 0.98:
        return "warn"
    return "fail"


def _accuracy_status(p95: float, threshold: float) -> str:
    if p95 <= threshold:
        return "ok"
    if p95 <= threshold * 1.5:
        return "warn"
    return "fail"


def _overall_status(statuses: list[str]) -> str:
    if "fail" in statuses:
        return "fail"
    if "warn" in statuses:
        return "warn"
    return "ok"


def _build_summary(
    fmt: str,
    median_reduction_pct: float,
    speed_by_bucket: dict[str, Any],
    quality: dict[str, Any] | None,
    accuracy: dict[str, Any] | None,
) -> str:
    """Generate a one-sentence plain-English summary for a format card."""
    parts: list[str] = []

    # Compression
    parts.append(f"Compresses {fmt.upper()} by {median_reduction_pct:.0f}% on average.")

    # Quality (lossy only)
    if quality is not None:
        q_status = quality["status"]
        ssim_p50 = quality["ssim_p50"]
        ssim_worst = quality["ssim_worst"]
        n_below = quality["n_below"]
        threshold = quality["threshold"]

        if q_status == "ok":
            # Summarise using median; include worst for context.
            if ssim_p50 >= 0.99:
                verdict = "visually lossless"
            elif ssim_p50 >= 0.97:
                verdict = "essentially identical"
            elif ssim_p50 >= 0.95:
                verdict = "very good quality"
            else:
                verdict = "acceptable quality"
            parts.append(
                f"Output looks {verdict} (SSIM {ssim_p50:.3f} median, worst {ssim_worst:.3f})."
            )
        elif q_status == "warn":
            parts.append(
                f"Quality near threshold — median SSIM {ssim_p50:.3f} (worst {ssim_worst:.3f}),"
                f" {n_below} case(s) below {threshold:.2f}."
            )
        else:  # fail
            parts.append(
                f"Quality concern — {n_below} case(s) below SSIM {threshold:.2f}"
                f" (worst {ssim_worst:.3f})."
            )

    # Speed — find worst-offender bucket among non-ok buckets, fallback to overall worst
    worst_bucket_ms: float = 0.0
    worst_bucket_name: str = ""
    for bucket, info in speed_by_bucket.items():
        if info["p95_ms"] > worst_bucket_ms:
            worst_bucket_ms = info["p95_ms"]
            worst_bucket_name = bucket

    speed_statuses = [v["status"] for v in speed_by_bucket.values()]
    if all(s == "ok" for s in speed_statuses):
        if worst_bucket_name:
            ms_str = f"{worst_bucket_ms:.0f}ms"
            parts.append(f"Speed within SLO ({ms_str} on {worst_bucket_name}).")
    else:
        # Name the worst-offending non-ok bucket explicitly.
        non_ok_buckets = [
            (b, info) for b, info in speed_by_bucket.items() if info["status"] != "ok"
        ]
        if non_ok_buckets:
            # Pick highest p95 among the failing ones.
            worst_b, worst_info = max(non_ok_buckets, key=lambda x: x[1]["p95_ms"])
            ms_str = f"{worst_info['p95_ms']:.0f}ms"
            slo = worst_info["slo_ms"]
            parts.append(f"Speed concern — p95 {ms_str} on {worst_b} (SLO: {slo}ms).")

    # Accuracy
    if accuracy is not None:
        p95 = accuracy["size_rel_error_p95"]
        acc_status = accuracy["status"]
        threshold_acc = accuracy["threshold"]
        if acc_status == "ok":
            parts.append(f"Prediction within ±{p95:.0f}%.")
        elif acc_status == "warn":
            parts.append(
                f"Prediction accuracy marginal — size_rel_error p95 {p95:.0f}% (threshold {threshold_acc:.0f}%)."
            )
        else:  # fail
            parts.append(
                f"Prediction off — size_rel_error p95 {p95:.0f}% exceeds {threshold_acc:.0f}% threshold."
            )

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Core scorecard builder
# ---------------------------------------------------------------------------


def _detect_pr_mode(run_data: dict[str, Any]) -> bool:
    """Return True if this is a pr-mode run (has quality/accuracy blocks)."""
    if run_data.get("mode") == "pr":
        return True
    # Heuristic: check the first successful iteration for quality/accuracy keys.
    for it in run_data.get("iterations", []):
        if _is_failure(it):
            continue
        if "quality" in it or "accuracy" in it:
            return True
    return False


def build_scorecard(run_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the list of per-format scorecard records from a parsed run JSON.

    Works with both quick-mode (timing only) and pr-mode (timing + quality +
    accuracy) runs. Quick-mode runs produce ``quality=null`` and
    ``accuracy=null`` for all formats.
    """
    is_pr = _detect_pr_mode(run_data)
    iterations: list[dict[str, Any]] = run_data.get("iterations", [])
    stats: list[dict[str, Any]] = run_data.get("stats", [])

    # ----------------------------------------------------------------
    # Build per-format data from stats (timing + reduction)
    # ----------------------------------------------------------------
    # stats rows already have aggregated p50/p95 and reduction_pct.
    by_fmt_stats: dict[str, list[dict[str, Any]]] = {}
    for s in stats:
        fmt = s.get("format", "unknown")
        by_fmt_stats.setdefault(fmt, []).append(s)

    # ----------------------------------------------------------------
    # Build per-format quality + accuracy from iterations (pr-mode only)
    # ----------------------------------------------------------------
    # De-duplicate by case_id (pr-mode repeats iteration rows for timing).
    seen_cases: set[str] = set()
    by_fmt_iters: dict[str, list[dict[str, Any]]] = {}
    for it in iterations:
        if _is_failure(it):
            continue
        cid = it.get("case_id", "")
        if cid in seen_cases:
            continue
        seen_cases.add(cid)
        fmt = it.get("format", "unknown")
        by_fmt_iters.setdefault(fmt, []).append(it)

    # ----------------------------------------------------------------
    # Assemble per-format records
    # ----------------------------------------------------------------
    all_formats = sorted(set(by_fmt_stats) | set(by_fmt_iters))
    records: list[dict[str, Any]] = []

    for fmt in all_formats:
        fmt_stats = by_fmt_stats.get(fmt, [])
        fmt_iters = by_fmt_iters.get(fmt, [])

        # --- n_cases ---
        n_cases = len(fmt_stats)

        # --- median_reduction_pct ---
        reductions = [s.get("reduction_pct", 0.0) for s in fmt_stats]
        median_reduction_pct = round(median(reductions) if reductions else 0.0, 1)

        # --- speed_by_bucket ---
        # Group stats rows by bucket, then derive p95_ms per bucket.
        by_bucket: dict[str, list[float]] = {}
        for s in fmt_stats:
            bucket = s.get("bucket", "small")
            p95 = s.get("p95_ms", 0.0)
            by_bucket.setdefault(bucket, []).append(p95)

        speed_by_bucket: dict[str, dict[str, Any]] = {}
        for bucket, p95_vals in by_bucket.items():
            base_slo = LATENCY_P95_SLOS_MS.get(bucket, 2000)
            relax = LATENCY_FORMAT_RELAX.get((fmt, bucket), 1.0)
            effective_slo = int(base_slo * relax)
            bucket_p95 = round(_percentile(p95_vals, 95), 1)
            speed_by_bucket[bucket] = {
                "p95_ms": bucket_p95,
                "slo_ms": effective_slo,
                "status": _speed_status(bucket_p95, effective_slo),
            }

        # --- quality ---
        quality_record: dict[str, Any] | None = None
        if is_pr and fmt not in _LOSSLESS_FORMATS:
            # Collect SSIM values from quality blocks. All presets contribute;
            # we report the worst threshold (most lenient preset = "high" = 0.95).
            ssim_by_preset: dict[str, list[float]] = {}
            for it in fmt_iters:
                q = it.get("quality")
                if not isinstance(q, dict):
                    continue
                ssim_val = q.get("ssim")
                if ssim_val is None:
                    continue
                preset = it.get("preset", "medium")
                ssim_by_preset.setdefault(preset, []).append(float(ssim_val))

            all_ssims: list[float] = [v for vals in ssim_by_preset.values() for v in vals]
            if all_ssims:
                ssim_p50 = _percentile(all_ssims, 50)
                ssim_worst = min(all_ssims)
                # Use the lowest (most lenient) threshold across the presets observed.
                # This means a "fail" is genuinely bad, not an artifact of preset mismatch.
                observed_thresholds = [SSIM_THRESHOLDS.get(p, SSIM_DEFAULT) for p in ssim_by_preset]
                threshold = min(observed_thresholds) if observed_thresholds else SSIM_DEFAULT
                n_below = sum(
                    1
                    for it in fmt_iters
                    if isinstance(it.get("quality"), dict)
                    and it["quality"].get("ssim") is not None
                    and float(it["quality"]["ssim"])
                    < SSIM_THRESHOLDS.get(it.get("preset", "medium"), SSIM_DEFAULT)
                )
                quality_record = {
                    "ssim_p50": round(ssim_p50, 4),
                    "ssim_worst": round(ssim_worst, 4),
                    "threshold": threshold,
                    "n_below": n_below,
                    "status": _quality_status(ssim_worst, threshold),
                }

        # --- accuracy ---
        accuracy_record: dict[str, Any] | None = None
        if is_pr:
            size_rel_errs: list[float] = []
            for it in fmt_iters:
                acc = it.get("accuracy")
                if not isinstance(acc, dict):
                    continue
                val = acc.get("size_rel_error_pct")
                if val is not None:
                    size_rel_errs.append(abs(float(val)))

            if size_rel_errs:
                p95_err = round(_percentile(size_rel_errs, 95), 2)
                threshold = ESTIMATION_SIZE_REL_ERROR["p95_max"]
                accuracy_record = {
                    "size_rel_error_p95": p95_err,
                    "threshold": threshold,
                    "status": _accuracy_status(p95_err, threshold),
                }

        # --- overall_status ---
        sub_statuses: list[str] = []
        for bkt in speed_by_bucket.values():
            sub_statuses.append(bkt["status"])
        if quality_record is not None:
            sub_statuses.append(quality_record["status"])
        if accuracy_record is not None:
            sub_statuses.append(accuracy_record["status"])
        overall = _overall_status(sub_statuses) if sub_statuses else "ok"

        # --- summary ---
        summary = _build_summary(
            fmt=fmt,
            median_reduction_pct=median_reduction_pct,
            speed_by_bucket=speed_by_bucket,
            quality=quality_record,
            accuracy=accuracy_record,
        )

        records.append(
            {
                "format": fmt,
                "n_cases": n_cases,
                "median_reduction_pct": median_reduction_pct,
                "speed_by_bucket": speed_by_bucket,
                "quality": quality_record,
                "accuracy": accuracy_record,
                "overall_status": overall,
                "summary": summary,
            }
        )

    # Sort: fail first, then warn, then ok, then alpha within group.
    _status_order = {"fail": 0, "warn": 1, "ok": 2}
    records.sort(key=lambda r: (_status_order.get(r["overall_status"], 3), r["format"]))
    return records


# ---------------------------------------------------------------------------
# Load a baseline JSON for the scorecard
# ---------------------------------------------------------------------------


def load_scorecard_data(baseline_path: Path) -> dict[str, Any] | None:
    """Load and parse a run JSON for scorecard use.

    Returns None if the file is missing or has an unexpected schema version.
    Does *not* raise — the dashboard must render gracefully with no data.
    """
    if not baseline_path.exists():
        return None
    try:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if raw.get("schema_version") != 2:
        return None
    return raw


def build_kpis(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive top-level KPI numbers from the list of format records.

    Returns::

        {
            "avg_reduction_pct": 54.2,
            "quality_green": 5,
            "quality_total": 5,
            "speed_green": 28,
            "speed_total": 32,
            "accuracy_green": 4,
            "accuracy_total": 5,
        }
    """
    if not records:
        return {
            "avg_reduction_pct": 0.0,
            "quality_green": 0,
            "quality_total": 0,
            "speed_green": 0,
            "speed_total": 0,
            "accuracy_green": 0,
            "accuracy_total": 0,
        }

    reductions = [r["median_reduction_pct"] for r in records if r["n_cases"] > 0]
    avg_reduction = round(sum(reductions) / len(reductions), 1) if reductions else 0.0

    quality_green = sum(
        1 for r in records if r["quality"] is not None and r["quality"]["status"] == "ok"
    )
    quality_total = sum(1 for r in records if r["quality"] is not None)

    speed_green = 0
    speed_total = 0
    for r in records:
        for bkt in r["speed_by_bucket"].values():
            speed_total += 1
            if bkt["status"] == "ok":
                speed_green += 1

    accuracy_green = sum(
        1 for r in records if r["accuracy"] is not None and r["accuracy"]["status"] == "ok"
    )
    accuracy_total = sum(1 for r in records if r["accuracy"] is not None)

    return {
        "avg_reduction_pct": avg_reduction,
        "quality_green": quality_green,
        "quality_total": quality_total,
        "speed_green": speed_green,
        "speed_total": speed_total,
        "accuracy_green": accuracy_green,
        "accuracy_total": accuracy_total,
    }
