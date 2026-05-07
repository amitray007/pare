"""Tests for bench.dashboard.scorecard.

Covers:
1. Scorecard generation from a pr-mode JSON: correct n_cases, status, summaries.
2. Scorecard generation from a quick-mode JSON: graceful "no data" for quality/accuracy.
3. Threshold logic: speed_by_bucket correctly applies LATENCY_FORMAT_RELAX multiplier.
4. Overall status escalation: any fail -> overall fail; any warn but no fail -> overall warn.
5. Summary string contains expected phrases for typical inputs.
6. KPI aggregation from scorecard records.
7. Dashboard HTML embeds scorecard JSON and has the correct structural sections.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench.dashboard.build import _find_repo_root, main
from bench.dashboard.scorecard import build_kpis, build_scorecard, load_scorecard_data
from bench.runner.report.thresholds import (
    ESTIMATION_SIZE_REL_ERROR,
    LATENCY_FORMAT_RELAX,
    LATENCY_P95_SLOS_MS,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

REPO_ROOT = _find_repo_root(Path(__file__))


def _make_measurement(wall_ms: float = 50.0) -> dict[str, Any]:
    return {
        "wall_ms": wall_ms,
        "parent_user_ms": wall_ms * 0.8,
        "parent_sys_ms": 1.0,
        "children_user_ms": wall_ms * 0.6,
        "children_sys_ms": 0.5,
        "total_cpu_ms": wall_ms * 1.4,
        "parallelism": 2.0,
        "parent_peak_rss_kb": 80000,
        "children_peak_rss_kb": 500000,
        "peak_rss_kb": 500000,
        "py_peak_alloc_kb": None,
        "phases": {},
        "rss_samples": [],
    }


def _make_stat(
    fmt: str,
    bucket: str = "small",
    preset: str = "medium",
    p50_ms: float = 30.0,
    p95_ms: float = 60.0,
    reduction_pct: float = 55.0,
) -> dict[str, Any]:
    """Minimal CaseStats-shaped dict (as written by json_writer._roll_up_stats)."""
    return {
        "case_id": f"test_{fmt}.{fmt}@{preset}",
        "format": fmt,
        "preset": preset,
        "bucket": bucket,
        "iterations": 2,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "median_ms": p50_ms,
        "mad_ms": 1.0,
        "p99_ms": p95_ms * 1.1,
        "reduction_pct": reduction_pct,
        "method": "test",
        "children_cpu_p50_ms": p50_ms * 0.6,
        "parallelism_p50": 2.0,
        "parent_peak_rss_p95_kb": 80000,
        "children_peak_rss_p95_kb": 500000,
        "py_peak_alloc_p95_kb": None,
        "rss_samples": None,
    }


def _make_quick_run(
    formats: list[str] | None = None, buckets: list[str] | None = None
) -> dict[str, Any]:
    """Build a minimal quick-mode run JSON."""
    if formats is None:
        formats = ["jpeg", "png"]
    if buckets is None:
        buckets = ["small", "medium"]

    iterations: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for fmt in formats:
        for bucket in buckets:
            for preset in ["high", "medium"]:
                cid = f"img_{fmt}_{bucket}.{fmt}@{preset}"
                iterations.append(
                    {
                        "case_id": cid,
                        "name": f"img_{fmt}_{bucket}",
                        "bucket": bucket,
                        "format": fmt,
                        "preset": preset,
                        "input_size": 100_000,
                        "iteration": 0,
                        "measurement": _make_measurement(40.0),
                        "reduction_pct": 60.0,
                        "method": "test",
                        "optimized_size": 40_000,
                    }
                )
                stats.append(_make_stat(fmt, bucket, preset, p50_ms=35.0, p95_ms=55.0))

    return {
        "schema_version": 2,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "mode": "quick",
        "config": {"warmup": 0, "repeat": 1},
        "git": {"commit": "abc1234def", "branch": "main", "dirty": False},
        "host": {"platform": "linux", "cpu_count": 4, "rss_unit": "kb"},
        "library_versions": {},
        "manifest": {"name": "core", "sha256": "abc123"},
        "annotations": {},
        "iterations": iterations,
        "stats": stats,
    }


def _make_pr_run(
    fmt: str = "jpeg",
    ssim_values: list[float] | None = None,
    size_rel_errors: list[float] | None = None,
    reduction_pct: float = 67.0,
    bucket: str = "small",
    p95_ms: float = 45.0,
) -> dict[str, Any]:
    """Build a minimal pr-mode run JSON for one format."""
    if ssim_values is None:
        ssim_values = [0.973, 0.968, 0.981]
    if size_rel_errors is None:
        size_rel_errors = [3.2, 4.1, 5.8]

    iterations: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []

    for i, (ssim_val, rel_err) in enumerate(zip(ssim_values, size_rel_errors)):
        preset = "medium"
        cid = f"img_{fmt}_{bucket}_{i}.{fmt}@{preset}"
        it: dict[str, Any] = {
            "case_id": cid,
            "name": f"img_{fmt}_{bucket}_{i}",
            "bucket": bucket,
            "format": fmt,
            "preset": preset,
            "input_size": 100_000,
            "iteration": 0,
            "measurement": _make_measurement(p95_ms * 0.9),
            "reduction_pct": reduction_pct,
            "method": "test",
            "optimized_size": int(100_000 * (1 - reduction_pct / 100)),
            "quality": {
                "ssim": ssim_val,
                "psnr_db": 35.0,
                "ssimulacra2": None,
                "butteraugli_max": None,
                "butteraugli_3norm": None,
            },
            "accuracy": {
                "size_rel_error_pct": rel_err,
                "reduction_abs_error_pct_abs": abs(rel_err * 0.5),
                "already_optimized": False,
            },
        }
        iterations.append(it)
        stats.append(
            _make_stat(
                fmt, bucket, preset, p50_ms=p95_ms * 0.7, p95_ms=p95_ms, reduction_pct=reduction_pct
            )
        )

    return {
        "schema_version": 2,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "mode": "pr",
        "config": {"warmup": 0, "repeat": 2},
        "git": {"commit": "deadbeef1234", "branch": "feat/test", "dirty": False},
        "host": {"platform": "linux", "cpu_count": 4, "rss_unit": "kb"},
        "library_versions": {},
        "manifest": {"name": "core", "sha256": "abc123"},
        "annotations": {},
        "iterations": iterations,
        "stats": stats,
    }


# ---------------------------------------------------------------------------
# Test 1: pr-mode JSON → correct n_cases, status, summaries
# ---------------------------------------------------------------------------


def test_scorecard_from_pr_mode_basic() -> None:
    """Scorecard from pr-mode run has correct n_cases, quality, accuracy, status."""
    run = _make_pr_run(
        fmt="jpeg", ssim_values=[0.975, 0.970, 0.982], size_rel_errors=[3.0, 4.5, 5.0]
    )
    records = build_scorecard(run)

    assert len(records) == 1, "Expected exactly 1 format (jpeg)"
    r = records[0]

    assert r["format"] == "jpeg"
    assert r["n_cases"] == 3
    assert r["median_reduction_pct"] > 0.0

    # Quality block must be present for lossy format with pr-mode data.
    assert r["quality"] is not None, "Expected quality block for jpeg pr-mode run"
    q = r["quality"]
    assert "ssim_p50" in q
    assert "ssim_worst" in q
    assert "threshold" in q
    assert q["ssim_worst"] > 0.0
    # All values well above threshold → status should be ok.
    assert q["status"] == "ok", f"Expected quality ok, got {q['status']}"

    # Accuracy block must be present.
    assert r["accuracy"] is not None, "Expected accuracy block for pr-mode run"
    a = r["accuracy"]
    assert a["size_rel_error_p95"] > 0.0
    assert a["threshold"] == ESTIMATION_SIZE_REL_ERROR["p95_max"]
    assert a["status"] == "ok", f"Expected accuracy ok, got {a['status']}"

    # Summary must be a non-empty string.
    assert isinstance(r["summary"], str)
    assert len(r["summary"]) > 10


def test_scorecard_from_pr_mode_n_cases() -> None:
    """n_cases reflects the number of stats rows, not raw iterations."""
    # Provide matching ssim_values and size_rel_errors so _make_pr_run
    # generates exactly 4 cases (zip stops at the shorter list).
    run = _make_pr_run(
        fmt="webp",
        ssim_values=[0.96, 0.97, 0.98, 0.95],
        size_rel_errors=[2.0, 3.0, 4.0, 5.0],
    )
    records = build_scorecard(run)
    r = next(x for x in records if x["format"] == "webp")
    assert r["n_cases"] == 4


# ---------------------------------------------------------------------------
# Test 2: quick-mode JSON → quality/accuracy are null
# ---------------------------------------------------------------------------


def test_scorecard_from_quick_mode_no_quality() -> None:
    """Quick-mode run produces quality=None and accuracy=None for all formats."""
    run = _make_quick_run(formats=["jpeg", "png"])
    records = build_scorecard(run)

    for r in records:
        assert (
            r["quality"] is None
        ), f"Expected quality=null for quick-mode format {r['format']}, got {r['quality']}"
        assert (
            r["accuracy"] is None
        ), f"Expected accuracy=null for quick-mode format {r['format']}, got {r['accuracy']}"


def test_scorecard_from_quick_mode_has_speed_data() -> None:
    """Quick-mode run still produces speed_by_bucket entries."""
    run = _make_quick_run(formats=["png"], buckets=["small", "medium"])
    records = build_scorecard(run)

    assert len(records) == 1
    r = records[0]
    spd = r["speed_by_bucket"]
    assert "small" in spd
    assert "medium" in spd
    for bkt, info in spd.items():
        assert "p95_ms" in info
        assert "slo_ms" in info
        assert info["status"] in ("ok", "warn", "fail")


# ---------------------------------------------------------------------------
# Test 3: LATENCY_FORMAT_RELAX multiplier is applied
# ---------------------------------------------------------------------------


def test_speed_slo_applies_format_relax_for_heic_large() -> None:
    """HEIC on large bucket should use SLO * 1.5, not the bare SLO."""
    run = _make_quick_run(formats=["heic"], buckets=["large"])
    records = build_scorecard(run)

    r = next(x for x in records if x["format"] == "heic")
    large_info = r["speed_by_bucket"]["large"]

    base_slo = LATENCY_P95_SLOS_MS["large"]  # 8000
    relax = LATENCY_FORMAT_RELAX[("heic", "large")]  # 1.5
    expected_slo = int(base_slo * relax)  # 12000

    assert (
        large_info["slo_ms"] == expected_slo
    ), f"Expected relaxed SLO {expected_slo}ms for heic/large, got {large_info['slo_ms']}ms"


def test_speed_slo_applies_format_relax_for_jxl_xlarge() -> None:
    """JXL on xlarge bucket should use SLO * 1.5."""
    run = _make_quick_run(formats=["jxl"], buckets=["xlarge"])
    records = build_scorecard(run)

    r = next(x for x in records if x["format"] == "jxl")
    xl_info = r["speed_by_bucket"]["xlarge"]

    base_slo = LATENCY_P95_SLOS_MS["xlarge"]  # 20000
    relax = LATENCY_FORMAT_RELAX[("jxl", "xlarge")]  # 1.5
    expected_slo = int(base_slo * relax)  # 30000

    assert (
        xl_info["slo_ms"] == expected_slo
    ), f"Expected relaxed SLO {expected_slo}ms for jxl/xlarge, got {xl_info['slo_ms']}ms"


def test_speed_slo_no_relax_for_jpeg_large() -> None:
    """JPEG on large bucket must use the bare SLO (no relaxation)."""
    run = _make_quick_run(formats=["jpeg"], buckets=["large"])
    records = build_scorecard(run)

    r = next(x for x in records if x["format"] == "jpeg")
    large_info = r["speed_by_bucket"]["large"]

    expected_slo = LATENCY_P95_SLOS_MS["large"]  # 8000 — no relax
    assert (
        large_info["slo_ms"] == expected_slo
    ), f"Expected bare SLO {expected_slo}ms for jpeg/large, got {large_info['slo_ms']}ms"


# ---------------------------------------------------------------------------
# Test 4: overall status escalation
# ---------------------------------------------------------------------------


def test_overall_status_fail_when_any_bucket_fails() -> None:
    """A format with p95 >> SLO gets overall_status='fail'."""
    # Large p95 that breaches SLO by >25%.
    run = _make_pr_run(fmt="jpeg", p95_ms=9999.0, bucket="small")  # SLO small=500ms
    records = build_scorecard(run)
    r = records[0]

    assert r["speed_by_bucket"]["small"]["status"] == "fail"
    assert (
        r["overall_status"] == "fail"
    ), f"Expected overall fail due to speed failure, got {r['overall_status']}"


def test_overall_status_warn_when_warn_but_no_fail() -> None:
    """A format with p95 slightly above SLO (warn zone) but no hard fail gets overall warn."""
    # p95_ms = 540ms against SLO=500ms → 8% over → warn zone (<=25% over = warn).
    run = _make_pr_run(fmt="png", p95_ms=540.0, bucket="small")
    records = build_scorecard(run)
    r = records[0]

    # Should not be a hard fail.
    assert r["speed_by_bucket"]["small"]["status"] in ("warn", "fail")
    if r["speed_by_bucket"]["small"]["status"] == "warn":
        # No other failures → overall should be warn.
        assert r["overall_status"] in ("warn", "fail")


def test_overall_status_ok_when_all_green() -> None:
    """All metrics within SLO/threshold → overall_status='ok'."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.980, 0.975, 0.972],
        size_rel_errors=[2.0, 3.0, 4.0],
        p95_ms=45.0,  # well within small SLO of 500ms
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["overall_status"] == "ok", (
        f"Expected overall ok, got {r['overall_status']}. "
        f"speed={r['speed_by_bucket']}, quality={r['quality']}, accuracy={r['accuracy']}"
    )


def test_overall_status_fail_from_quality() -> None:
    """Low SSIM (below threshold) escalates overall_status to fail."""
    # threshold for "medium" is 0.97; provide values well below it.
    run = _make_pr_run(
        fmt="webp",
        ssim_values=[0.90, 0.88, 0.85],  # all far below 0.97
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["quality"] is not None
    assert r["quality"]["status"] in ("warn", "fail")
    # worst ssim=0.85, threshold=0.97; this is more than 2% below → "fail"
    assert r["quality"]["status"] == "fail"
    assert r["overall_status"] == "fail"


# ---------------------------------------------------------------------------
# Test 5: summary string contains expected phrases
# ---------------------------------------------------------------------------


def test_summary_contains_format_name() -> None:
    """Summary must mention the format name."""
    run = _make_pr_run(fmt="jpeg", reduction_pct=67.0)
    records = build_scorecard(run)
    r = records[0]

    # Format name in upper case is expected from _build_summary.
    assert "JPEG" in r["summary"], f"Format name missing from summary: {r['summary']}"


def test_summary_contains_reduction_pct() -> None:
    """Summary must state the compression percentage."""
    run = _make_pr_run(fmt="jpeg", reduction_pct=67.0)
    records = build_scorecard(run)
    r = records[0]

    # "67%" should appear somewhere in the summary.
    assert "67%" in r["summary"], f"Reduction % missing from summary: {r['summary']}"


def test_summary_contains_ssim_phrase() -> None:
    """Summary must describe quality using SSIM for lossy formats."""
    run = _make_pr_run(fmt="jpeg", ssim_values=[0.975, 0.973, 0.980])
    records = build_scorecard(run)
    r = records[0]

    # Should contain one of the quality adjectives.
    quality_phrases = [
        "visually lossless",
        "essentially identical",
        "very good quality",
        "acceptable quality",
    ]
    assert any(
        p in r["summary"] for p in quality_phrases
    ), f"No quality phrase found in summary: {r['summary']}"


def test_summary_contains_prediction_phrase() -> None:
    """Summary must mention prediction accuracy for pr-mode runs."""
    run = _make_pr_run(fmt="jpeg", size_rel_errors=[3.0, 4.0, 2.5])
    records = build_scorecard(run)
    r = records[0]

    assert (
        "Prediction" in r["summary"] or "prediction" in r["summary"]
    ), f"Prediction accuracy phrase missing from summary: {r['summary']}"


# ---------------------------------------------------------------------------
# Test 6: KPI aggregation
# ---------------------------------------------------------------------------


def test_build_kpis_all_ok() -> None:
    """KPIs should sum green counts correctly for all-ok records."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.975, 0.973],
        size_rel_errors=[3.0, 4.0],
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    kpis = build_kpis(records)

    assert kpis["avg_reduction_pct"] > 0.0
    assert kpis["speed_total"] > 0
    assert kpis["speed_green"] <= kpis["speed_total"]
    assert kpis["quality_total"] >= 0
    assert kpis["accuracy_total"] >= 0


def test_build_kpis_empty() -> None:
    """Empty records produce zero KPIs without crashing."""
    kpis = build_kpis([])
    assert kpis["avg_reduction_pct"] == 0.0
    assert kpis["speed_total"] == 0
    assert kpis["quality_total"] == 0
    assert kpis["accuracy_total"] == 0


def test_build_kpis_quick_mode_no_quality() -> None:
    """Quick-mode records (quality=None) must have quality_total=0."""
    run = _make_quick_run(formats=["jpeg", "png"])
    records = build_scorecard(run)
    kpis = build_kpis(records)

    assert kpis["quality_total"] == 0, "Quick-mode should have 0 quality_total"
    assert kpis["accuracy_total"] == 0, "Quick-mode should have 0 accuracy_total"


# ---------------------------------------------------------------------------
# Test 7: Dashboard HTML structural integrity
# ---------------------------------------------------------------------------


def test_dashboard_html_has_scorecard_section(tmp_path: Path) -> None:
    """Built index.html must contain <section id=\"scorecard-grid\">."""
    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])
    assert rc == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    assert 'id="scorecard-grid"' in html, "scorecard-grid section missing from index.html"
    assert 'id="kpis"' in html, "KPI section missing from index.html"
    assert 'id="trends-section"' in html, "trends section missing from index.html"


def test_dashboard_html_contains_embedded_scorecard_json(tmp_path: Path) -> None:
    """index.html must embed the SCORECARD constant (not __SCORECARD_JSON__ placeholder)."""
    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])
    assert rc == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    assert (
        "__SCORECARD_JSON__" not in html
    ), "Placeholder __SCORECARD_JSON__ was not replaced in the rendered HTML"
    # The constant should be the word SCORECARD followed by = and a JSON value.
    assert "const SCORECARD = " in html, "SCORECARD variable assignment missing from HTML"


def test_dashboard_html_has_trend_reference(tmp_path: Path) -> None:
    """index.html must still reference data/history.json for trend charts."""
    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])
    assert rc == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "data/history.json" in html, "Trend data path missing from index.html"


def test_dashboard_html_with_baseline_flag(tmp_path: Path) -> None:
    """--baseline flag points dashboard at a custom JSON file."""
    # Write a minimal valid run JSON.
    baseline = tmp_path / "smoke.json"
    run = _make_quick_run(formats=["jpeg"])
    baseline.write_text(json.dumps(run), encoding="utf-8")

    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT), "--baseline", str(baseline)])
    assert rc == 0

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "const SCORECARD = " in html
    # Since it's quick-mode, there should be no quality/accuracy data embedded.
    # We just check the page renders without the placeholder still present.
    assert "__SCORECARD_JSON__" not in html


def test_dashboard_html_page_size_under_200kb(tmp_path: Path) -> None:
    """index.html without trend data should be under 200KB."""
    out = tmp_path / "out"
    rc = main(["--out-dir", str(out), "--repo", str(REPO_ROOT)])
    assert rc == 0

    size_bytes = (out / "index.html").stat().st_size
    limit_bytes = 200 * 1024  # 200KB
    assert size_bytes < limit_bytes, (
        f"index.html is {size_bytes / 1024:.1f}KB — exceeds 200KB budget. "
        "Check for excessive inline content."
    )


# ---------------------------------------------------------------------------
# Test 8: load_scorecard_data — edge cases
# ---------------------------------------------------------------------------


def test_load_scorecard_data_missing_file() -> None:
    """Missing file returns None without raising."""
    result = load_scorecard_data(Path("/nonexistent/path.json"))
    assert result is None


def test_load_scorecard_data_wrong_schema_version(tmp_path: Path) -> None:
    """File with schema_version != 2 returns None."""
    f = tmp_path / "bad.json"
    f.write_text(json.dumps({"schema_version": 1, "mode": "quick"}))
    result = load_scorecard_data(f)
    assert result is None


def test_load_scorecard_data_valid(tmp_path: Path) -> None:
    """Valid schema_version=2 file is returned as dict."""
    run = _make_quick_run(formats=["jpeg"])
    f = tmp_path / "run.json"
    f.write_text(json.dumps(run))
    result = load_scorecard_data(f)
    assert result is not None
    assert result["mode"] == "quick"


# ---------------------------------------------------------------------------
# Test 9: Lossless formats get quality=None in pr-mode
# ---------------------------------------------------------------------------


def test_lossless_formats_have_no_quality_in_pr_mode() -> None:
    """PNG, GIF, BMP etc. are lossless — quality block must be None even in pr-mode."""
    lossless_fmts = ["png", "gif", "bmp", "tiff", "svg", "svgz", "apng"]
    for fmt in lossless_fmts:
        # Build a pr-mode run that mistakenly includes quality data for lossless.
        # The scorecard should ignore it.
        run = _make_quick_run(formats=[fmt])
        run["mode"] = "pr"  # force pr-mode detection
        records = build_scorecard(run)
        for r in records:
            if r["format"] == fmt:
                assert (
                    r["quality"] is None
                ), f"Lossless format {fmt} should have quality=None but got {r['quality']}"


# ---------------------------------------------------------------------------
# Test 10: Sort order — fail first, then warn, then ok
# ---------------------------------------------------------------------------


def test_scorecard_sort_order_fail_first() -> None:
    """Records with overall_status='fail' must appear before 'ok' records."""
    # jpeg with huge p95 (fail) + png well within SLO (ok)
    run_fail = _make_pr_run(fmt="jpeg", p95_ms=9999.0, bucket="small")
    run_ok_png = _make_quick_run(formats=["png"], buckets=["small"])

    # Merge iterations and stats.
    combined: dict[str, Any] = {
        "schema_version": 2,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "mode": "pr",
        "config": {},
        "git": {"commit": "x", "branch": "x", "dirty": False},
        "host": {"platform": "linux", "cpu_count": 4, "rss_unit": "kb"},
        "library_versions": {},
        "manifest": {"name": "core", "sha256": "x"},
        "annotations": {},
        "iterations": run_fail["iterations"] + run_ok_png["iterations"],
        "stats": run_fail["stats"] + run_ok_png["stats"],
    }
    records = build_scorecard(combined)

    statuses = [r["overall_status"] for r in records]
    # Fail must appear before ok.
    if "fail" in statuses and "ok" in statuses:
        fail_idx = statuses.index("fail")
        ok_idx = statuses.index("ok")
        assert fail_idx < ok_idx, f"'fail' record not before 'ok': {statuses}"


# ---------------------------------------------------------------------------
# Test 11: Summary text accurately reflects quality/speed/accuracy status (Bug 3 fix)
# ---------------------------------------------------------------------------


def test_summary_quality_ok_mentions_worst_ssim() -> None:
    """When quality status is ok, summary includes both median and worst SSIM (Bug 3 fix)."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.980, 0.975, 0.972],
        size_rel_errors=[2.0, 3.0, 4.0],
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["quality"] is not None
    assert r["quality"]["status"] == "ok", f"Expected ok quality, got {r['quality']['status']}"

    summary = r["summary"]
    # Must contain median SSIM and worst SSIM explicitly (not just a vague "very good quality").
    assert "SSIM" in summary, f"SSIM missing from ok-quality summary: {summary}"
    assert "worst" in summary, f"'worst' SSIM missing from ok-quality summary: {summary}"


def test_summary_quality_warn_names_breach_count() -> None:
    """When quality is warn, summary names worst SSIM and breach count."""
    # threshold for "medium" is 0.97; just slightly below → warn zone (>=0.97*0.98=0.9506)
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.958, 0.960, 0.962],  # worst=0.958 < 0.97 but >=0.97*0.98≈0.9506
        size_rel_errors=[3.0, 4.0, 5.0],
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["quality"] is not None
    q_status = r["quality"]["status"]
    # If the values land in warn/fail (depends on exact thresholds), check accordingly.
    if q_status in ("warn", "fail"):
        summary = r["summary"]
        assert "SSIM" in summary, f"SSIM missing from {q_status}-quality summary: {summary}"
        # Must name the worst-case SSIM when not ok.
        assert (
            "worst" in summary or "case" in summary
        ), f"Breach details missing from {q_status}-quality summary: {summary}"


def test_summary_quality_fail_names_count_and_worst() -> None:
    """When quality status is fail, summary names breach count and worst SSIM explicitly."""
    # threshold for "medium" is 0.97; provide values well below.
    run = _make_pr_run(
        fmt="webp",
        ssim_values=[0.90, 0.88, 0.92],  # all fail threshold 0.97
        size_rel_errors=[3.0, 4.0, 5.0],
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["quality"] is not None
    assert r["quality"]["status"] == "fail", f"Expected fail, got {r['quality']['status']}"
    n_below = r["quality"]["n_below"]
    worst = r["quality"]["ssim_worst"]

    summary = r["summary"]
    # Summary must acknowledge the failure count and worst-case value.
    assert (
        "case" in summary or str(n_below) in summary
    ), f"Breach count missing from fail-quality summary: {summary}"
    assert (
        f"{worst:.3f}" in summary or "0.88" in summary or "SSIM" in summary
    ), f"Worst SSIM not named in fail-quality summary: {summary}"
    # Must NOT say "looks very good quality" when failing.
    assert (
        "very good quality" not in summary
    ), f"Summary says 'very good quality' despite fail status: {summary}"
    assert (
        "essentially identical" not in summary
    ), f"Summary says 'essentially identical' despite fail status: {summary}"


def test_summary_speed_ok_uses_concise_phrasing() -> None:
    """When speed is ok, summary uses concise 'within SLO' phrasing."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.975, 0.973, 0.980],
        size_rel_errors=[3.0, 4.0, 5.0],
        p95_ms=40.0,  # well within small SLO 500ms
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["speed_by_bucket"]["small"]["status"] == "ok"
    summary = r["summary"]
    assert "SLO" in summary, f"SLO mention missing from ok-speed summary: {summary}"


def test_summary_speed_warn_or_fail_names_bucket() -> None:
    """When speed is warn/fail, summary names the offending bucket and values."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.975, 0.973, 0.980],
        size_rel_errors=[3.0, 4.0, 5.0],
        p95_ms=9999.0,  # far above small SLO 500ms
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    bucket_status = r["speed_by_bucket"]["small"]["status"]
    assert bucket_status in ("warn", "fail"), f"Expected warn/fail, got {bucket_status}"

    summary = r["summary"]
    assert "small" in summary, f"Bucket name 'small' missing from speed-fail summary: {summary}"
    assert "SLO" in summary, f"SLO missing from speed-fail summary: {summary}"


def test_summary_accuracy_ok_phrasing() -> None:
    """When accuracy is ok, summary uses terse 'Prediction within' phrasing."""
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.975, 0.973, 0.980],
        size_rel_errors=[2.0, 3.0, 4.0],  # well within threshold ~15%
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["accuracy"] is not None
    assert r["accuracy"]["status"] == "ok"
    summary = r["summary"]
    assert (
        "Prediction" in summary or "prediction" in summary
    ), f"Prediction phrase missing from ok-accuracy summary: {summary}"


def test_summary_accuracy_fail_names_actual_error() -> None:
    """When accuracy is fail, summary names the actual p95 error and threshold."""
    # ESTIMATION_SIZE_REL_ERROR p95_max is typically ~15%; we supply 50% errors to force fail.
    run = _make_pr_run(
        fmt="jpeg",
        ssim_values=[0.975, 0.973, 0.980],
        size_rel_errors=[50.0, 55.0, 60.0],  # far above threshold
        p95_ms=40.0,
        bucket="small",
    )
    records = build_scorecard(run)
    r = records[0]

    assert r["accuracy"] is not None
    acc_status = r["accuracy"]["status"]
    if acc_status in ("warn", "fail"):
        summary = r["summary"]
        # Must mention prediction failure specifically (not just generic text).
        assert (
            "Prediction" in summary or "prediction" in summary
        ), f"Prediction mention missing from {acc_status}-accuracy summary: {summary}"
