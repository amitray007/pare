"""Benchmark CLI entry point.

Usage:
    python -m benchmarks.run                          # Run all presets, save report
    python -m benchmarks.run --preset high --fmt png   # Filtered run
    python -m benchmarks.run --json                    # JSON to stdout
    python -m benchmarks.run --compare                 # Compare last two runs
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.constants import PRESETS_BY_NAME
from benchmarks.report import export_json, generate_html_report, print_report
from benchmarks.runner import run_suite

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _ensure_reports_dir() -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    return REPORTS_DIR


def _timestamp_stem() -> str:
    return datetime.now(timezone.utc).strftime("benchmark-%Y%m%d-%H%M%S")


def _save_reports(suite) -> tuple[Path, Path]:
    """Save HTML + JSON reports and return their paths."""
    reports_dir = _ensure_reports_dir()
    stem = _timestamp_stem()

    html_path = reports_dir / f"{stem}.html"
    json_path = reports_dir / f"{stem}.json"

    html_path.write_text(generate_html_report(suite), encoding="utf-8")
    json_path.write_text(export_json(suite), encoding="utf-8")

    return html_path, json_path


def _find_latest_json_reports(n: int = 2) -> list[Path]:
    """Find the n most recent JSON reports in the reports directory."""
    if not REPORTS_DIR.exists():
        return []
    jsons = sorted(REPORTS_DIR.glob("benchmark-*.json"), reverse=True)
    return jsons[:n]


def _compare_reports(path_a: Path, path_b: Path) -> None:
    """Print a delta summary between two JSON benchmark reports."""
    a = json.loads(path_a.read_text(encoding="utf-8"))
    b = json.loads(path_b.read_text(encoding="utf-8"))

    print("\nComparing benchmarks:")
    print(f"  OLD: {path_b.name}  ({b.get('timestamp', '?')}  git:{b.get('git_commit', '?')})")
    print(f"  NEW: {path_a.name}  ({a.get('timestamp', '?')}  git:{a.get('git_commit', '?')})")
    print("=" * 80)

    # Index results by (name, preset) for matching
    def _index(data):
        idx = {}
        for r in data.get("results", []):
            key = (r["name"], r.get("preset", ""))
            idx[key] = r
        return idx

    old_idx = _index(b)
    new_idx = _index(a)

    # Compare per-format aggregate
    old_by_fmt, new_by_fmt = {}, {}
    for key, r in old_idx.items():
        fmt = r["format"]
        old_by_fmt.setdefault(fmt, []).append(r)
    for key, r in new_idx.items():
        fmt = r["format"]
        new_by_fmt.setdefault(fmt, []).append(r)

    all_fmts = sorted(set(list(old_by_fmt.keys()) + list(new_by_fmt.keys())))

    print(
        f"\n  {'Format':<10} {'Old Avg%':>9} {'New Avg%':>9} {'Delta':>8} {'Old ms':>8} {'New ms':>8} {'Delta':>8}"
    )
    print("  " + "-" * 65)

    for fmt in all_fmts:
        old_results = [r for r in old_by_fmt.get(fmt, []) if not r.get("opt_error")]
        new_results = [r for r in new_by_fmt.get(fmt, []) if not r.get("opt_error")]
        if not old_results or not new_results:
            continue

        old_avg = sum(r["reduction_pct"] for r in old_results) / len(old_results)
        new_avg = sum(r["reduction_pct"] for r in new_results) / len(new_results)
        old_ms = sum(r["opt_time_ms"] for r in old_results) / len(old_results)
        new_ms = sum(r["opt_time_ms"] for r in new_results) / len(new_results)

        d_pct = new_avg - old_avg
        d_ms = new_ms - old_ms
        sign_pct = "+" if d_pct >= 0 else ""
        sign_ms = "+" if d_ms >= 0 else ""

        print(
            f"  {fmt.upper():<10} {old_avg:>8.1f}% {new_avg:>8.1f}% {sign_pct}{d_pct:>6.1f}% "
            f"{old_ms:>7.0f}ms {new_ms:>7.0f}ms {sign_ms}{d_ms:>6.0f}ms"
        )

    # Estimation accuracy delta
    old_valid = [
        r for r in b.get("results", []) if not r.get("opt_error") and not r.get("est_error")
    ]
    new_valid = [
        r for r in a.get("results", []) if not r.get("opt_error") and not r.get("est_error")
    ]
    if old_valid and new_valid:
        old_est_err = sum(r["est_accuracy_error_pct"] for r in old_valid) / len(old_valid)
        new_est_err = sum(r["est_accuracy_error_pct"] for r in new_valid) / len(new_valid)
        d_est = new_est_err - old_est_err
        sign = "+" if d_est >= 0 else ""
        print(
            f"\n  Estimation avg error: {old_est_err:.1f}% -> {new_est_err:.1f}% ({sign}{d_est:.1f}%)"
        )


def main():
    parser = argparse.ArgumentParser(description="Pare image optimization benchmarks")
    parser.add_argument("--fmt", help="Filter by format (png, jpeg, webp, gif, svg, bmp, tiff, avif, heic, jxl)")
    parser.add_argument(
        "--category",
        help="Filter by size category (tiny, small-l, medium-l, large-l, square, vector)",
    )
    parser.add_argument("--preset", help="Run only this preset (high, medium, low)")
    parser.add_argument(
        "--json", action="store_true", help="Output JSON to stdout instead of table"
    )
    parser.add_argument("-o", "--output", help="Write output to file instead of stdout")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving reports to reports/ folder"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare the last two benchmark runs"
    )
    args = parser.parse_args()

    # --compare mode
    if args.compare:
        reports = _find_latest_json_reports(2)
        if len(reports) < 2:
            print(
                f"Need at least 2 reports in {REPORTS_DIR}/ to compare " f"(found {len(reports)})",
                file=sys.stderr,
            )
            sys.exit(1)
        _compare_reports(reports[0], reports[1])
        return

    # Resolve presets
    presets = None
    if args.preset:
        name = args.preset.upper()
        if name not in PRESETS_BY_NAME:
            parser.error(
                f"Unknown preset '{args.preset}'. Choose from: {', '.join(PRESETS_BY_NAME)}"
            )
        presets = [PRESETS_BY_NAME[name]]

    suite = asyncio.run(
        run_suite(
            fmt_filter=args.fmt,
            category_filter=args.category,
            presets=presets,
            progress=not args.no_progress,
        )
    )

    # Output to stdout/file
    if args.json:
        text = export_json(suite)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Results written to {args.output}", file=sys.stderr)
        else:
            print(text)
    else:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                print_report(suite, file=f)
            print(f"Report written to {args.output}", file=sys.stderr)
        else:
            print_report(suite)

    # Save persistent reports unless disabled
    if not args.no_save:
        html_path, json_path = _save_reports(suite)
        print("\n  Reports saved:", file=sys.stderr)
        print(f"    HTML: {html_path}", file=sys.stderr)
        print(f"    JSON: {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
