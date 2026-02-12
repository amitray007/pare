"""Benchmark report generation.

Formats benchmark results into readable tables grouped by format,
with summary statistics, estimation accuracy analysis, and HTML output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

from benchmarks.runner import BenchmarkResult, BenchmarkSuite

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_size(n: int) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.1f}MB"
    if n >= 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n}B"


def _fmt_speed(bytes_per_second: float) -> str:
    mb_s = bytes_per_second / 1_048_576
    if mb_s >= 1:
        return f"{mb_s:.1f}"
    return f"{mb_s:.2f}"


def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "#" * filled + "-" * (width - filled)


def _group_by_preset_and_fmt(
    results: list[BenchmarkResult],
) -> dict[str, dict[str, list[BenchmarkResult]]]:
    """Group results by preset name, then by format."""
    grouped: dict[str, dict[str, list[BenchmarkResult]]] = {}
    for r in results:
        preset = r.preset_name or "default"
        grouped.setdefault(preset, {})
        grouped[preset].setdefault(r.case.fmt, []).append(r)
    return grouped


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Console report (original, updated for presets)
# ---------------------------------------------------------------------------


def print_report(suite: BenchmarkSuite, file=None) -> None:
    """Print a formatted benchmark report."""
    out = file or sys.stdout

    print("\n" + "=" * 100, file=out)
    print("  PARE BENCHMARK REPORT", file=out)
    print("=" * 100, file=out)
    presets_label = ", ".join(suite.presets_used) if suite.presets_used else "default"
    print(
        f"  Presets: {presets_label}  |  Cases: {suite.cases_run}  |  "
        f"Failed: {suite.cases_failed}  |  Time: {suite.total_time_s:.1f}s",
        file=out,
    )
    print("=" * 100, file=out)

    by_preset_fmt = _group_by_preset_and_fmt(suite.results)

    for preset_name in suite.presets_used or sorted(by_preset_fmt.keys()):
        fmt_groups = by_preset_fmt.get(preset_name, {})
        print(f"\n{'*' * 100}", file=out)
        print(f"  PRESET: {preset_name}", file=out)
        print(f"{'*' * 100}", file=out)

        for fmt in sorted(fmt_groups.keys()):
            _print_format_table(fmt, fmt_groups[fmt], out)

    _print_summary(suite, out)
    _print_estimation_accuracy(suite, out)


def _print_format_table(fmt: str, results: list[BenchmarkResult], out) -> None:
    print(f"\n--- {fmt.upper()} ({len(results)} cases) ---", file=out)

    header = (
        f"  {'Name':<40} {'Orig':>8} {'Opt':>8} {'Reduc':>7} "
        f"{'MB/s':>6} {'Time':>8}  {'Method'}"
    )
    print(header, file=out)
    print("  " + "-" * 95, file=out)

    for r in results:
        if r.opt_error:
            print(f"  {r.case.name:<40} {'ERROR':>8} {r.opt_error}", file=out)
            continue

        print(
            f"  {r.case.name:<40} "
            f"{_fmt_size(len(r.case.data)):>8} "
            f"{_fmt_size(r.optimized_size):>8} "
            f"{r.reduction_pct:>6.1f}% "
            f"{_fmt_speed(r.bytes_per_second):>6} "
            f"{r.opt_time_ms:>7.0f}ms "
            f" {r.method}",
            file=out,
        )

    valid = [r for r in results if not r.opt_error]
    if valid:
        avg_reduction = sum(r.reduction_pct for r in valid) / len(valid)
        max_reduction = max(r.reduction_pct for r in valid)
        avg_time = sum(r.opt_time_ms for r in valid) / len(valid)
        total_orig = sum(len(r.case.data) for r in valid)
        total_opt = sum(r.optimized_size for r in valid)
        overall_pct = (1 - total_opt / total_orig) * 100 if total_orig else 0

        print(
            f"\n  Avg reduction: {avg_reduction:.1f}%  |  Max: {max_reduction:.1f}%  |  "
            f"Weighted: {overall_pct:.1f}%  |  Avg time: {avg_time:.0f}ms",
            file=out,
        )


def _print_summary(suite: BenchmarkSuite, out) -> None:
    print(f"\n{'=' * 100}", file=out)
    print("  SUMMARY BY FORMAT", file=out)
    print(f"{'=' * 100}", file=out)

    by_fmt: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in suite.results:
        if not r.opt_error:
            by_fmt[r.case.fmt].append(r)

    print(
        f"  {'Format':<10} {'Cases':>6} {'Avg %':>7} {'Max %':>7} {'Weighted %':>11} {'Avg ms':>8}",
        file=out,
    )
    print("  " + "-" * 52, file=out)

    for fmt in sorted(by_fmt.keys()):
        results = by_fmt[fmt]
        avg_pct = sum(r.reduction_pct for r in results) / len(results)
        max_pct = max(r.reduction_pct for r in results)
        total_orig = sum(len(r.case.data) for r in results)
        total_opt = sum(r.optimized_size for r in results)
        weighted = (1 - total_opt / total_orig) * 100 if total_orig else 0
        avg_ms = sum(r.opt_time_ms for r in results) / len(results)

        print(
            f"  {fmt.upper():<10} {len(results):>6} {avg_pct:>6.1f}% {max_pct:>6.1f}% "
            f"{weighted:>10.1f}% {avg_ms:>7.0f}ms",
            file=out,
        )


def _print_estimation_accuracy(suite: BenchmarkSuite, out) -> None:
    print(f"\n{'=' * 100}", file=out)
    print("  ESTIMATION ACCURACY", file=out)
    print(f"{'=' * 100}", file=out)

    by_fmt: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in suite.results:
        if not r.opt_error and not r.est_error:
            by_fmt[r.case.fmt].append(r)

    print(
        f"  {'Format':<10} {'Cases':>6} {'Avg Err':>8} {'Max Err':>8} {'Avg Est%':>9} {'Avg Act%':>9}",
        file=out,
    )
    print("  " + "-" * 54, file=out)

    for fmt in sorted(by_fmt.keys()):
        results = by_fmt[fmt]
        if not results:
            continue
        avg_err = sum(r.est_error_pct for r in results) / len(results)
        max_err = max(r.est_error_pct for r in results)
        avg_est = sum(r.est_reduction_pct for r in results) / len(results)
        avg_act = sum(r.reduction_pct for r in results) / len(results)

        print(
            f"  {fmt.upper():<10} {len(results):>6} {avg_err:>7.1f}% {max_err:>7.1f}% "
            f"{avg_est:>8.1f}% {avg_act:>8.1f}%",
            file=out,
        )

    all_valid = [r for r in suite.results if not r.opt_error and not r.est_error]
    if all_valid:
        worst = sorted(all_valid, key=lambda r: r.est_error_pct, reverse=True)[:10]
        print("\n  Top 10 worst estimates:", file=out)
        print(f"  {'Name':<40} {'Est%':>6} {'Act%':>6} {'Err%':>6}", file=out)
        print("  " + "-" * 60, file=out)
        for r in worst:
            print(
                f"  {r.case.name:<40} {r.est_reduction_pct:>5.1f}% {r.reduction_pct:>5.1f}% "
                f"{r.est_error_pct:>5.1f}%",
                file=out,
            )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(suite: BenchmarkSuite) -> str:
    """Export results as JSON for programmatic analysis."""
    timestamp = datetime.now(timezone.utc).isoformat()
    records = []
    for r in suite.results:
        records.append(
            {
                "name": r.case.name,
                "format": r.case.fmt,
                "category": r.case.category,
                "content": r.case.content,
                "quality": r.case.quality,
                "preset": r.preset_name,
                "original_size": len(r.case.data),
                "optimized_size": r.optimized_size,
                "reduction_pct": round(r.reduction_pct, 2),
                "method": r.method,
                "opt_time_ms": round(r.opt_time_ms, 1),
                "bytes_per_second": round(r.bytes_per_second, 0),
                "opt_error": r.opt_error or None,
                "est_reduction_pct": round(r.est_reduction_pct, 2),
                "est_potential": r.est_potential,
                "est_confidence": r.est_confidence,
                "est_time_ms": round(r.est_time_ms, 1),
                "est_error": r.est_error or None,
                "est_accuracy_error_pct": round(r.est_error_pct, 1),
            }
        )
    return json.dumps(
        {
            "timestamp": timestamp,
            "git_commit": _git_commit_hash(),
            "presets_used": suite.presets_used,
            "cases_run": suite.cases_run,
            "cases_failed": suite.cases_failed,
            "total_time_s": round(suite.total_time_s, 1),
            "results": records,
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pare Benchmark Report</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --text: #e6edf3; --text-dim: #8b949e; --accent: #58a6ff;
    --green: #3fb950; --red: #f85149; --yellow: #d29922;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); padding: 24px; line-height: 1.5;
  }}
  h1 {{ color: var(--accent); margin-bottom: 8px; }}
  .meta {{ color: var(--text-dim); margin-bottom: 24px; font-size: 14px; }}
  .meta code {{ background: var(--surface); padding: 2px 6px; border-radius: 4px; }}
  h2 {{ color: var(--accent); margin: 32px 0 12px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
  h3 {{ color: var(--text); margin: 20px 0 8px; }}
  table {{
    width: 100%; border-collapse: collapse; margin-bottom: 16px;
    font-size: 13px; font-family: 'Cascadia Code', 'Fira Code', monospace;
  }}
  th, td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--border); }}
  th {{ background: var(--surface); color: var(--text-dim); font-weight: 600; position: sticky; top: 0; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover td {{ background: rgba(88,166,255,0.04); }}
  .good {{ color: var(--green); }}
  .warn {{ color: var(--yellow); }}
  .bad {{ color: var(--red); }}
  .error-cell {{ color: var(--red); font-style: italic; }}
  .section {{ background: var(--surface); border-radius: 8px; padding: 16px 20px; margin-bottom: 24px; }}
  .bar {{ display: inline-block; height: 12px; border-radius: 2px; background: var(--accent); opacity: 0.7; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px; }}
  .stat-card {{
    background: var(--surface); border-radius: 8px; padding: 16px;
    border: 1px solid var(--border);
  }}
  .stat-card .label {{ color: var(--text-dim); font-size: 12px; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 24px; font-weight: bold; color: var(--accent); }}
</style>
</head>
<body>
<h1>Pare Benchmark Report</h1>
<div class="meta">
  Generated: <code>{timestamp}</code> &nbsp;|&nbsp;
  Git: <code>{git_hash}</code> &nbsp;|&nbsp;
  Presets: <code>{presets}</code> &nbsp;|&nbsp;
  Cases: <code>{cases_run}</code> &nbsp;|&nbsp;
  Failed: <code>{cases_failed}</code> &nbsp;|&nbsp;
  Time: <code>{total_time:.1f}s</code>
</div>

{summary_cards}

{preset_sections}

{summary_table}

{estimation_section}

</body>
</html>
"""


def generate_html_report(suite: BenchmarkSuite) -> str:
    """Generate a full HTML benchmark report."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    git_hash = _git_commit_hash()
    presets_label = ", ".join(suite.presets_used) if suite.presets_used else "default"

    by_preset_fmt = _group_by_preset_and_fmt(suite.results)

    # Summary cards
    valid = [r for r in suite.results if not r.opt_error]
    total_orig = sum(len(r.case.data) for r in valid)
    total_opt = sum(r.optimized_size for r in valid)
    overall_reduction = (1 - total_opt / total_orig) * 100 if total_orig else 0
    avg_time = sum(r.opt_time_ms for r in valid) / len(valid) if valid else 0
    avg_speed = sum(r.bytes_per_second for r in valid) / len(valid) if valid else 0

    summary_cards = _html_summary_cards(
        overall_reduction,
        avg_time,
        avg_speed,
        len(valid),
        suite.cases_failed,
    )

    # Per-preset sections
    preset_sections = []
    for preset_name in suite.presets_used or sorted(by_preset_fmt.keys()):
        fmt_groups = by_preset_fmt.get(preset_name, {})
        preset_sections.append(_html_preset_section(preset_name, fmt_groups))

    # Summary table
    summary_table = _html_summary_table(suite)

    # Estimation accuracy
    estimation_section = _html_estimation_section(suite)

    return _HTML_TEMPLATE.format(
        timestamp=timestamp,
        git_hash=git_hash,
        presets=presets_label,
        cases_run=suite.cases_run,
        cases_failed=suite.cases_failed,
        total_time=suite.total_time_s,
        summary_cards=summary_cards,
        preset_sections="\n".join(preset_sections),
        summary_table=summary_table,
        estimation_section=estimation_section,
    )


def _html_summary_cards(
    overall_pct: float,
    avg_ms: float,
    avg_speed: float,
    valid: int,
    failed: int,
) -> str:
    return f"""<div class="summary-grid">
  <div class="stat-card"><div class="label">Overall Reduction</div><div class="value">{overall_pct:.1f}%</div></div>
  <div class="stat-card"><div class="label">Avg Time</div><div class="value">{avg_ms:.0f}ms</div></div>
  <div class="stat-card"><div class="label">Avg Throughput</div><div class="value">{avg_speed / 1_048_576:.1f} MB/s</div></div>
  <div class="stat-card"><div class="label">Cases Passed</div><div class="value">{valid}</div></div>
  <div class="stat-card"><div class="label">Cases Failed</div><div class="value {'bad' if failed else 'good'}">{failed}</div></div>
</div>"""


def _html_preset_section(preset_name: str, fmt_groups: dict[str, list[BenchmarkResult]]) -> str:
    parts = [f"<h2>Preset: {preset_name}</h2>"]
    for fmt in sorted(fmt_groups.keys()):
        results = fmt_groups[fmt]
        parts.append(f'<div class="section"><h3>{fmt.upper()} ({len(results)} cases)</h3>')
        parts.append("<table><thead><tr>")
        parts.append(
            "<th>Name</th><th>Dimensions</th><th>Original</th><th>Optimized</th>"
            "<th>Reduction</th><th>MB/s</th><th>Time</th><th>Method</th>"
            "<th>Est%</th><th>Est Error</th>"
        )
        parts.append("</tr></thead><tbody>")

        for r in results:
            if r.opt_error:
                parts.append(
                    f"<tr><td>{_h(r.case.name)}</td>"
                    f'<td colspan="9" class="error-cell">ERROR: {_h(r.opt_error)}</td></tr>'
                )
                continue

            reduction_cls = _reduction_class(r.reduction_pct)
            est_err_cls = _est_error_class(r.est_error_pct)
            dims = r.case.name.split()[-1] if "x" in r.case.name.split()[-1] else "-"

            parts.append(
                f"<tr>"
                f"<td>{_h(r.case.name)}</td>"
                f"<td>{dims}</td>"
                f"<td>{_fmt_size(len(r.case.data))}</td>"
                f"<td>{_fmt_size(r.optimized_size)}</td>"
                f'<td class="{reduction_cls}">{r.reduction_pct:.1f}%'
                f' <span class="bar" style="width:{max(1, int(r.reduction_pct))}px"></span></td>'
                f"<td>{_fmt_speed(r.bytes_per_second)}</td>"
                f"<td>{r.opt_time_ms:.0f}ms</td>"
                f"<td>{_h(r.method)}</td>"
                f"<td>{r.est_reduction_pct:.1f}%</td>"
                f'<td class="{est_err_cls}">{r.est_error_pct:.1f}%</td>'
                f"</tr>"
            )

        parts.append("</tbody></table>")

        # Per-format stats row
        valid = [r for r in results if not r.opt_error]
        if valid:
            avg_red = sum(r.reduction_pct for r in valid) / len(valid)
            max_red = max(r.reduction_pct for r in valid)
            avg_ms = sum(r.opt_time_ms for r in valid) / len(valid)
            parts.append(
                f'<p style="color:var(--text-dim);font-size:12px;margin-top:4px">'
                f"Avg: {avg_red:.1f}% | Max: {max_red:.1f}% | Avg time: {avg_ms:.0f}ms</p>"
            )
        parts.append("</div>")

    return "\n".join(parts)


def _html_summary_table(suite: BenchmarkSuite) -> str:
    by_fmt: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in suite.results:
        if not r.opt_error:
            by_fmt[r.case.fmt].append(r)

    rows = []
    for fmt in sorted(by_fmt.keys()):
        results = by_fmt[fmt]
        avg_pct = sum(r.reduction_pct for r in results) / len(results)
        max_pct = max(r.reduction_pct for r in results)
        total_orig = sum(len(r.case.data) for r in results)
        total_opt = sum(r.optimized_size for r in results)
        weighted = (1 - total_opt / total_orig) * 100 if total_orig else 0
        avg_ms = sum(r.opt_time_ms for r in results) / len(results)
        avg_speed = sum(r.bytes_per_second for r in results) / len(results)

        rows.append(
            f"<tr><td>{fmt.upper()}</td><td>{len(results)}</td>"
            f"<td>{avg_pct:.1f}%</td><td>{max_pct:.1f}%</td>"
            f"<td>{weighted:.1f}%</td><td>{avg_ms:.0f}ms</td>"
            f"<td>{_fmt_speed(avg_speed)}</td></tr>"
        )

    return (
        "<h2>Summary by Format</h2>"
        '<div class="section"><table><thead><tr>'
        "<th>Format</th><th>Cases</th><th>Avg %</th><th>Max %</th>"
        "<th>Weighted %</th><th>Avg Time</th><th>Avg MB/s</th>"
        "</tr></thead><tbody>" + "\n".join(rows) + "</tbody></table></div>"
    )


def _html_estimation_section(suite: BenchmarkSuite) -> str:
    by_fmt: dict[str, list[BenchmarkResult]] = defaultdict(list)
    for r in suite.results:
        if not r.opt_error and not r.est_error:
            by_fmt[r.case.fmt].append(r)

    rows = []
    for fmt in sorted(by_fmt.keys()):
        results = by_fmt[fmt]
        if not results:
            continue
        avg_err = sum(r.est_error_pct for r in results) / len(results)
        max_err = max(r.est_error_pct for r in results)
        avg_est = sum(r.est_reduction_pct for r in results) / len(results)
        avg_act = sum(r.reduction_pct for r in results) / len(results)

        rows.append(
            f"<tr><td>{fmt.upper()}</td><td>{len(results)}</td>"
            f'<td class="{_est_error_class(avg_err)}">{avg_err:.1f}%</td>'
            f'<td class="{_est_error_class(max_err)}">{max_err:.1f}%</td>'
            f"<td>{avg_est:.1f}%</td><td>{avg_act:.1f}%</td></tr>"
        )

    # Worst estimates
    all_valid = [r for r in suite.results if not r.opt_error and not r.est_error]
    worst_rows = []
    if all_valid:
        worst = sorted(all_valid, key=lambda r: r.est_error_pct, reverse=True)[:10]
        for r in worst:
            worst_rows.append(
                f"<tr><td>{_h(r.case.name)}</td><td>{r.preset_name}</td>"
                f"<td>{r.est_reduction_pct:.1f}%</td>"
                f"<td>{r.reduction_pct:.1f}%</td>"
                f'<td class="{_est_error_class(r.est_error_pct)}">{r.est_error_pct:.1f}%</td></tr>'
            )

    return (
        "<h2>Estimation Accuracy</h2>"
        '<div class="section"><table><thead><tr>'
        "<th>Format</th><th>Cases</th><th>Avg Error</th><th>Max Error</th>"
        "<th>Avg Est%</th><th>Avg Act%</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
        + (
            "<h3>Top 10 Worst Estimates</h3>"
            "<table><thead><tr>"
            "<th>Name</th><th>Preset</th><th>Est%</th><th>Act%</th><th>Error</th>"
            "</tr></thead><tbody>" + "\n".join(worst_rows) + "</tbody></table>"
            if worst_rows
            else ""
        )
        + "</div>"
    )


def _h(text: str) -> str:
    """Minimal HTML escaping."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _reduction_class(pct: float) -> str:
    if pct >= 30:
        return "good"
    if pct >= 10:
        return "warn"
    return "bad"


def _est_error_class(pct: float) -> str:
    if pct <= 5:
        return "good"
    if pct <= 15:
        return "warn"
    return "bad"
