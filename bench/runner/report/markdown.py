"""Markdown report writer for PR comments and step summaries.

Tables are grouped by `(format, preset)` so format-specific regressions
stand out at a glance. Numbers are formatted with `~3` significant
digits — enough to rank cases without drowning in noise.
"""

from __future__ import annotations

from typing import Any

from bench.runner.stats import CaseStats


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    if v >= 100:
        return f"{v:.0f}ms"
    if v >= 10:
        return f"{v:.1f}ms"
    return f"{v:.2f}ms"


def _fmt_kb(v: int | None) -> str:
    if v is None:
        return "-"
    if v >= 1024 * 1024:
        return f"{v / (1024 * 1024):.1f}GB"
    if v >= 1024:
        return f"{v / 1024:.1f}MB"
    return f"{v}KB"


def render_run(run: dict[str, Any]) -> str:
    """Render a full run JSON payload as Markdown."""
    out: list[str] = []
    out.append(f"# Pare bench — `{run['mode']}` mode")
    out.append("")
    out.append(_render_metadata(run))
    out.append("")

    errors = [it for it in run["iterations"] if "error" in it]
    if errors:
        out.append(f"## Errors ({len(errors)})")
        out.append("")
        for e in errors[:10]:
            out.append(f"- `{e['case_id']}`: {e['error']}")
        if len(errors) > 10:
            out.append(f"- … {len(errors) - 10} more")
        out.append("")

    stats = _stats_from_run(run)
    if not stats:
        out.append("_No successful iterations to report._")
        return "\n".join(out)

    out.append("## Per-case results")
    out.append("")
    out.append(_render_stats_table(stats))

    if run["mode"] == "memory":
        out.append("")
        out.append("## Memory headline (peak RSS — capacity planning)")
        out.append("")
        out.append(_render_memory_table(stats))

    return "\n".join(out)


def _render_metadata(run: dict[str, Any]) -> str:
    git = run.get("git", {})
    git_str = f"{git.get('branch', '?')} @ {git.get('commit', '?')[:8]}" + (
        " (dirty)" if git.get("dirty") else ""
    )
    annotations = run.get("annotations") or {}
    ann_lines = "\n".join(f"- **{k}**: {v}" for k, v in annotations.items())

    cfg = run.get("config", {})
    cfg_pairs = ", ".join(f"{k}={v}" for k, v in cfg.items())

    lines = [
        f"- **timestamp**: {run['timestamp']}",
        f"- **git**: {git_str}",
        f"- **host**: {run['host']['platform']} ({run['host']['cpu_count']} CPUs)",
        f"- **manifest**: {run['manifest']['name']} (`{run['manifest']['sha256'][:12]}`)",
        f"- **config**: {cfg_pairs}",
    ]
    if ann_lines:
        lines.append(ann_lines)
    return "\n".join(lines)


def _render_stats_table(stats: list[CaseStats]) -> str:
    lines = [
        "| case_id | iter | p50 | p95 | median±MAD | child CPU p50 | parallel | RSS p95 | red% | method |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(stats, key=lambda x: (x.format, x.preset, x.case_id)):
        lines.append(
            "| `{cid}` | {it} | {p50} | {p95} | {med}±{mad} | {ccpu} | {par:.2f}× | {rss} | {red:.1f}% | {meth} |".format(
                cid=s.case_id,
                it=s.iterations,
                p50=_fmt_ms(s.p50_ms),
                p95=_fmt_ms(s.p95_ms),
                med=_fmt_ms(s.median_ms),
                mad=_fmt_ms(s.mad_ms),
                ccpu=_fmt_ms(s.children_cpu_p50_ms),
                par=s.parallelism_p50,
                rss=_fmt_kb(s.children_peak_rss_p95_kb),
                red=s.reduction_pct,
                meth=(s.method or "-")[:24],
            )
        )
    return "\n".join(lines)


def _render_memory_table(stats: list[CaseStats]) -> str:
    lines = [
        "| case_id | parent peak | children peak | py heap peak |",
        "|---|---|---|---|",
    ]
    for s in sorted(
        stats, key=lambda x: -max(x.parent_peak_rss_p95_kb, x.children_peak_rss_p95_kb)
    ):
        lines.append(
            f"| `{s.case_id}` | {_fmt_kb(s.parent_peak_rss_p95_kb)} "
            f"| {_fmt_kb(s.children_peak_rss_p95_kb)} "
            f"| {_fmt_kb(s.py_peak_alloc_p95_kb)} |"
        )
    return "\n".join(lines)


def _stats_from_run(run: dict[str, Any]) -> list[CaseStats]:
    """Reconstruct CaseStats from the JSON payload's `stats` array."""
    rebuilt = []
    for s in run.get("stats", []):
        rebuilt.append(CaseStats(**s))
    return rebuilt
