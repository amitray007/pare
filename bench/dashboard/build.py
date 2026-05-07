"""Build a static HTML dashboard from the git history of reports/baseline.core.json.

Reads every prior version of the baseline via ``git log --follow``, extracts
per-format trend data, and renders ``index.html`` + ``data/history.json`` into
``dashboard/dist/``.  Idempotent — safe to run repeatedly; output dir is wiped
each run.

The dashboard also renders a per-format **scorecard** at the top of the page,
reading the current baseline (or a file passed via ``--baseline``) to show
compression, speed, quality, and accuracy at a glance.

A ``quality-samples/index.html`` sub-page is generated when ``--with-samples``
is set (the default).  It shows the worst-quality case per lossy format with
base64-embedded before/after PNG thumbnails.

CLI::

    python -m bench.dashboard.build [--out-dir dashboard/dist] [--limit 100]
    python -m bench.dashboard.build --repo /path/to/repo --out-dir /tmp/dash
    python -m bench.dashboard.build --baseline /tmp/pr_smoke.json --out-dir /tmp/dash
    python -m bench.dashboard.build --baseline /tmp/pr.json --out-dir /tmp/dash --no-with-samples
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

from bench.dashboard.samples import build_sample_records
from bench.dashboard.scorecard import build_kpis, build_scorecard, load_scorecard_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_PATH = "reports/baseline.core.json"
_SCHEMA_VERSION = 2

# All formats the corpus currently covers (used as column order in history).
KNOWN_FORMATS = [
    "jpeg",
    "png",
    "webp",
    "avif",
    "heic",
    "jxl",
    "gif",
    "apng",
    "bmp",
    "tiff",
    "svg",
    "svgz",
]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def list_commits(repo_root: Path, file_path: str, limit: int) -> list[dict[str, str]]:
    """Return commits that touched *file_path*, oldest-first.

    Each entry: ``{sha, unix_ts, subject}``.
    """
    result = _git(
        "log",
        "--follow",
        "--format=%H %ct %s",
        "--",
        file_path,
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    commits: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        sha = parts[0]
        unix_ts = parts[1]
        subject = parts[2] if len(parts) > 2 else ""
        commits.append({"sha": sha, "unix_ts": unix_ts, "subject": subject})

    # git log returns newest-first; we want oldest-first for x-axis flow.
    commits.reverse()

    # Apply limit from the newest end (keep the last N after reversal).
    if limit > 0 and len(commits) > limit:
        commits = commits[-limit:]

    return commits


def show_file(repo_root: Path, sha: str, file_path: str) -> str | None:
    """Return the content of *file_path* at *sha*, or None if missing."""
    result = _git("show", f"{sha}:{file_path}", cwd=repo_root, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


# ---------------------------------------------------------------------------
# Run JSON parsing
# ---------------------------------------------------------------------------


def _parse_run(raw: str) -> dict[str, Any] | None:
    """Parse a run JSON string. Returns None on schema mismatch or parse error."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return None
    return data


def _aggregate_by_format(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group *stats* entries by format and compute per-format medians.

    Returns a dict keyed by format name::

        {
            "jpeg": {"p50_ms": 23.4, "p95_ms": 45.1, "peak_rss_kb": 70000, "n": 21},
            ...
        }
    """
    by_fmt: dict[str, list[dict[str, Any]]] = {}
    for s in stats:
        fmt = s.get("format", "unknown")
        by_fmt.setdefault(fmt, []).append(s)

    result: dict[str, dict[str, Any]] = {}
    for fmt, entries in by_fmt.items():
        p50_list = [e["p50_ms"] for e in entries if "p50_ms" in e]
        p95_list = [e["p95_ms"] for e in entries if "p95_ms" in e]
        # Peak RSS = max of (parent + children) across cases within this format.
        # We use the p95 values from each stat row to stay consistent with
        # what CaseStats records.
        rss_list = [
            e.get("parent_peak_rss_p95_kb", 0) + e.get("children_peak_rss_p95_kb", 0)
            for e in entries
            if "parent_peak_rss_p95_kb" in e
        ]
        result[fmt] = {
            "p50_ms": round(median(p50_list), 3) if p50_list else 0.0,
            "p95_ms": round(median(p95_list), 3) if p95_list else 0.0,
            "peak_rss_kb": int(median(rss_list)) if rss_list else 0,
            "n": len(entries),
        }
    return result


def extract_run_record(
    sha: str,
    unix_ts: str,
    subject: str,
    run_data: dict[str, Any],
) -> dict[str, Any]:
    """Build the history.json run record from a parsed run JSON."""
    stats: list[dict[str, Any]] = run_data.get("stats", [])
    iterations: list[dict[str, Any]] = run_data.get("iterations", [])

    n_errors = sum(1 for it in iterations if it.get("error") is not None)
    n_cases = len(stats)

    ts_int = int(unix_ts)
    iso_date = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc).strftime("%Y-%m-%d")

    # Short SHA: 7 chars is conventional.
    short_sha = sha[:7]

    return {
        "sha": sha,
        "short_sha": short_sha,
        "timestamp_unix": ts_int,
        "iso_date": iso_date,
        "subject": subject,
        "mode": run_data.get("mode", "quick"),
        "n_cases": n_cases,
        "n_errors": n_errors,
        "by_format": _aggregate_by_format(stats),
    }


# ---------------------------------------------------------------------------
# History builder
# ---------------------------------------------------------------------------


def build_history(
    repo_root: Path,
    limit: int = 100,
) -> dict[str, Any]:
    """Walk git log and collect one record per historical baseline commit.

    Returns the full ``history.json`` payload (not yet serialized).
    """
    commits = list_commits(repo_root, BASELINE_PATH, limit)
    runs: list[dict[str, Any]] = []

    for commit in commits:
        sha = commit["sha"]
        raw = show_file(repo_root, sha, BASELINE_PATH)
        if raw is None:
            # File didn't exist at this commit — shouldn't happen with
            # ``--follow`` but guard anyway.
            continue
        run_data = _parse_run(raw)
        if run_data is None:
            # Pre-schema-v2 or malformed — skip gracefully.
            continue
        record = extract_run_record(
            sha=sha,
            unix_ts=commit["unix_ts"],
            subject=commit["subject"],
            run_data=run_data,
        )
        runs.append(record)

    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    latest_sha = runs[-1]["short_sha"] if runs else "none"

    return {
        "generated_at": now_iso,
        "manifest": "core",
        "latest_sha": latest_sha,
        "runs": runs,  # oldest-first; guaranteed by list_commits()
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _template_dir() -> Path:
    return Path(__file__).parent / "template"


def render_output(
    history: dict[str, Any],
    out_dir: Path,
    scorecard_run: dict[str, Any] | None = None,
    *,
    with_samples: bool = True,
    corpus_root: Path | None = None,
) -> None:
    """Write ``index.html`` and ``data/history.json`` to *out_dir*.

    Wipes *out_dir* first so repeated runs stay idempotent.

    If *scorecard_run* is provided (a parsed run JSON), scorecard data is
    embedded inline in the HTML as a JS variable for immediate first-paint
    rendering without an extra network fetch.

    If *with_samples* is True (the default), also renders
    ``quality-samples/index.html`` with per-format worst-case visual samples.
    Pass *corpus_root* so thumbnails can be generated; if None, numeric-only
    cards are rendered.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "data").mkdir()

    # Build scorecard payload (or empty if no data).
    if scorecard_run is not None:
        scorecard_records = build_scorecard(scorecard_run)
        kpis = build_kpis(scorecard_records)
        git_info = scorecard_run.get("git", {})
        run_meta = {
            "mode": scorecard_run.get("mode", "unknown"),
            "timestamp": scorecard_run.get("timestamp", ""),
            "branch": git_info.get("branch", ""),
            "commit": git_info.get("commit", "")[:7],
            "is_pr_mode": scorecard_run.get("mode") == "pr"
            or any(
                "quality" in it or "accuracy" in it
                for it in scorecard_run.get("iterations", [])[:1]
            ),
        }
    else:
        scorecard_records = []
        kpis = {
            "avg_reduction_pct": 0.0,
            "quality_green": 0,
            "quality_total": 0,
            "speed_green": 0,
            "speed_total": 0,
            "accuracy_green": 0,
            "accuracy_total": 0,
        }
        run_meta = {"mode": "", "timestamp": "", "branch": "", "commit": "", "is_pr_mode": False}

    scorecard_json = json.dumps(
        {"records": scorecard_records, "kpis": kpis, "meta": run_meta}, separators=(",", ":")
    )

    # Load HTML template and embed scorecard JSON inline (avoids extra fetch for first paint).
    src_html = _template_dir() / "index.html"
    html = src_html.read_text(encoding="utf-8")
    html = html.replace("__SCORECARD_JSON__", scorecard_json)

    (out_dir / "index.html").write_text(html, encoding="utf-8")

    # Write the trend data file.
    json_path = out_dir / "data" / "history.json"
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    # Optionally render the quality-samples sub-page.
    if with_samples:
        try:
            render_samples_page(out_dir, scorecard_run, corpus_root=corpus_root)
        except Exception as exc:
            import traceback

            print(
                f"[dashboard] WARNING: samples page generation failed: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc(file=sys.stderr)


def _escape_html(s: str) -> str:
    """Minimal HTML-escape for values embedded in the samples page."""
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _status_icon(status: str) -> str:
    if status == "fail":
        return "❌"
    if status == "warn":
        return "⚠️"
    return "✅"


def _badge_class(status: str) -> str:
    if status == "ok":
        return "badge-ok"
    if status == "warn":
        return "badge-warn"
    if status == "fail":
        return "badge-fail"
    return "badge-ok"


def _render_sample_card(rec: dict[str, Any]) -> str:
    """Render one HTML <article> for a format record."""
    fmt = rec["format"].upper()
    status = rec.get("status", "ok")
    is_lossless = rec.get("lossless", False)
    no_data = rec.get("no_data", False)

    badge_cls = _badge_class(status)
    icon = _status_icon(status)
    card_cls = "sample-card"
    if is_lossless:
        card_cls += " lossless"
    else:
        card_cls += f" status-{status}"

    lines: list[str] = []
    lines.append(f'<article class="{card_cls}">')
    lines.append('  <div class="card-header">')
    lines.append(f"    <h2>{_escape_html(fmt)}</h2>")
    lines.append(
        f'    <span class="status-badge {badge_cls}">{icon} {_escape_html(status.upper())}</span>'
    )
    lines.append("  </div>")

    if is_lossless:
        lines.append(
            '  <p class="lossless-note">Lossless — output is pixel-identical to input.</p>'
        )
    elif no_data:
        lines.append(
            '  <p class="no-data-note">No pr-mode quality data available for this format in the current run.</p>'
        )
    else:
        ssim = rec.get("ssim")
        psnr = rec.get("psnr_db")
        case_id = rec.get("case_id") or ""
        preset = rec.get("preset") or ""
        size_orig = rec.get("size_orig_kb", 0)
        size_opt = rec.get("size_opt_kb", 0)
        reduction = rec.get("reduction_pct", 0.0)
        threshold = rec.get("ssim_threshold")

        # Metrics row
        ssim_str = f"{ssim:.4f}" if ssim is not None else "—"
        psnr_str = f"{psnr:.1f} dB" if psnr is not None else "—"
        thresh_str = f"threshold {threshold:.2f}" if threshold is not None else ""
        lines.append(
            f'  <p class="card-meta">'
            f"    <strong>SSIM</strong> {_escape_html(ssim_str)}"
            f"    &nbsp;/&nbsp; <strong>PSNR</strong> {_escape_html(psnr_str)}"
            + (f"    &nbsp;&middot;&nbsp; {_escape_html(thresh_str)}" if thresh_str else "")
            + "  </p>"
        )

        # Case + compression row
        if case_id:
            lines.append(
                f'  <p class="card-meta">Case: <code>{_escape_html(case_id)}</code>'
                f" &nbsp;&middot;&nbsp; preset <strong>{_escape_html(preset)}</strong></p>"
            )
        if size_orig > 0 and size_opt > 0:
            lines.append(
                f'  <p class="card-meta">Compressed by <strong>{reduction:.0f}%</strong>'
                f" ({size_orig:.1f} KB &rarr; {size_opt:.1f} KB)</p>"
            )

        # Thumbnails (if available)
        orig_b64 = rec.get("orig_thumb_b64")
        opt_b64 = rec.get("opt_thumb_b64")

        if orig_b64 and opt_b64:
            lines.append('  <div class="sample-pair">')
            lines.append("    <figure>")
            lines.append(
                f'      <img src="data:image/png;base64,{orig_b64}"'
                f' alt="Original {_escape_html(fmt)}" loading="lazy" />'
            )
            lines.append("      <figcaption>Original</figcaption>")
            lines.append("    </figure>")
            lines.append("    <figure>")
            lines.append(
                f'      <img src="data:image/png;base64,{opt_b64}"'
                f' alt="Optimized {_escape_html(fmt)}" loading="lazy" />'
            )
            lines.append(
                f"      <figcaption>Optimized (preset={_escape_html(preset)})</figcaption>"
            )
            lines.append("    </figure>")
            lines.append("  </div>")
        else:
            lines.append('  <div class="sample-pair">')
            lines.append("    <figure>")
            lines.append(
                '      <div class="no-thumb">Thumbnails not available — corpus not built</div>'
            )
            lines.append("    </figure>")
            lines.append("    <figure>")
            lines.append(
                '      <div class="no-thumb">Run: python -m bench.corpus build --manifest core</div>'
            )
            lines.append("    </figure>")
            lines.append("  </div>")

    lines.append("</article>")
    return "\n".join(lines)


def render_samples_page(
    out_dir: Path,
    scorecard_run: dict[str, Any] | None,
    corpus_root: Path | None = None,
) -> None:
    """Render ``quality-samples/index.html`` inside *out_dir*.

    Parameters
    ----------
    out_dir:
        The dashboard output directory (must already exist — main ``index.html``
        must be written first).
    scorecard_run:
        Parsed bench run JSON.  May be None (quick-mode or missing) — in that
        case a "no samples available" placeholder is written.
    corpus_root:
        Path to the corpus data directory for thumbnail generation.  If None,
        numeric-only cards are rendered.
    """
    samples_dir = out_dir / "quality-samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    src_template = _template_dir() / "quality-samples.html"
    template_html = src_template.read_text(encoding="utf-8")

    if scorecard_run is None or scorecard_run.get("mode") not in ("pr",):
        # Check heuristically for quality data even if mode isn't set to "pr"
        has_quality = any(
            isinstance(it.get("quality"), dict)
            for it in (scorecard_run or {}).get("iterations", [])[:5]
        )
        if scorecard_run is None or (scorecard_run.get("mode") != "pr" and not has_quality):
            body_html = (
                '<div class="notice">'
                "<p>No quality samples available yet.</p>"
                "<p>Samples are generated from a pr-mode run. Run:</p>"
                "<p><code>python -m bench.run --mode pr --manifest core "
                "--out /tmp/pr.json</code></p>"
                "<p>then rebuild the dashboard with "
                "<code>--baseline /tmp/pr.json</code></p>"
                "</div>"
            )
            meta_html = ""
            html = template_html.replace("__BODY_HTML__", body_html).replace(
                "__META_HTML__", meta_html
            )
            (samples_dir / "index.html").write_text(html, encoding="utf-8")
            return

    # Build per-format records.
    try:
        records = build_sample_records(scorecard_run, corpus_root=corpus_root)
    except Exception as exc:
        import traceback

        print(f"[samples] ERROR building sample records: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        records = []

    # Render cards.
    if not records:
        body_html = (
            '<div class="notice">No sample records generated — '
            "check that the baseline is a pr-mode run.</div>"
        )
    else:
        # Section: lossy formats
        lossy_recs = [r for r in records if not r.get("lossless")]
        lossless_recs = [r for r in records if r.get("lossless")]

        parts: list[str] = []
        if lossy_recs:
            parts.append('<p class="section-title">Lossy formats — worst-quality case</p>')
            parts.append('<div class="cards">')
            for rec in lossy_recs:
                parts.append(_render_sample_card(rec))
            parts.append("</div>")

        if lossless_recs:
            parts.append('<p class="section-title">Lossless formats</p>')
            parts.append('<div class="cards">')
            for rec in lossless_recs:
                parts.append(_render_sample_card(rec))
            parts.append("</div>")

        body_html = "\n".join(parts)

    # Meta line
    ts = scorecard_run.get("timestamp", "") if scorecard_run else ""
    ts_date = ts.split("T")[0] if ts else "unknown"
    git_info = (scorecard_run or {}).get("git", {})
    branch = git_info.get("branch", "?")
    commit = (git_info.get("commit", "") or "")[:7] or "?"
    meta_html = (
        f"Run date: {_escape_html(ts_date)} &middot; {_escape_html(branch)}@{_escape_html(commit)}"
    )

    html = template_html.replace("__BODY_HTML__", body_html).replace("__META_HTML__", meta_html)
    (samples_dir / "index.html").write_text(html, encoding="utf-8")


def _no_baseline_page(out_dir: Path) -> None:
    """Write a minimal placeholder when there is no baseline history yet."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "data").mkdir()

    placeholder = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": "core",
        "latest_sha": "none",
        "runs": [],
    }
    (out_dir / "data" / "history.json").write_text(
        json.dumps(placeholder, indent=2) + "\n", encoding="utf-8"
    )

    # Still copy the template so a valid page is served (with no scorecard data).
    src_html = _template_dir() / "index.html"
    if src_html.exists():
        html = src_html.read_text(encoding="utf-8")
        html = html.replace("__SCORECARD_JSON__", "null")
        (out_dir / "index.html").write_text(html, encoding="utf-8")
    else:
        (out_dir / "index.html").write_text(
            "<!doctype html><html><body>"
            "<h1>Pare bench dashboard</h1>"
            "<p>No baseline pinned yet. Run the bench workflow on main first.</p>"
            "</body></html>\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* until we find a ``.git`` directory."""
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a static HTML dashboard from bench git history.",
        prog="bench.dashboard.build",
    )
    parser.add_argument(
        "--out-dir",
        default="dashboard/dist",
        help="Output directory (wiped and recreated each run). Default: dashboard/dist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of historical baseline commits to include. Default: 100",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to the git repo root. Defaults to auto-detection from CWD.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Path to a bench run JSON to use as scorecard data source. "
            "Defaults to reports/baseline.core.json relative to the repo root."
        ),
    )
    parser.add_argument(
        "--with-samples",
        dest="with_samples",
        action="store_true",
        default=True,
        help=(
            "Generate the quality-samples/ sub-page (default: true). "
            "Requires a pr-mode baseline for thumbnails."
        ),
    )
    parser.add_argument(
        "--no-with-samples",
        dest="with_samples",
        action="store_false",
        help="Skip quality-samples/ sub-page generation (fast dev rebuilds).",
    )
    parser.add_argument(
        "--corpus-root",
        default=None,
        help=(
            "Path to the on-disk corpus directory for thumbnail generation. "
            "Defaults to bench/corpus/data relative to the repo root. "
            "Set to empty string to skip thumbnails."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve() if args.repo else _find_repo_root(Path.cwd())
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    print(f"[dashboard] repo root : {repo_root}", file=sys.stderr)
    print(f"[dashboard] output dir: {out_dir}", file=sys.stderr)

    # Resolve scorecard baseline path.
    if args.baseline:
        scorecard_path = Path(args.baseline).resolve()
    else:
        scorecard_path = repo_root / BASELINE_PATH
    print(f"[dashboard] scorecard : {scorecard_path}", file=sys.stderr)

    # Resolve corpus root for sample thumbnails.
    if args.corpus_root is not None:
        corpus_root: Path | None = Path(args.corpus_root).resolve() if args.corpus_root else None
    else:
        # Check tests/corpus first (current actual write location), then legacy path.
        _candidates = [
            repo_root / "tests" / "corpus",
            repo_root / "bench" / "corpus" / "data",
        ]
        corpus_root = next((p for p in _candidates if p.exists()), None)
    if corpus_root is not None:
        print(f"[dashboard] corpus    : {corpus_root}", file=sys.stderr)
    else:
        print(
            "[dashboard] corpus    : not found — thumbnails will be skipped",
            file=sys.stderr,
        )

    # Load scorecard data (graceful — None if missing or wrong schema).
    scorecard_run = load_scorecard_data(scorecard_path)
    if scorecard_run is None:
        print(
            "[dashboard] No scorecard data found — scorecard section will show placeholder.",
            file=sys.stderr,
        )
    else:
        mode = scorecard_run.get("mode", "?")
        n_iters = len(scorecard_run.get("iterations", []))
        print(
            f"[dashboard] Loaded scorecard: mode={mode}, {n_iters} iterations",
            file=sys.stderr,
        )

    history = build_history(repo_root, limit=args.limit)
    n_runs = len(history["runs"])

    if n_runs == 0:
        print(
            "[dashboard] No baseline history found — writing placeholder page.",
            file=sys.stderr,
        )
        _no_baseline_page(out_dir)
        return 0

    render_output(
        history,
        out_dir,
        scorecard_run=scorecard_run,
        with_samples=args.with_samples,
        corpus_root=corpus_root,
    )
    print(f"[dashboard] Wrote {n_runs} run(s) to {out_dir}", file=sys.stderr)
    if args.with_samples:
        samples_idx = out_dir / "quality-samples" / "index.html"
        if samples_idx.exists():
            size_kb = samples_idx.stat().st_size / 1024
            print(
                f"[dashboard] samples page: {samples_idx} ({size_kb:.0f} KB)",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
