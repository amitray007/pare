"""Benchmark dashboard server.

Standalone FastAPI app on port 8081 for running focused benchmarks
and viewing results in a mission-control style dashboard.

Usage:
    python -m benchmarks.server
    # or
    uvicorn benchmarks.server:app --port 8081 --reload
"""

import asyncio
import json
import os
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from benchmarks.cases import BenchmarkCase
from benchmarks.constants import CORPUS_GROUPS, PRESETS_BY_NAME
from benchmarks.corpus import load_corpus_cases as _load_corpus
from benchmarks.corpus import scan_corpus_by_group
from benchmarks.runner import run_single

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "tests" / "corpus"
DATA_DIR = ROOT / ".benchmark-data"
RUNS_DIR = DATA_DIR / "runs"
DASHBOARD_HTML = Path(__file__).resolve().parent / "templates/dashboard.html"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Pare Benchmark Dashboard", version="1.0.0")

# In-memory tracking of active runs
_active_runs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class RunConfig(BaseModel):
    formats: list[str] = []
    presets: list[str] = ["HIGH", "MEDIUM", "LOW"]
    groups: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)


def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(ROOT),
        ).strip()
    except Exception:
        return "unknown"


_corpus_cache: dict | None = None


def _scan_corpus() -> dict[str, dict[str, list[Path]]]:
    """Scan corpus by group. Returns: {group: {format: [paths]}}"""
    global _corpus_cache
    if _corpus_cache is not None:
        return _corpus_cache
    _corpus_cache = scan_corpus_by_group(CORPUS_DIR)
    return _corpus_cache


def _get_available_formats() -> dict[str, int]:
    """Get available formats with file counts."""
    corpus = _scan_corpus()
    fmt_counts: dict[str, int] = {}
    for group_data in corpus.values():
        for fmt, files in group_data.items():
            fmt_counts[fmt] = fmt_counts.get(fmt, 0) + len(files)
    return fmt_counts


def _select_cases(
    formats: list[str],
    groups: list[str],
) -> list[BenchmarkCase]:
    """Select cases from corpus filtered by groups and formats."""
    return _load_corpus(
        CORPUS_DIR,
        groups=groups if groups else None,
        formats=formats if formats else None,
    )


def _compute_health(results_by_fmt: dict) -> dict[str, str]:
    """Compute pass/warn/fail for each format."""
    health = {}
    for fmt, preset_results in results_by_fmt.items():
        # Check preset differentiation
        avg_by_preset = {}
        for preset_name, results in preset_results.items():
            valid = [r for r in results if not r.get("opt_error")]
            if valid:
                avg_by_preset[preset_name] = sum(r["reduction_pct"] for r in valid) / len(valid)

        differentiates = True
        if "HIGH" in avg_by_preset and "MEDIUM" in avg_by_preset:
            if avg_by_preset["HIGH"] <= avg_by_preset["MEDIUM"]:
                differentiates = False
        if "MEDIUM" in avg_by_preset and "LOW" in avg_by_preset:
            if avg_by_preset["MEDIUM"] <= avg_by_preset["LOW"]:
                differentiates = False

        # Check estimation accuracy
        all_results = []
        for results in preset_results.values():
            all_results.extend(results)
        valid_est = [r for r in all_results if not r.get("opt_error") and not r.get("est_error")]
        avg_est_err = 0
        if valid_est:
            avg_est_err = sum(r["est_error_pct"] for r in valid_est) / len(valid_est)

        if differentiates and avg_est_err < 10:
            health[fmt] = "pass"
        elif differentiates or avg_est_err < 15:
            health[fmt] = "warn"
        else:
            health[fmt] = "fail"

    return health


def _save_run(run_data: dict):
    """Save a run to disk."""
    _ensure_dirs()
    run_id = run_data["id"]
    path = RUNS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")


def _load_runs() -> list[dict]:
    """Load all saved runs, sorted newest first."""
    _ensure_dirs()
    runs = []
    for path in sorted(RUNS_DIR.glob("run-*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            runs.append(data)
        except Exception:
            continue
    return runs


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML."""
    if not DASHBOARD_HTML.exists():
        raise HTTPException(status_code=500, detail="Dashboard HTML not found")
    return HTMLResponse(DASHBOARD_HTML.read_text(encoding="utf-8"))


@app.get("/api/corpus")
async def get_corpus():
    """List available groups, formats, and counts."""
    corpus = _scan_corpus()
    groups_info = {}
    fmt_totals: dict[str, int] = {}

    for group_key, fmt_data in corpus.items():
        group_files = {}
        for fmt, files in fmt_data.items():
            group_files[fmt] = len(files)
            fmt_totals[fmt] = fmt_totals.get(fmt, 0) + len(files)
        groups_info[group_key] = {
            "total": sum(len(f) for f in fmt_data.values()),
            "formats": group_files,
        }

    # Add group labels from constants
    for key, info in groups_info.items():
        if key in CORPUS_GROUPS:
            g = CORPUS_GROUPS[key]
            info["label"] = g.label
            info["badge"] = g.badge
            info["description"] = g.description

    return {
        "groups": groups_info,
        "formats": {fmt: {"total": count} for fmt, count in fmt_totals.items()},
        "corpus_dir": str(CORPUS_DIR),
    }


@app.post("/api/run")
async def start_run(config: RunConfig):
    """Start a benchmark run."""
    corpus = _scan_corpus()
    if not corpus:
        raise HTTPException(status_code=400, detail="No corpus found. Download it first.")

    fmt_counts = _get_available_formats()
    available_formats = list(fmt_counts.keys())
    formats = config.formats if config.formats else available_formats

    # Validate presets
    preset_names = [p.upper() for p in config.presets]
    for p in preset_names:
        if p not in PRESETS_BY_NAME:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {p}")

    run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    cases = _select_cases(formats, config.groups)
    if not cases:
        raise HTTPException(status_code=400, detail="No cases selected.")

    _active_runs[run_id] = {
        "config": config.model_dump(),
        "cases": cases,
        "preset_names": preset_names,
        "started": True,
    }

    return {
        "run_id": run_id,
        "cases_count": len(cases),
        "formats": formats,
        "presets": preset_names,
        "groups": config.groups or list(corpus.keys()),
        "total_tasks": len(cases) * len(preset_names),
    }


@app.get("/api/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE stream of benchmark results."""
    if run_id not in _active_runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run_info = _active_runs[run_id]
    cases = run_info["cases"]
    preset_names = run_info["preset_names"]
    config_data = run_info["config"]

    async def event_generator():
        t_start = time.perf_counter()
        all_results: dict[str, dict[str, list[dict]]] = {}
        done = 0
        total = len(cases) * len(preset_names)

        # Send initial event
        yield f"data: {json.dumps({'type': 'start', 'total': total, 'run_id': run_id})}\n\n"

        sem = asyncio.Semaphore(min(os.cpu_count() or 4, 8))

        async def run_case(case, preset_name):
            nonlocal done
            preset = PRESETS_BY_NAME[preset_name]
            async with sem:
                result = await run_single(case, preset.config, preset_name)
            done += 1
            current_done = done

            result_data = {
                "name": case.name,
                "format": case.fmt,
                "category": case.category,
                "content": case.content,
                "group": case.group,
                "preset": preset_name,
                "original_size": len(case.data),
                "optimized_size": result.optimized_size,
                "reduction_pct": round(result.reduction_pct, 2),
                "method": result.method,
                "opt_time_ms": round(result.opt_time_ms, 1),
                "opt_error": result.opt_error or None,
                "est_reduction_pct": round(result.est_reduction_pct, 2),
                "est_potential": result.est_potential,
                "est_confidence": result.est_confidence,
                "est_time_ms": round(result.est_time_ms, 1),
                "est_error": result.est_error or None,
                "est_error_pct": round(result.est_error_pct, 1),
            }

            # Track in memory
            fmt = case.fmt
            all_results.setdefault(fmt, {}).setdefault(preset_name, []).append(result_data)

            return result_data, current_done

        # Run all cases concurrently, yield as they complete
        tasks = []
        for preset_name in preset_names:
            for case in cases:
                tasks.append(asyncio.create_task(run_case(case, preset_name)))

        for coro in asyncio.as_completed(tasks):
            result_data, current_done = await coro
            progress = current_done / total * 100
            yield f"data: {json.dumps({'type': 'result', 'progress': round(progress, 1), 'done': current_done, 'total': total, 'result': result_data})}\n\n"

        # Compute health and save
        duration = time.perf_counter() - t_start
        health = _compute_health(all_results)

        # Flatten results for storage
        flat_results = []
        for fmt_results in all_results.values():
            for preset_results in fmt_results.values():
                flat_results.extend(preset_results)

        run_data = {
            "id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit_hash(),
            "config": config_data,
            "duration_s": round(duration, 1),
            "results": flat_results,
            "health": health,
            "formats_tested": list(all_results.keys()),
        }

        _save_run(run_data)

        yield f"data: {json.dumps({'type': 'complete', 'run_id': run_id, 'duration_s': round(duration, 1), 'health': health})}\n\n"

        # Cleanup
        _active_runs.pop(run_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs")
async def list_runs():
    """List past runs."""
    runs = _load_runs()
    # Return summaries (not full results)
    summaries = []
    for r in runs:
        summaries.append(
            {
                "id": r["id"],
                "timestamp": r.get("timestamp"),
                "git_commit": r.get("git_commit"),
                "config": r.get("config"),
                "duration_s": r.get("duration_s"),
                "health": r.get("health", {}),
                "formats_tested": r.get("formats_tested", []),
                "result_count": len(r.get("results", [])),
            }
        )
    return {"runs": summaries}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get full results of a past run."""
    _ensure_dirs()
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Delete a past run."""
    _ensure_dirs()
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Run not found")
    path.unlink()
    return {"deleted": run_id}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print("\n  Pare Benchmark Dashboard")
    print("  http://localhost:8081\n")
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")
