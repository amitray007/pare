"""JSON report schema v2 — write + load.

The schema is the canonical machine format. Markdown is a derived
view; `bench compare` reads JSON only.

Top-level structure:

    {
      "schema_version": 2,
      "timestamp": "...",                 # UTC ISO-8601
      "mode": "quick|timing|memory",
      "config": { ... },                  # warmup/repeat/seed/shuffle/etc.
      "git": { "commit", "branch", "dirty" },
      "host": { "platform", "cpu_count", "rss_unit" },
      "library_versions": { "Pillow": "10.4.0", ... },
      "manifest": { "name", "sha256" },
      "annotations": { "key": "val", ... },     # from --annotate KEY=VAL
      "iterations": [ ... ],                    # one entry per iteration
      "stats": [ ... ]                          # rolled up via CaseStats
    }
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform as platform_mod
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bench.corpus.manifest import collect_library_versions
from bench.runner.stats import CaseStats, summarize_iterations

SCHEMA_VERSION = 2


def _available_cpu_count() -> int:
    """Return the number of CPUs the current process can actually use.

    Prefers `os.sched_getaffinity` on Linux because it respects cpuset/cgroup
    constraints (e.g. `docker run --cpuset-cpus=0-7`). Falls back to
    `os.cpu_count()` on platforms without sched_getaffinity (macOS, Windows).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


@dataclass
class GitInfo:
    commit: str = ""
    branch: str = ""
    dirty: bool = False


@dataclass
class HostInfo:
    """Environment fingerprint for a run.

    `platform` and `cpu_count` gate comparability in bench.compare. The rest are
    diagnostic: when a format drifts without any code change, these are what
    identify the cause. The May 2026 baseline recorded neither the Python version
    nor the OS release, so a three-month SVG drift of +6.6% -> +10.8% could not be
    attributed to anything (scour is pure Python and untouched by the optimizers).
    """

    platform: str
    cpu_count: int
    rss_unit: str = "kb"
    python_version: str = field(default_factory=lambda: platform_mod.python_version())
    python_implementation: str = field(default_factory=lambda: platform_mod.python_implementation())
    machine: str = field(default_factory=lambda: platform_mod.machine())
    platform_release: str = field(default_factory=lambda: platform_mod.release())


@dataclass
class RunMetadata:
    mode: str
    config: dict[str, Any]
    annotations: dict[str, str] = field(default_factory=dict)
    manifest_name: str = ""
    manifest_sha256: str = ""
    git: GitInfo = field(default_factory=GitInfo)
    host: HostInfo = field(
        default_factory=lambda: HostInfo(
            platform=platform_mod.system().lower(),
            cpu_count=_available_cpu_count(),
        )
    )
    library_versions: dict[str, str] = field(default_factory=collect_library_versions)
    timestamp: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


def detect_git_info(cwd: Path | None = None) -> GitInfo:
    """Best-effort `git` snapshot. Empty fields if not in a repo."""

    def _run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except (FileNotFoundError, subprocess.SubprocessError):
            return ""

    commit = _run("rev-parse", "HEAD")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    dirty_check = _run("status", "--porcelain")
    return GitInfo(commit=commit, branch=branch, dirty=bool(dirty_check))


def manifest_sha256(manifest_path: Path) -> str:
    h = hashlib.sha256()
    h.update(manifest_path.read_bytes())
    return h.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def write_run(
    metadata: RunMetadata,
    iterations: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Roll iterations into stats and persist the run as JSON."""
    stats = _roll_up_stats(iterations)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": metadata.timestamp,
        "mode": metadata.mode,
        "config": metadata.config,
        "git": asdict(metadata.git),
        "host": asdict(metadata.host),
        "library_versions": metadata.library_versions,
        "manifest": {
            "name": metadata.manifest_name,
            "sha256": metadata.manifest_sha256,
        },
        "annotations": metadata.annotations,
        "iterations": iterations,
        "stats": [asdict(s) for s in stats],
    }
    _atomic_write_json(out_path, payload)


def load_run(path: Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version={raw.get('schema_version')!r} "
            f"in {path} (expected {SCHEMA_VERSION})"
        )
    return raw


def _is_failure(it: dict[str, Any]) -> bool:
    """Return True if this iteration represents a run-time failure.

    Quick/timing/memory modes store ``error`` as a plain string.
    Accuracy mode stores its prediction-vs-actual metrics under
    ``accuracy``, and reuses ``error`` (a ``{phase, message}`` dict) only
    when estimate or optimize raised.  Either shape under ``error`` is a
    failure; an absent ``error`` key is success.
    """
    err = it.get("error")
    if err is None:
        return False
    return isinstance(err, (str, dict))


def _roll_up_stats(iterations: list[dict[str, Any]]) -> list[CaseStats]:
    """Group iterations by case_id and produce one CaseStats per group."""
    by_case: dict[str, list[dict[str, Any]]] = {}
    case_meta: dict[str, dict[str, Any]] = {}
    for it in iterations:
        if _is_failure(it):
            # Errored iterations don't contribute to stats; surfaced
            # separately in the report.
            continue
        cid = it["case_id"]
        by_case.setdefault(cid, []).append(it["measurement"])
        case_meta.setdefault(cid, it)

    stats: list[CaseStats] = []
    for cid, ms in by_case.items():
        meta = case_meta[cid]
        # Map the measurement dicts to the keys summarize_iterations expects.
        iter_dicts = [
            {
                "wall_ms": m["wall_ms"],
                "children_cpu_ms": m["children_user_ms"] + m["children_sys_ms"],
                "children_peak_rss_kb": m["children_peak_rss_kb"],
                "parent_peak_rss_kb": m["parent_peak_rss_kb"],
                "parallelism": m["parallelism"],
                "py_peak_alloc_kb": m.get("py_peak_alloc_kb"),
                "rss_samples": m.get("rss_samples") or [],
            }
            for m in ms
        ]
        stats.append(
            summarize_iterations(
                cid,
                meta["bucket"],
                meta["format"],
                meta["preset"],
                iter_dicts,
                reduction_pct=meta.get("reduction_pct", 0.0),
                method=meta.get("method", ""),
            )
        )
    return stats
