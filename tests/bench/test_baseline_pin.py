"""Sanity guards for the pinned baseline at reports/baseline.core.json.

These tests ensure that whoever commits the baseline hasn't accidentally
shipped a truncated, schema-broken, or otherwise malformed file — and that
the compare engine reports zero regressions when a run is diffed against
itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

# Canonical location of the pinned baseline relative to the repo root.
# tests/ is two levels below the project root, so we go up two dirs.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BASELINE_PATH = _REPO_ROOT / "reports" / "baseline.core.json"
BASELINE_UPDATE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "bench-baseline-update.yml"


# ---------------------------------------------------------------------------
# 1. Baseline exists, parses cleanly, and is a quick-mode run with ≥100 rows
# ---------------------------------------------------------------------------


def test_baseline_file_exists_and_parses():
    """reports/baseline.core.json must exist, load as schema_version=2,
    be a 'quick' run, and contain at least 100 iteration rows.
    """
    assert BASELINE_PATH.exists(), (
        f"Pinned baseline not found at {BASELINE_PATH}. "
        "Run: python -m bench.run --mode quick --manifest core "
        "--annotate env=local-venv-bootstrap --out reports/baseline.core.json"
    )

    from bench.runner.report.json_writer import load_run

    run = load_run(BASELINE_PATH)

    assert run["mode"] == "quick", f"Expected mode='quick', got {run['mode']!r}"
    assert len(run["iterations"]) >= 100, (
        f"Expected ≥100 iteration rows, got {len(run['iterations'])}. "
        "Was the baseline generated with --manifest core?"
    )


# ---------------------------------------------------------------------------
# 2. Self-compare exits 0 with no regressions
# ---------------------------------------------------------------------------


def test_baseline_compared_to_itself_has_no_regressions():
    """Diffing the baseline against itself must produce zero regressions
    and exit code 0, exercising the compare engine on a known-good pair.
    """
    pytest.importorskip("bench.runner.compare")  # skip if bench not importable

    assert (
        BASELINE_PATH.exists()
    ), f"Pinned baseline not found at {BASELINE_PATH}; skipping compare test."

    from bench.runner.compare import compare

    result = compare(BASELINE_PATH, BASELINE_PATH, threshold_pct=10.0)

    assert (
        result.regressions == []
    ), f"Self-compare should have 0 regressions; found {len(result.regressions)}: " + ", ".join(
        d.case_id for d in result.regressions
    )
    assert result.exit_code == 0, f"Self-compare should exit 0; got {result.exit_code}"


# ---------------------------------------------------------------------------
# 3. Baseline renders as non-empty markdown with the per-case table
# ---------------------------------------------------------------------------


def test_baseline_renders_as_markdown():
    """render_run() must produce non-empty markdown containing the
    '## Per-case results' section header that PR comment consumers rely on.
    """
    assert (
        BASELINE_PATH.exists()
    ), f"Pinned baseline not found at {BASELINE_PATH}; skipping render test."

    from bench.runner.report.json_writer import load_run
    from bench.runner.report.markdown import render_run

    run = load_run(BASELINE_PATH)
    md = render_run(run)

    assert md, "render_run() returned an empty string"
    assert "## Per-case results" in md, (
        "'## Per-case results' header not found in markdown output. "
        "The baseline may have zero successful iterations."
    )


# ---------------------------------------------------------------------------
# 4. bench-baseline-update workflow YAML is valid and has required keys
# ---------------------------------------------------------------------------


def test_baseline_update_workflow_yaml_valid():
    """bench-baseline-update.yml must parse as valid YAML and contain the
    required structural keys: permissions (contents + issues), concurrency
    with cancel-in-progress=false, and the [skip ci] guard on the job's if:.
    """
    assert BASELINE_UPDATE_WORKFLOW.exists(), (
        f"Workflow file not found at {BASELINE_UPDATE_WORKFLOW}. "
        "Expected .github/workflows/bench-baseline-update.yml to exist."
    )

    wf = yaml.safe_load(BASELINE_UPDATE_WORKFLOW.read_text())

    # Top-level permissions
    perms = wf.get("permissions", {})
    assert (
        perms.get("contents") == "write"
    ), "permissions.contents must be 'write' so the bot commit can push to main"
    assert (
        perms.get("issues") == "write"
    ), "permissions.issues must be 'write' so the drift issue can be opened"

    # Concurrency — must NOT cancel so back-to-back merges serialize correctly
    concurrency = wf.get("concurrency", {})
    assert concurrency.get("cancel-in-progress") is False, (
        "concurrency.cancel-in-progress must be false so serialized merges "
        "each get a baseline check rather than the later one cancelling the earlier"
    )

    # Job-level [skip ci] guard
    jobs = wf.get("jobs", {})
    assert jobs, "No jobs defined in the workflow"
    job_name = next(iter(jobs))
    job = jobs[job_name]
    job_if = job.get("if", "")
    assert "[skip ci]" in str(job_if), (
        "The job 'if:' condition must guard against [skip ci] commits "
        "to prevent the bot's own refresh commit from re-triggering the workflow"
    )
    assert "[skip bench-baseline]" in str(job_if), (
        "The job 'if:' condition must guard against [skip bench-baseline] commits "
        "to allow opting out of baseline refresh on individual pushes"
    )
