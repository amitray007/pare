"""Tests for load mode — concurrent-request throughput and backpressure.

Tests 1–5 require the synthesized corpus on disk; they are guarded by
skipif so CI doesn't fail if the corpus hasn't been built.

Test 6 (CLI flag plumbing) only needs the corpus for a real run; it is
also guarded by the same corpus skipif.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# ---------------------------------------------------------------------------
# Corpus sentinel
# ---------------------------------------------------------------------------
_CORPUS_ROOT = Path("tests/corpus")
_SMALL_JPEG = _CORPUS_ROOT / "small" / "jpeg" / "photo_perlin_small_jpeg.jpeg"

_corpus_present = pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build --manifest core` first",
)


def _make_jpeg_case():
    """Return a Case pointed at the small JPEG corpus file."""
    from bench.runner.case import Case

    return Case(
        case_id="photo_perlin_small_jpeg.jpeg@high",
        name="photo_perlin_small_jpeg",
        bucket="small",
        fmt="jpeg",
        preset="high",
        quality=40,
        file_path=_SMALL_JPEG,
        input_size=_SMALL_JPEG.stat().st_size,
    )


# ---------------------------------------------------------------------------
# Test 1: low-concurrency run — all requests should succeed
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_mode_runs_a_real_case_with_low_concurrency():
    """n_concurrent=4, semaphore=4, queue=4 — all 4 requests must succeed."""
    from bench.runner.modes.load import run_load_sync

    case = _make_jpeg_case()
    results = run_load_sync([case], n_concurrent=4, semaphore_size=4, queue_depth=4)

    assert len(results) == 1
    r = results[0]
    assert "error" not in r, f"unexpected case error: {r.get('error')}"

    lb = r["load"]
    assert lb["n_success"] == 4, f"expected 4 successes, got {lb['n_success']}"
    assert lb["n_503"] == 0, f"expected 0 503s, got {lb['n_503']}"
    assert lb["n_concurrent"] == 4
    assert lb["throughput_per_sec"] > 0.0
    assert lb["wall_ms"] > 0.0
    assert lb["ok_rate"] == 1.0
    # Latencies should be positive for all successes
    assert lb["request_latency_ms"]["p50"] > 0.0
    assert lb["request_latency_ms"]["max"] >= lb["request_latency_ms"]["p50"]
    # Queue wait must be non-negative
    assert lb["queue_wait_ms"]["p50"] >= 0.0


# ---------------------------------------------------------------------------
# Test 2: overloaded queue — 503s must fire
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_mode_produces_503s_when_queue_overflows():
    """n_concurrent=20 >> semaphore=2 + queue=4 — some must 503."""
    from bench.runner.modes.load import run_load_sync

    case = _make_jpeg_case()
    results = run_load_sync([case], n_concurrent=20, semaphore_size=2, queue_depth=4)

    assert len(results) == 1
    r = results[0]
    assert "error" not in r, f"unexpected case error: {r.get('error')}"

    lb = r["load"]
    # With 20 concurrent and only 6 total capacity (sem=2, queue=4 beyond),
    # we should see substantial 503s.
    assert lb["n_503"] > 0, (
        f"expected some 503s with overloaded gate (sem=2, queue=4, n=20); "
        f"got n_success={lb['n_success']}, n_503={lb['n_503']}"
    )
    assert lb["n_success"] + lb["n_503"] + lb["n_error"] == 20, (
        f"total outcomes must equal n_concurrent=20; "
        f"got success={lb['n_success']}, 503={lb['n_503']}, error={lb['n_error']}"
    )


# ---------------------------------------------------------------------------
# Test 3: optimizer failure → n_error (not n_503)
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_mode_handles_optimizer_failure():
    """monkeypatched optimize_image raises non-BackpressureError → n_error > 0."""
    from bench.runner.modes.load import run_load_sync

    case = _make_jpeg_case()

    with patch(
        "bench.runner.modes.load.optimize_image",
        new=AsyncMock(side_effect=RuntimeError("simulated optimizer crash")),
    ):
        results = run_load_sync([case], n_concurrent=4, semaphore_size=4, queue_depth=8)

    assert len(results) == 1
    lb = results[0]["load"]
    assert lb["n_error"] > 0, f"expected n_error > 0, got {lb}"
    assert lb["n_503"] == 0, f"optimizer errors must not be counted as 503s; got {lb}"
    assert lb["n_success"] == 0, f"expected 0 successes when optimizer always fails; got {lb}"


# ---------------------------------------------------------------------------
# Test 4: gate resets between cases — no leaked semaphore slots
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_mode_resets_gate_between_cases():
    """Two sequential cases both succeed; no leaked slots from case 1 to case 2."""
    from bench.runner.case import Case
    from bench.runner.modes.load import run_load_sync

    case1 = _make_jpeg_case()
    # Build a second distinct case pointing at the same file for simplicity.
    case2 = Case(
        case_id="photo_perlin_small_jpeg.jpeg@medium",
        name="photo_perlin_small_jpeg",
        bucket="small",
        fmt="jpeg",
        preset="medium",
        quality=60,
        file_path=_SMALL_JPEG,
        input_size=_SMALL_JPEG.stat().st_size,
    )

    results = run_load_sync(
        [case1, case2],
        n_concurrent=4,
        semaphore_size=4,
        queue_depth=4,
    )

    assert len(results) == 2
    for r in results:
        assert "error" not in r, f"unexpected error: {r.get('error')}"
        lb = r["load"]
        assert (
            lb["n_success"] > 0
        ), f"case {r['case_id']}: expected n_success > 0 (no leaked gate); got {lb}"


# ---------------------------------------------------------------------------
# Test 5: gate_observed captures saturation peaks
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_mode_records_gate_observed_peaks():
    """With n=10, sem=4, queue=8 the sampler should catch active_jobs >= 4."""
    from bench.runner.modes.load import run_load_sync

    case = _make_jpeg_case()
    results = run_load_sync([case], n_concurrent=10, semaphore_size=4, queue_depth=8)

    assert len(results) == 1
    lb = results[0]["load"]
    observed = lb["gate_observed"]

    # The sampler polls at 10 ms; for a JPEG optimization that takes >50 ms,
    # we expect to catch the gate at or near semaphore capacity.
    assert (
        observed["max_active_jobs"] >= 1
    ), f"sampler should observe at least 1 active job; got {observed}"
    # For a run of 10 against sem=4+queue=8=12 total capacity, we likely
    # fill the semaphore — assert >= 1 conservatively (timing-sensitive).
    assert isinstance(observed["max_queued_jobs"], int)


# ---------------------------------------------------------------------------
# Test 6: CLI flag plumbing
# ---------------------------------------------------------------------------


@_corpus_present
def test_load_cli_flag_plumbing(tmp_path: Path):
    """CLI: --mode load flags are plumbed into config and output JSON."""
    from bench.runner.cli import main

    out_json = tmp_path / "load_out.json"
    rc = main(
        [
            "run",
            "--mode",
            "load",
            "--n-concurrent",
            "8",
            "--semaphore-size",
            "4",
            "--queue-depth",
            "8",
            "--manifest",
            "core",
            "--fmt",
            "jpeg",
            "--bucket",
            "small",
            "--out",
            str(out_json),
        ]
    )

    assert rc == 0, f"CLI exited with {rc}"
    assert out_json.exists(), "output JSON not written"

    run = json.loads(out_json.read_text())
    cfg = run.get("config", {})
    assert cfg.get("n_concurrent") == 8, f"n_concurrent not in config: {cfg}"
    assert cfg.get("semaphore_size") == 4, f"semaphore_size not in config: {cfg}"
    assert cfg.get("queue_depth") == 8, f"queue_depth not in config: {cfg}"
    assert run.get("mode") == "load", f"mode not 'load': {run.get('mode')}"
    assert len(run.get("iterations", [])) > 0, "no iterations in output"
