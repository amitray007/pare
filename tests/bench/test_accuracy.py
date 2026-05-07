"""Tests for accuracy mode (estimate vs actual optimization).

These tests exercise the full pipeline: corpus file -> estimate() +
optimize_image() -> per-case error metrics.

The first test requires a real corpus file on disk. It is guarded by a
skipif so CI doesn't fail if the corpus hasn't been built yet.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bench.corpus.builder import build
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
from bench.runner.case import Case, load_cases
from bench.runner.modes.accuracy import run_accuracy, run_accuracy_sync
from schemas import EstimateResponse, OptimizeResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORPUS_ROOT = Path("tests/corpus")
_SMALL_JPEG = _CORPUS_ROOT / "small" / "jpeg" / "photo_perlin_small_jpeg.jpeg"


def _make_small_case(tmp_path: Path, fmt: str = "png") -> Case:
    """Build a tiny synthesized corpus and return one Case from it."""
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="acc_test",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=7,
                width=128,
                height=96,
                output_formats=[fmt],
            )
        ],
    )
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations
    cases = load_cases(m, tmp_path, preset_filter={"high"})
    assert cases
    return cases[0]


# ---------------------------------------------------------------------------
# Real-corpus integration test
# ---------------------------------------------------------------------------

pytestmark_real = pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build --manifest core` first",
)


@pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build` first",
)
def test_accuracy_mode_runs_a_real_case():
    """End-to-end: load a real JPEG corpus file, run accuracy mode, check schema."""
    # Build a minimal Case pointing at the real file
    case = Case(
        case_id="photo_perlin_small_jpeg.jpeg@high",
        name="photo_perlin_small_jpeg",
        bucket="small",
        fmt="jpeg",
        preset="high",
        quality=40,
        file_path=_SMALL_JPEG,
        input_size=_SMALL_JPEG.stat().st_size,
    )

    results = run_accuracy_sync([case])
    assert len(results) == 1
    r = results[0]

    # Top-level required fields
    assert r["case_id"] == "photo_perlin_small_jpeg.jpeg@high"
    assert r["format"] == "jpeg"
    assert r["preset"] == "high"
    assert r["iteration"] == 0
    assert r["input_size"] > 0

    # Estimate block
    est = r["estimate"]
    assert est["wall_ms"] > 0
    assert est["predicted_size"] > 0
    assert isinstance(est["predicted_reduction_pct"], (int, float))
    assert est["method"]
    assert est["confidence"] in ("high", "medium", "low")
    assert isinstance(est["already_optimized"], bool)
    assert "measurement" in est

    # Optimize block
    opt = r["optimize"]
    assert opt["wall_ms"] > 0
    assert opt["actual_size"] > 0
    assert isinstance(opt["actual_reduction_pct"], (int, float))
    assert opt["method"]
    assert "measurement" in opt
    assert "tool_invocations" in opt

    # Accuracy metrics on success live under "accuracy"; "error" is reserved
    # for the failure path (estimate/optimize raised).
    assert "error" not in r, "success row must not have 'error' key"
    acc = r["accuracy"]
    assert "size_abs_error_bytes" in acc
    assert "size_rel_error_pct" in acc
    assert "reduction_abs_error_pct" in acc
    assert "reduction_abs_error_pct_abs" in acc
    # Absolute value must be non-negative
    assert acc["reduction_abs_error_pct_abs"] >= 0


# ---------------------------------------------------------------------------
# Failure-handling tests (use monkeypatching on small synthesized cases)
# ---------------------------------------------------------------------------


def test_accuracy_handles_estimate_failure(tmp_path: Path):
    """If estimate() raises, the row must have error.phase == 'estimate'."""
    case = _make_small_case(tmp_path)

    with patch(
        "bench.runner.modes.accuracy.estimate",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        results = asyncio.run(run_accuracy([case]))

    assert len(results) == 1
    r = results[0]
    assert "error" in r
    err = r["error"]
    assert err["phase"] == "estimate"
    assert "boom" in err["message"]
    # estimate block must not appear (it never succeeded)
    assert "estimate" not in r
    # optimize block must not appear (we stopped early)
    assert "optimize" not in r


def test_accuracy_handles_optimize_failure(tmp_path: Path):
    """If optimize_image() raises after estimate succeeds, the row must
    have estimate block AND error.phase == 'optimize'."""
    case = _make_small_case(tmp_path)

    # Build a valid EstimateResponse the mock will return
    fake_est = EstimateResponse(
        original_size=case.input_size,
        original_format="png",
        dimensions={"width": 128, "height": 96},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=case.input_size // 2,
        estimated_reduction_percent=50.0,
        optimization_potential="high",
        method="exact",
        already_optimized=False,
        confidence="high",
    )

    with patch(
        "bench.runner.modes.accuracy.estimate",
        new=AsyncMock(return_value=fake_est),
    ):
        with patch(
            "bench.runner.modes.accuracy.optimize_image",
            new=AsyncMock(side_effect=RuntimeError("optimizer exploded")),
        ):
            results = asyncio.run(run_accuracy([case]))

    assert len(results) == 1
    r = results[0]

    # estimate block must be present (it succeeded)
    assert "estimate" in r

    # error must flag optimize phase
    err = r["error"]
    assert err["phase"] == "optimize"
    assert "optimizer exploded" in err["message"]

    # optimize block must not appear
    assert "optimize" not in r


def test_accuracy_error_signs(tmp_path: Path):
    """Verify sign conventions for error metrics.

    Case A: predicted_size > actual_size  =>  size_abs_error_bytes > 0,
                                              size_rel_error_pct > 0
    Case B: predicted_reduction < actual_reduction  =>  reduction_abs_error_pct < 0
    """
    case = _make_small_case(tmp_path)
    input_size = case.input_size

    # Predicted size is larger than actual (estimator over-estimated final size)
    predicted_size = input_size // 2
    actual_size = predicted_size - 1000  # smaller than prediction

    fake_est = EstimateResponse(
        original_size=input_size,
        original_format="png",
        dimensions={"width": 128, "height": 96},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=predicted_size,
        estimated_reduction_percent=50.0,
        optimization_potential="high",
        method="exact",
        already_optimized=False,
        confidence="high",
    )
    fake_opt = OptimizeResult(
        success=True,
        original_size=input_size,
        optimized_size=actual_size,
        reduction_percent=round((input_size - actual_size) / input_size * 100, 1),
        format="png",
        method="oxipng",
        optimized_bytes=b"x" * actual_size,
    )

    with patch(
        "bench.runner.modes.accuracy.estimate",
        new=AsyncMock(return_value=fake_est),
    ):
        with patch(
            "bench.runner.modes.accuracy.optimize_image",
            new=AsyncMock(return_value=fake_opt),
        ):
            results = asyncio.run(run_accuracy([case]))

    r = results[0]
    acc = r["accuracy"]

    # Predicted size > actual => positive size error
    assert acc["size_abs_error_bytes"] > 0
    assert acc["size_rel_error_pct"] > 0

    # Predicted reduction (50%) < actual reduction (higher because actual_size is smaller)
    # => reduction_abs_error_pct should be negative
    actual_reduction = fake_opt.reduction_percent
    assert actual_reduction > 50.0  # verify our test setup is correct
    assert acc["reduction_abs_error_pct"] < 0, (
        f"expected reduction_abs_error_pct < 0 when "
        f"predicted_reduction=50.0% < actual_reduction={actual_reduction:.1f}%"
    )
    assert acc["reduction_abs_error_pct_abs"] > 0
