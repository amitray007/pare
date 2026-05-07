"""Tests for PR mode — timing + accuracy + quality in a single pass.

Test structure mirrors the existing accuracy/quality test pattern:
  - Unit tests with mocked optimizers (no corpus dependency)
  - Integration tests guarded by skipif when real corpus is absent

The tests verify:
  1. All three blocks (measurement, accuracy, quality) are present per case.
  2. Lossless formats produce quality=None and don't crash.
  3. Lossy formats produce SSIM/PSNR in the quality block.
  4. Accuracy errors are computed correctly from known predicted vs actual.
  5. The markdown report contains all three sections.
  6. Timing iterations are collected correctly (N results per case).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from bench.corpus.builder import build
from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
from bench.runner.case import Case, load_cases
from bench.runner.modes.pr import run_pr, run_pr_sync
from schemas import EstimateResponse, OptimizeResult

# ---------------------------------------------------------------------------
# Corpus sentinel
# ---------------------------------------------------------------------------
_CORPUS_ROOT = Path("tests/corpus")
_SMALL_JPEG = _CORPUS_ROOT / "small" / "jpeg" / "photo_perlin_small_jpeg.jpeg"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path, fmt: str = "png") -> Case:
    """Build a tiny synthesized corpus and return one Case from it."""
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="pr_test",
                bucket=Bucket.SMALL,
                content_kind="photo_noise",
                seed=42,
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


def _fake_estimate(case: Case, estimated_size: int = 0) -> EstimateResponse:
    if estimated_size == 0:
        estimated_size = case.input_size // 2
    return EstimateResponse(
        original_size=case.input_size,
        original_format=case.fmt,
        dimensions={"width": 128, "height": 96},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=round(
            (case.input_size - estimated_size) / case.input_size * 100, 1
        ),
        optimization_potential="high",
        method="direct_encode",
        already_optimized=False,
        confidence="high",
    )


def _fake_optimize(case: Case, actual_size: int = 0) -> OptimizeResult:
    if actual_size == 0:
        actual_size = case.input_size // 3
    # Return a tiny valid JPEG as the optimized bytes so quality scoring
    # can attempt to decode it (it will gracefully handle decode failures).
    optimized_bytes = case.file_path.read_bytes()
    return OptimizeResult(
        success=True,
        original_size=case.input_size,
        optimized_size=actual_size,
        reduction_percent=round((case.input_size - actual_size) / case.input_size * 100, 1),
        format=case.fmt,
        method="pngquant+oxipng",
        optimized_bytes=optimized_bytes,
    )


# ---------------------------------------------------------------------------
# Test 1: PR mode returns repeat results per case
# ---------------------------------------------------------------------------


def test_pr_mode_returns_repeat_results_per_case(tmp_path: Path):
    """With repeat=2, pr mode should return 2 iteration dicts per case."""
    case = _make_case(tmp_path)

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        results = run_pr_sync([case], warmup=0, repeat=2)

    assert len(results) == 2, f"expected 2 (repeat=2), got {len(results)}"
    assert results[0]["iteration"] == 0
    assert results[1]["iteration"] == 1


# ---------------------------------------------------------------------------
# Test 2: All three blocks present in success rows
# ---------------------------------------------------------------------------


def test_pr_mode_all_three_blocks_present(tmp_path: Path):
    """Every successful result row must have measurement, accuracy, and quality blocks."""
    case = _make_case(tmp_path)

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        results = run_pr_sync([case], warmup=0, repeat=1)

    assert len(results) == 1
    r = results[0]

    assert "error" not in r, f"unexpected error: {r.get('error')}"
    assert "measurement" in r, "measurement block missing"
    assert "accuracy" in r, "accuracy block missing"
    assert "estimate" in r, "estimate sub-block missing"
    assert "optimize" in r, "optimize sub-block missing"

    acc = r["accuracy"]
    assert "size_abs_error_bytes" in acc
    assert "size_rel_error_pct" in acc
    assert "reduction_abs_error_pct" in acc
    assert "reduction_abs_error_pct_abs" in acc
    assert acc["reduction_abs_error_pct_abs"] >= 0


# ---------------------------------------------------------------------------
# Test 3: Lossless format produces quality=None
# ---------------------------------------------------------------------------


def test_pr_mode_lossless_format_quality_is_none(tmp_path: Path):
    """PNG (lossless) cases must have quality=None in every result row."""
    case = _make_case(tmp_path, fmt="png")
    assert case.fmt == "png"

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        results = run_pr_sync([case], warmup=0, repeat=1)

    assert len(results) == 1
    r = results[0]
    assert "error" not in r, f"unexpected error: {r.get('error')}"
    assert r["quality"] is None, f"expected quality=None for lossless PNG, got {r['quality']!r}"


# ---------------------------------------------------------------------------
# Test 4: Accuracy errors computed correctly from known values
# ---------------------------------------------------------------------------


def test_pr_mode_accuracy_errors_known_values(tmp_path: Path):
    """Error metrics must match the analytic formula for known inputs.

    predicted_size = input // 2    (50% reduction)
    actual_size    = input // 4    (75% actual reduction)

    size_abs_error_bytes = predicted - actual = input // 2 - input // 4 = input // 4
    size_rel_error_pct   = (predicted - actual) / actual * 100
                         = (input//4) / (input//4) * 100 = 100.0
    reduction_abs_error  = 50.0 - 75.0 = -25.0   (predicted was optimistic)
    """
    case = _make_case(tmp_path)
    input_size = case.input_size

    predicted_size = input_size // 2
    actual_size = input_size // 4

    fake_est = EstimateResponse(
        original_size=input_size,
        original_format="png",
        dimensions={"width": 128, "height": 96},
        color_type="rgb",
        bit_depth=8,
        estimated_optimized_size=predicted_size,
        estimated_reduction_percent=50.0,
        optimization_potential="high",
        method="direct_encode",
        already_optimized=False,
        confidence="high",
    )
    fake_opt = OptimizeResult(
        success=True,
        original_size=input_size,
        optimized_size=actual_size,
        reduction_percent=75.0,
        format="png",
        method="oxipng",
        optimized_bytes=case.file_path.read_bytes(),
    )

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=fake_est),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=fake_opt),
        ),
    ):
        results = run_pr_sync([case], warmup=0, repeat=1)

    r = results[0]
    acc = r["accuracy"]

    assert acc["size_abs_error_bytes"] == predicted_size - actual_size
    # size_rel_error_pct = (predicted-actual) / actual * 100 ≈ 100%
    expected_rel = round(100.0 * (predicted_size - actual_size) / actual_size, 3)
    assert abs(acc["size_rel_error_pct"] - expected_rel) < 0.01, acc

    # reduction_abs_error_pct = predicted_reduction - actual_reduction = 50 - 75 = -25
    assert abs(acc["reduction_abs_error_pct"] - (-25.0)) < 0.01, acc
    assert acc["reduction_abs_error_pct_abs"] > 0


# ---------------------------------------------------------------------------
# Test 5: Estimate failure yields error.phase == 'estimate'
# ---------------------------------------------------------------------------


def test_pr_mode_estimate_failure(tmp_path: Path):
    """If estimate() raises, result row must have error.phase == 'estimate'."""
    case = _make_case(tmp_path)

    with patch(
        "bench.runner.modes.pr.estimate",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        results = asyncio.run(run_pr([case], warmup=0, repeat=1))

    assert len(results) == 1
    r = results[0]
    assert "error" in r
    assert r["error"]["phase"] == "estimate"
    assert "boom" in r["error"]["message"]
    assert "accuracy" not in r
    assert "quality" not in r


# ---------------------------------------------------------------------------
# Test 6: Optimize failure in accuracy step
# ---------------------------------------------------------------------------


def test_pr_mode_optimize_failure_in_accuracy_step(tmp_path: Path):
    """If optimize_image() raises during the accuracy step, the error row
    must have error.phase == 'optimize' and preserve the estimate block."""
    case = _make_case(tmp_path)

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(side_effect=RuntimeError("optimizer exploded")),
        ),
    ):
        results = asyncio.run(run_pr([case], warmup=0, repeat=1))

    assert len(results) == 1
    r = results[0]
    assert "error" in r
    assert r["error"]["phase"] == "optimize"
    # estimate block must be present (it succeeded before the optimize attempt)
    assert "estimate" in r


# ---------------------------------------------------------------------------
# Test 7: Top-level shims for stats roll-up
# ---------------------------------------------------------------------------


def test_pr_mode_top_level_shims_for_rollup(tmp_path: Path):
    """reduction_pct, method, optimized_size must appear at top level for
    json_writer._roll_up_stats compatibility."""
    case = _make_case(tmp_path)

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        results = run_pr_sync([case], warmup=0, repeat=1)

    r = results[0]
    assert "reduction_pct" in r, "reduction_pct must be at top level for stats rollup"
    assert "method" in r, "method must be at top level"
    assert "optimized_size" in r, "optimized_size must be at top level"
    assert r["iteration"] == 0


# ---------------------------------------------------------------------------
# Test 8: Markdown report contains all three sections
# ---------------------------------------------------------------------------


def test_pr_mode_markdown_contains_all_three_sections(tmp_path: Path):
    """render_run() for a pr-mode run must include timing, quality, and accuracy sections."""
    from bench.runner.report.json_writer import RunMetadata, load_run, write_run
    from bench.runner.report.markdown import render_run

    case = _make_case(tmp_path)

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        iterations = run_pr_sync([case], warmup=0, repeat=1)

    out_path = tmp_path / "pr_run.json"
    metadata = RunMetadata(
        mode="pr",
        config={"warmup": 0, "repeat": 1, "quality_fast": True},
        manifest_name="test",
        manifest_sha256="abc",
    )
    write_run(metadata, iterations, out_path)

    run = load_run(out_path)
    assert run["mode"] == "pr"

    md = render_run(run)
    assert "## Per-format timing summary" in md, "timing section missing from PR markdown"
    assert "## Per-format quality summary" in md, "quality section missing from PR markdown"
    assert "## Per-format estimation accuracy" in md, "accuracy section missing from PR markdown"
    assert "<details>" in md, "per-case detail collapsible section missing"


# ---------------------------------------------------------------------------
# Test 9: SSIM threshold constants are defined and ordered correctly
# ---------------------------------------------------------------------------


def test_ssim_thresholds_ordered_correctly():
    """LOW preset must require the highest SSIM (most lossless-like)."""
    from bench.runner.report.markdown import PR_SSIM_THRESHOLD

    assert PR_SSIM_THRESHOLD["high"] < PR_SSIM_THRESHOLD["medium"] < PR_SSIM_THRESHOLD["low"]
    assert PR_SSIM_THRESHOLD["high"] == 0.95
    assert PR_SSIM_THRESHOLD["medium"] == 0.97
    assert PR_SSIM_THRESHOLD["low"] == 0.99


# ---------------------------------------------------------------------------
# Test 10: CLI --mode pr wires through correctly
# ---------------------------------------------------------------------------


def test_pr_mode_cli_integration(tmp_path: Path):
    """CLI --mode pr must write a JSON file with mode='pr' and all three blocks."""
    from bench.runner.cli import main

    case = _make_case(tmp_path)
    out_file = tmp_path / "pr_out.json"

    with (
        patch(
            "bench.runner.modes.pr.estimate",
            new=AsyncMock(return_value=_fake_estimate(case)),
        ),
        patch(
            "bench.runner.modes.pr.optimize_image",
            new=AsyncMock(return_value=_fake_optimize(case)),
        ),
    ):
        # Use --corpus pointing at tmp_path and a minimal manifest file.
        # We write a fake manifest JSON that the CLI can load.
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        # We'll supply the built manifest via a temp manifest file, but the
        # CLI loads manifests by name. Easiest path: patch load_cases.
        with patch(
            "bench.runner.cli.load_cases",
            return_value=[case],
        ):
            exit_code = main(
                [
                    "run",
                    "--mode",
                    "pr",
                    "--manifest",
                    "core",
                    "--repeat",
                    "1",
                    "--warmup",
                    "0",
                    "--quality-fast",
                    "--out",
                    str(out_file),
                ]
            )

    assert exit_code == 0, f"CLI exited with {exit_code}"
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert data["mode"] == "pr"
    cfg = data["config"]
    assert cfg.get("quality_fast") is True
    assert "stages" in cfg
    assert "estimate" in cfg["stages"]
    assert "timing" in cfg["stages"]

    iters = data.get("iterations", [])
    assert iters, "no iterations in output"
    it = iters[0]
    assert "accuracy" in it, "accuracy block missing from CLI pr output"


# ---------------------------------------------------------------------------
# Test 11 (real corpus): End-to-end on a real JPEG case
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build --manifest core` first",
)
def test_pr_mode_real_case_jpeg():
    """End-to-end integration: load a real JPEG, run pr mode, verify all blocks."""
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

    results = run_pr_sync([case], warmup=0, repeat=2, fast_quality=True)
    assert len(results) == 2, f"expected 2 timing iters, got {len(results)}"

    for r in results:
        assert "error" not in r, f"unexpected error: {r.get('error')}"
        assert "measurement" in r
        assert r["measurement"]["wall_ms"] > 0

        # Accuracy block
        acc = r.get("accuracy")
        assert isinstance(acc, dict), "accuracy block missing"
        assert acc["reduction_abs_error_pct_abs"] >= 0

        # Quality block — JPEG is lossy so must have SSIM
        q = r.get("quality")
        assert isinstance(q, dict), "quality block must be a dict for JPEG"
        assert q["ssim"] is not None, "ssim must be non-null for JPEG"
        assert 0.0 <= q["ssim"] <= 1.0, f"ssim={q['ssim']} out of [0, 1]"

        # Timing metadata
        assert "iteration" in r
        assert r["reduction_pct"] >= 0
        assert r["method"]
