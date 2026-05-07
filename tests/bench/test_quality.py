"""Tests for the quality mode perceptual metric helpers and runner.

Tests 1-5 are pure-unit tests with no corpus dependency.
Tests 6-7 require the synthesized corpus on disk; they are guarded by
skipif so CI doesn't fail if the corpus hasn't been built.
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from bench.runner.quality import butteraugli_scores, psnr_db, ssim, ssimulacra2_score

# ---------------------------------------------------------------------------
# Corpus sentinel
# ---------------------------------------------------------------------------
_CORPUS_ROOT = Path("tests/corpus")
_SMALL_JPEG = _CORPUS_ROOT / "small" / "jpeg" / "photo_perlin_small_jpeg.jpeg"

# ---------------------------------------------------------------------------
# Test 1: SSIM on identical arrays returns 1.0
# ---------------------------------------------------------------------------


def test_ssim_identical_arrays_returns_one():
    """SSIM of a reference with itself must be exactly 1.0."""
    rng = np.random.default_rng(0)
    arr = rng.random((32, 32, 3), dtype=np.float32)
    result = ssim(arr, arr)
    assert result is not None
    assert abs(result - 1.0) < 1e-5, f"expected ~1.0, got {result}"


# ---------------------------------------------------------------------------
# Test 2: SSIM on uncorrelated random noise returns a low value
# ---------------------------------------------------------------------------


def test_ssim_random_arrays_returns_low_value():
    """SSIM between two uncorrelated noise images should be well below 0.5."""
    rng = np.random.default_rng(42)
    arr_a = rng.random((64, 64, 3), dtype=np.float32)
    arr_b = rng.random((64, 64, 3), dtype=np.float32)
    result = ssim(arr_a, arr_b)
    assert result is not None
    assert result < 0.5, f"expected SSIM < 0.5 for random noise, got {result}"


# ---------------------------------------------------------------------------
# Test 3: PSNR on identical arrays returns None (infinite PSNR)
# ---------------------------------------------------------------------------


def test_psnr_identical_arrays_returns_none():
    """Identical arrays → MSE=0 → PSNR is infinite → must return None."""
    rng = np.random.default_rng(7)
    arr = rng.random((32, 32, 3), dtype=np.float32)
    result = psnr_db(arr, arr)
    assert result is None, f"expected None for identical arrays, got {result}"


# ---------------------------------------------------------------------------
# Test 4: ssimulacra2 subprocess returns a float when binary is present
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("ssimulacra2") is None,
    reason="ssimulacra2 binary not on PATH; install libjxl tools to enable",
)
def test_ssimulacra2_subprocess_returns_float_when_binary_present():
    """ssimulacra2_score returns a finite float in reasonable range."""
    # Create a reference PNG and a lossy-encoded distorted PNG
    rng = np.random.default_rng(99)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    ref_img = Image.fromarray(arr, mode="RGB")

    # Distort via JPEG encoding at low quality
    buf = io.BytesIO()
    ref_img.save(buf, format="JPEG", quality=30)
    buf.seek(0)
    dist_img = Image.open(buf).convert("RGB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_png = tmp / "ref.png"
        dist_png = tmp / "dist.png"
        ref_img.save(str(ref_png), format="PNG")
        dist_img.save(str(dist_png), format="PNG")

        score = ssimulacra2_score(ref_png, dist_png)

    assert score is not None, "ssimulacra2_score returned None unexpectedly"
    assert isinstance(score, float)
    # Score should be finite and within documented range for low-quality JPEG
    assert -50.0 < score < 100.0, f"score {score} out of expected range"


# ---------------------------------------------------------------------------
# Test 5: butteraugli subprocess parses both max and 3-norm
# ---------------------------------------------------------------------------

_BUTTERAUGLI_AVAILABLE = (
    shutil.which("butteraugli_main") is not None
    or Path("/opt/homebrew/opt/jpeg-xl/bin/butteraugli_main").is_file()
)


@pytest.mark.skipif(
    not _BUTTERAUGLI_AVAILABLE,
    reason="butteraugli_main not found; install libjxl tools to enable",
)
def test_butteraugli_subprocess_parses_max_and_3norm():
    """butteraugli_scores returns two positive finite floats for degraded images."""
    rng = np.random.default_rng(33)
    arr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    ref_img = Image.fromarray(arr, mode="RGB")

    buf = io.BytesIO()
    ref_img.save(buf, format="JPEG", quality=30)
    buf.seek(0)
    dist_img = Image.open(buf).convert("RGB")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        ref_png = tmp / "ref.png"
        dist_png = tmp / "dist.png"
        ref_img.save(str(ref_png), format="PNG")
        dist_img.save(str(dist_png), format="PNG")

        ba_max, ba_norm = butteraugli_scores(ref_png, dist_png)

    assert ba_max is not None, "butteraugli_max was None"
    assert ba_norm is not None, "butteraugli_3norm was None"
    assert isinstance(ba_max, float)
    assert isinstance(ba_norm, float)
    # butteraugli values are positive for non-identical images
    assert ba_max >= 0.0, f"butteraugli_max={ba_max} should be >= 0"
    assert ba_norm >= 0.0, f"butteraugli_3norm={ba_norm} should be >= 0"


# ---------------------------------------------------------------------------
# Test 6: quality mode runs a real lossy case end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build --manifest core` first",
)
def test_quality_mode_runs_a_real_case():
    """End-to-end: load a real JPEG corpus file, run quality mode, check schema."""
    from bench.runner.case import Case
    from bench.runner.modes.quality import run_quality_sync

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

    results = run_quality_sync([case])
    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    r = results[0]

    # Top-level required fields
    assert r["case_id"] == "photo_perlin_small_jpeg.jpeg@high"
    assert r["format"] == "jpeg"
    assert r["preset"] == "high"
    assert r["iteration"] == 0
    assert r["input_size"] > 0
    assert "error" not in r, f"unexpected error: {r.get('error')}"

    # Measurement block (optimize-step only)
    assert "measurement" in r
    assert r["measurement"]["wall_ms"] > 0

    # Quality block must be present
    q = r.get("quality")
    assert isinstance(q, dict), f"quality block missing or wrong type: {q!r}"

    # Pure-numpy metrics should always succeed
    assert q["ssim"] is not None, "ssim should be non-null for a real JPEG case"
    assert 0.0 <= q["ssim"] <= 1.0, f"ssim={q['ssim']} out of [0, 1]"

    # PSNR: either a positive float or None with perfect_match=True
    if q.get("perfect_match"):
        assert q["psnr_db"] is None
    else:
        assert q["psnr_db"] is not None, "psnr_db should be non-null for lossy JPEG"
        assert q["psnr_db"] > 0.0, f"psnr_db={q['psnr_db']} should be positive"

    assert "wall_ms" in q
    assert q["wall_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Test 7: quality mode skips lossless format cases
# ---------------------------------------------------------------------------


def test_quality_mode_skips_lossless_format(tmp_path: Path):
    """PNG cases must be filtered out before iteration."""
    from bench.corpus.builder import build
    from bench.corpus.manifest import Bucket, Manifest, ManifestEntry
    from bench.runner.case import load_cases
    from bench.runner.modes.quality import run_quality_sync

    # Build a tiny PNG corpus in tmp_path.
    # 64×48 synthesized PNG ≈ 9 KB → falls in the "tiny" bucket.
    m = Manifest(
        name="t",
        library_versions={},
        entries=[
            ManifestEntry(
                name="qual_lossless_test",
                bucket=Bucket.TINY,
                content_kind="photo_noise",
                seed=13,
                width=64,
                height=48,
                output_formats=["png"],
            )
        ],
    )
    outcome = build(m, tmp_path)
    assert outcome.ok, outcome.bucket_violations

    cases = load_cases(m, tmp_path, preset_filter={"high"})
    assert cases, "expected at least one PNG case from the builder"
    assert all(c.fmt == "png" for c in cases)

    results = run_quality_sync(cases)
    # All PNG cases are filtered → results list is empty
    assert results == [], f"expected empty results for lossless-only cases, got {len(results)} rows"


# ---------------------------------------------------------------------------
# Test 8: --quality-fast skips subprocess metrics (ssimulacra2 + butteraugli)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus file not present; run `python -m bench.corpus build --manifest core` first",
)
def test_quality_fast_skips_subprocess_metrics():
    """fast=True: ssim/psnr_db populated; ssimulacra2/butteraugli_max null; wall_ms < 500."""
    from bench.runner.case import Case
    from bench.runner.modes.quality import run_quality_sync

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

    results = run_quality_sync([case], fast=True)
    assert len(results) == 1, f"expected 1 result, got {len(results)}"
    r = results[0]

    assert "error" not in r, f"unexpected error: {r.get('error')}"
    q = r.get("quality")
    assert isinstance(q, dict), f"quality block missing or wrong type: {q!r}"

    # Pure-numpy metrics must be populated
    assert q["ssim"] is not None, "ssim should be non-null in fast mode"
    assert 0.0 <= q["ssim"] <= 1.0, f"ssim={q['ssim']} out of [0, 1]"
    # psnr_db is None only when perfect_match is True (lossless result)
    if not q.get("perfect_match"):
        assert q["psnr_db"] is not None, "psnr_db should be non-null for lossy JPEG"

    # Subprocess metrics must be null in fast mode
    assert (
        q["ssimulacra2"] is None
    ), f"ssimulacra2 should be null in fast mode, got {q['ssimulacra2']}"
    assert (
        q["butteraugli_max"] is None
    ), f"butteraugli_max should be null in fast mode, got {q['butteraugli_max']}"

    # Fast mode must be well under subprocess wall time (< 500ms)
    assert q["wall_ms"] < 500, f"fast mode wall_ms={q['wall_ms']} should be < 500ms"


# ---------------------------------------------------------------------------
# Test 9: --quality-fast CLI flag wires through to config and output JSON
# ---------------------------------------------------------------------------


def test_quality_fast_cli_flag(tmp_path: Path):
    """CLI --quality-fast flag sets config.metrics and config.quality_fast in output JSON."""
    import json

    from bench.runner.cli import main

    out_file = tmp_path / "out.json"

    # Only run if the corpus has been built (same guard as test 6/8).
    if not _SMALL_JPEG.exists():
        pytest.skip(
            "corpus file not present; run `python -m bench.corpus build --manifest core` first"
        )

    exit_code = main(
        [
            "run",
            "--mode",
            "quality",
            "--quality-fast",
            "--manifest",
            "core",
            "--fmt",
            "jpeg",
            "--bucket",
            "small",
            "--out",
            str(out_file),
        ]
    )
    assert exit_code == 0, f"bench.run exited with non-zero code {exit_code}"
    assert out_file.exists(), "output JSON was not written"

    data = json.loads(out_file.read_text())
    cfg = data.get("config", {})

    # Config must reflect fast mode
    assert cfg.get("quality_fast") is True, f"config.quality_fast not True: {cfg}"
    assert cfg.get("metrics") == [
        "ssim",
        "psnr",
    ], f"config.metrics should be ['ssim', 'psnr'] in fast mode, got {cfg.get('metrics')}"

    # At least one iteration must have quality data with null subprocess fields
    quality_iters = [it for it in data.get("iterations", []) if "quality" in it]
    assert quality_iters, "no quality iterations in output"
    q = quality_iters[0]["quality"]
    assert (
        q["ssimulacra2"] is None
    ), f"ssimulacra2 should be null in fast mode, got {q['ssimulacra2']}"
    assert (
        q["butteraugli_max"] is None
    ), f"butteraugli_max should be null in fast mode, got {q['butteraugli_max']}"
