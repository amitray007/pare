"""Tests for bench.dashboard.samples and the quality-samples sub-page renderer.

Covers:
1. Sample generator — worst-case per format from mock pr-mode JSON.
2. Lossless formats return lossless=True with no thumb keys.
3. Missing corpus path — records have no thumb_b64 keys but other fields populate.
4. Page renderer — with full sample data writes valid HTML.
5. Page renderer — with no sample data writes a "no samples" placeholder.
6. Lossy format with no quality data in run — no_data=True stub is returned.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import pytest

from bench.dashboard.build import _find_repo_root, render_samples_page
from bench.dashboard.samples import (
    _build_lossless_record,
    _find_worst_ssim_per_format,
    build_sample_records,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = _find_repo_root(Path(__file__))


def _make_pr_iteration(
    fmt: str,
    case_id: str,
    preset: str = "high",
    ssim: float = 0.92,
    psnr: float = 30.0,
    input_size: int = 500_000,
    actual_size: int = 100_000,
    actual_reduction_pct: float = 80.0,
    iteration: int = 0,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "name": case_id.split(".")[0] if "." in case_id else case_id,
        "bucket": "small",
        "format": fmt,
        "preset": preset,
        "input_size": input_size,
        "iteration": iteration,
        "quality": {
            "ssim": ssim,
            "psnr_db": psnr,
            "ssimulacra2": None,
            "butteraugli_max": None,
            "butteraugli_3norm": None,
            "wall_ms": 12.0,
        },
        "optimize": {
            "wall_ms": 50.0,
            "actual_size": actual_size,
            "actual_reduction_pct": actual_reduction_pct,
            "method": "test",
        },
        "accuracy": {
            "size_rel_error_pct": 5.0,
            "reduction_error_pct": 2.0,
        },
        "reduction_pct": actual_reduction_pct,
        "optimized_size": actual_size,
        "method": "test",
    }


def _make_pr_run(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "mode": "pr",
        "timestamp": "2026-05-01T12:00:00Z",
        "git": {"branch": "feat/test", "commit": "abc1234567890"},
        "iterations": iterations,
        "stats": [],
    }


# ---------------------------------------------------------------------------
# Test 1 — worst-case per format
# ---------------------------------------------------------------------------


def test_find_worst_ssim_per_format_basic() -> None:
    """The iteration with the lowest SSIM per format is selected."""
    iters = [
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.95),
        _make_pr_iteration("jpeg", "img2.jpeg@high", ssim=0.88),  # worst
        _make_pr_iteration("webp", "img.webp@medium", ssim=0.97),
    ]
    worst = _find_worst_ssim_per_format(iters)
    assert "jpeg" in worst
    assert "webp" in worst
    assert float(worst["jpeg"]["quality"]["ssim"]) == pytest.approx(0.88)
    assert float(worst["webp"]["quality"]["ssim"]) == pytest.approx(0.97)


def test_find_worst_ssim_deduplicates_case_id() -> None:
    """Duplicate case_ids (timing reruns) are de-duplicated — first occurrence wins."""
    iters = [
        _make_pr_iteration("jpeg", "same.jpeg@high", ssim=0.90, iteration=0),
        _make_pr_iteration("jpeg", "same.jpeg@high", ssim=0.90, iteration=1),
        _make_pr_iteration("jpeg", "other.jpeg@high", ssim=0.85, iteration=0),
    ]
    worst = _find_worst_ssim_per_format(iters)
    assert float(worst["jpeg"]["quality"]["ssim"]) == pytest.approx(0.85)


def test_find_worst_ssim_skips_errors() -> None:
    """Iterations with error keys are skipped."""
    iters = [
        {
            "case_id": "bad.jpeg@high",
            "format": "jpeg",
            "error": {"phase": "optimize", "message": "boom"},
        },
        _make_pr_iteration("jpeg", "good.jpeg@high", ssim=0.95),
    ]
    worst = _find_worst_ssim_per_format(iters)
    assert "jpeg" in worst
    assert worst["jpeg"]["case_id"] == "good.jpeg@high"


def test_find_worst_ssim_skips_lossless() -> None:
    """PNG/GIF/BMP iterations are ignored (lossless — no quality block)."""
    iters = [
        _make_pr_iteration("png", "img.png@low", ssim=1.0),
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.90),
    ]
    worst = _find_worst_ssim_per_format(iters)
    assert "png" not in worst
    assert "jpeg" in worst


# ---------------------------------------------------------------------------
# Test 2 — lossless format records
# ---------------------------------------------------------------------------


def test_lossless_record_shape() -> None:
    """_build_lossless_record returns lossless=True and no thumb keys."""
    rec = _build_lossless_record("png")
    assert rec["lossless"] is True
    assert rec["format"] == "png"
    assert rec["ssim"] is None
    assert "orig_thumb_b64" not in rec
    assert "opt_thumb_b64" not in rec


def test_build_sample_records_includes_all_lossless() -> None:
    """build_sample_records always returns records for all lossless formats."""
    run = _make_pr_run([])
    records = build_sample_records(run, corpus_root=None)
    lossless_fmts = {r["format"] for r in records if r.get("lossless")}
    for fmt in ("png", "apng", "gif", "bmp", "tiff", "svg", "svgz"):
        assert fmt in lossless_fmts, f"{fmt} missing from lossless records"


# ---------------------------------------------------------------------------
# Test 3 — missing corpus → no thumb keys, other fields populated
# ---------------------------------------------------------------------------


def test_build_sample_records_no_corpus_has_no_thumbs() -> None:
    """With corpus_root=None, no orig_thumb_b64/opt_thumb_b64 are generated."""
    iters = [
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.88),
    ]
    run = _make_pr_run(iters)
    records = build_sample_records(run, corpus_root=None)
    jpeg_rec = next(r for r in records if r["format"] == "jpeg")
    assert "orig_thumb_b64" not in jpeg_rec
    assert "opt_thumb_b64" not in jpeg_rec
    # But numeric fields ARE present.
    assert jpeg_rec["ssim"] == pytest.approx(0.88, abs=1e-4)
    assert jpeg_rec["case_id"] == "img.jpeg@high"
    assert jpeg_rec["lossless"] is False


def test_build_sample_records_nonexistent_corpus_dir(tmp_path: Path) -> None:
    """corpus_root pointing to a nonexistent dir → graceful fallback, no thumbs."""
    nonexistent = tmp_path / "does_not_exist"
    iters = [
        _make_pr_iteration("webp", "img.webp@medium", ssim=0.94),
    ]
    run = _make_pr_run(iters)
    records = build_sample_records(run, corpus_root=nonexistent)
    webp_rec = next(r for r in records if r["format"] == "webp")
    assert "orig_thumb_b64" not in webp_rec
    assert webp_rec["ssim"] is not None


# ---------------------------------------------------------------------------
# Test 4 — page renderer with full data writes valid HTML
# ---------------------------------------------------------------------------


def test_render_samples_page_writes_html(tmp_path: Path) -> None:
    """render_samples_page writes quality-samples/index.html containing expected sections."""
    out_dir = tmp_path / "dash"
    out_dir.mkdir()

    iters = [
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.88),
        _make_pr_iteration("webp", "img.webp@medium", ssim=0.95),
    ]
    run = _make_pr_run(iters)

    render_samples_page(out_dir, run, corpus_root=None)

    samples_index = out_dir / "quality-samples" / "index.html"
    assert samples_index.exists(), "quality-samples/index.html was not written"

    html = samples_index.read_text(encoding="utf-8")
    # Structure checks
    assert "Visual Quality Samples" in html
    assert "← Back to scorecard" in html
    assert "JPEG" in html
    assert "WEBP" in html
    # Lossless section
    assert "PNG" in html
    assert "Lossless" in html or "pixel-identical" in html


def test_render_samples_page_includes_ssim_values(tmp_path: Path) -> None:
    """The rendered page includes the SSIM value from the worst-case iteration."""
    out_dir = tmp_path / "dash"
    out_dir.mkdir()

    iters = [
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.8765),
    ]
    run = _make_pr_run(iters)
    render_samples_page(out_dir, run, corpus_root=None)

    html = (out_dir / "quality-samples" / "index.html").read_text(encoding="utf-8")
    assert "0.8765" in html


# ---------------------------------------------------------------------------
# Test 5 — page renderer with no data writes placeholder
# ---------------------------------------------------------------------------


def test_render_samples_page_no_data_writes_placeholder(tmp_path: Path) -> None:
    """With scorecard_run=None, the page renders a 'no samples' notice."""
    out_dir = tmp_path / "dash"
    out_dir.mkdir()

    render_samples_page(out_dir, scorecard_run=None, corpus_root=None)

    samples_index = out_dir / "quality-samples" / "index.html"
    assert samples_index.exists()
    html = samples_index.read_text(encoding="utf-8")
    # Must contain a placeholder / notice about missing data
    assert "No quality samples" in html or "no samples" in html.lower() or "pr-mode" in html


def test_render_samples_page_quick_mode_writes_placeholder(tmp_path: Path) -> None:
    """A quick-mode run (no quality blocks) renders the 'no samples' notice."""
    out_dir = tmp_path / "dash"
    out_dir.mkdir()

    quick_run: dict[str, Any] = {
        "schema_version": 2,
        "mode": "quick",
        "timestamp": "2026-05-01T10:00:00Z",
        "git": {"branch": "main", "commit": "deadbeef"},
        "iterations": [
            {
                "case_id": "img.jpeg@high",
                "format": "jpeg",
                "preset": "high",
                "bucket": "small",
                "input_size": 100_000,
                "iteration": 0,
                "reduction_pct": 60.0,
                "method": "jpegli",
            }
        ],
        "stats": [],
    }
    render_samples_page(out_dir, scorecard_run=quick_run, corpus_root=None)

    html = (out_dir / "quality-samples" / "index.html").read_text(encoding="utf-8")
    assert "pr-mode" in html or "No quality samples" in html or "no samples" in html.lower()


# ---------------------------------------------------------------------------
# Test 6 — lossy format not in run → no_data stub
# ---------------------------------------------------------------------------


def test_build_sample_records_missing_format_is_stub() -> None:
    """A lossy format absent from the run produces a no_data=True stub record."""
    iters = [
        _make_pr_iteration("jpeg", "img.jpeg@high", ssim=0.90),
        # "heic", "avif", "jxl", "webp" are absent
    ]
    run = _make_pr_run(iters)
    records = build_sample_records(run, corpus_root=None)
    fmt_map = {r["format"]: r for r in records}

    for fmt in ("heic", "avif", "jxl", "webp"):
        assert fmt in fmt_map, f"{fmt} missing from records"
        assert fmt_map[fmt].get("no_data") is True


# ---------------------------------------------------------------------------
# Test 7 — main CLI with --no-with-samples skips samples dir
# ---------------------------------------------------------------------------


def test_main_no_with_samples_skips_samples_dir(tmp_path: Path) -> None:
    """--no-with-samples flag prevents quality-samples/ from being created."""
    from bench.dashboard.build import main

    out = tmp_path / "out"
    rc = main(
        [
            "--out-dir",
            str(out),
            "--repo",
            str(REPO_ROOT),
            "--no-with-samples",
        ]
    )
    assert rc == 0
    assert not (
        out / "quality-samples"
    ).exists(), "quality-samples/ should not exist when --no-with-samples is passed"


# ---------------------------------------------------------------------------
# Test 8 — main CLI --with-samples creates the samples dir
# ---------------------------------------------------------------------------


def test_main_with_samples_creates_samples_dir(tmp_path: Path) -> None:
    """--with-samples (default) creates quality-samples/index.html."""
    from bench.dashboard.build import main

    out = tmp_path / "out"
    rc = main(
        [
            "--out-dir",
            str(out),
            "--repo",
            str(REPO_ROOT),
            "--with-samples",
        ]
    )
    assert rc == 0
    assert (
        out / "quality-samples" / "index.html"
    ).exists(), "quality-samples/index.html missing with --with-samples"


# ---------------------------------------------------------------------------
# Test 9 — thumbnail generation with mocked Pillow (no corpus needed)
# ---------------------------------------------------------------------------


def test_b64_thumbnail_round_trips(tmp_path: Path) -> None:
    """_b64_thumbnail encodes a small image and produces valid base64 PNG."""
    from PIL import Image

    from bench.dashboard.samples import _b64_thumbnail

    img = Image.new("RGB", (400, 300), color=(128, 0, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    b64 = _b64_thumbnail(png_bytes)
    decoded = base64.b64decode(b64)
    # Should be a valid PNG
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
    # Thumbnail must be <= 300px in each dimension
    thumb = Image.open(io.BytesIO(decoded))
    assert thumb.width <= 300
    assert thumb.height <= 300


def test_b64_thumbnail_caps_at_300px() -> None:
    """An oversized image is resized to fit within 300×300."""
    from PIL import Image

    from bench.dashboard.samples import _b64_thumbnail

    img = Image.new("RGB", (1200, 900), color=(0, 128, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    b64 = _b64_thumbnail(buf.getvalue())
    decoded = base64.b64decode(b64)
    thumb = Image.open(io.BytesIO(decoded))
    assert thumb.width <= 300
    assert thumb.height <= 300
