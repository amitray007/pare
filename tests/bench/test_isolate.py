"""Tests for isolated-subprocess timing mode (--isolate).

These tests exercise ``bench.runner.isolate.run_iteration_in_worker`` and
the ``--isolate`` flag wired through ``bench.runner.modes.timing`` and
``bench.runner.cli``.

The first three tests require real corpus files on disk and are guarded by
a skipif so CI doesn't fail if the corpus hasn't been built yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.runner.case import Case
from bench.runner.isolate import run_iteration_in_worker

# ---------------------------------------------------------------------------
# Corpus-presence gate
# ---------------------------------------------------------------------------

_CORPUS_ROOT = Path("tests/corpus")
_SMALL_JPEG = _CORPUS_ROOT / "small" / "jpeg" / "photo_perlin_small_jpeg.jpeg"
_XLARGE_PNG = _CORPUS_ROOT / "xlarge" / "png" / "photo_noise_xlarge_png.png"

pytestmark = pytest.mark.skipif(
    not _SMALL_JPEG.exists(),
    reason="corpus not built; run `python -m bench.corpus build --manifest core` first",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_jpeg_case() -> Case:
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


def _xlarge_png_case() -> Case:
    return Case(
        case_id="photo_noise_xlarge_png.png@high",
        name="photo_noise_xlarge_png",
        bucket="xlarge",
        fmt="png",
        preset="high",
        quality=40,
        file_path=_XLARGE_PNG,
        input_size=_XLARGE_PNG.stat().st_size,
    )


# ---------------------------------------------------------------------------
# Test 1: basic shape check
# ---------------------------------------------------------------------------


def test_run_iteration_in_worker_returns_per_iteration_dict():
    """Worker returns a dict with the expected per-iteration keys."""
    case = _small_jpeg_case()
    result = run_iteration_in_worker(case)

    # Must not be a failure dict
    assert "error" not in result, f"unexpected error: {result.get('error')}"

    # Identity fields
    assert result["case_id"] == case.case_id
    assert result["name"] == case.name
    assert result["bucket"] == case.bucket
    assert result["format"] == case.fmt
    assert result["preset"] == case.preset
    assert result["input_size"] > 0

    # Measurement sub-dict
    assert "measurement" in result
    m = result["measurement"]
    assert m["wall_ms"] > 0, f"wall_ms should be positive, got {m['wall_ms']}"
    assert "parent_peak_rss_kb" in m
    assert "children_peak_rss_kb" in m
    assert "peak_rss_kb" in m

    # Optimization output
    assert "reduction_pct" in result
    assert "method" in result
    assert result["method"]  # non-empty
    assert "optimized_size" in result
    assert result["optimized_size"] > 0

    # Tool invocations list (may be empty for all-Pillow paths)
    assert "tool_invocations" in result
    for inv in result["tool_invocations"]:
        assert "tool" in inv
        assert "wall_ms" in inv
        assert "exit_code" in inv


# ---------------------------------------------------------------------------
# Test 2: clean RSS per case — no watermark contamination
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _XLARGE_PNG.exists(),
    reason="xlarge PNG not present; run `python -m bench.corpus build --manifest core` first",
)
def test_isolate_produces_clean_per_case_rss():
    """Isolated runs don't inherit each other's RSS watermark.

    Strategy:
    - Run the large case first (high allocator pressure).
    - Run the small case second.
    - In isolated mode each has its own fresh process, so the small case's
      ``parent_peak_rss_kb`` should be clearly less than the large case's.
    - If both ran in the same process, the small case would inherit the
      large case's high-water mark and the values would be equal or very
      close.
    """
    large_case = _xlarge_png_case()
    small_case = _small_jpeg_case()

    large_result = run_iteration_in_worker(large_case)
    small_result = run_iteration_in_worker(small_case)

    assert "error" not in large_result, large_result.get("error")
    assert "error" not in small_result, small_result.get("error")

    large_rss = large_result["measurement"]["parent_peak_rss_kb"]
    small_rss = small_result["measurement"]["parent_peak_rss_kb"]

    # The large PNG case decodes a much bigger image; its parent RSS should
    # exceed the small JPEG case's. We use a factor of 1.5× as a conservative
    # lower-bound (in practice the ratio is typically 3–10×).
    assert large_rss > small_rss, (
        f"Expected large_rss ({large_rss} KB) > small_rss ({small_rss} KB). "
        "If they're equal the isolation isn't cleaning the watermark."
    )
    assert large_rss > small_rss * 1.5, (
        f"large_rss={large_rss} KB should be clearly larger than "
        f"small_rss={small_rss} KB (expected ratio > 1.5×). "
        "Worker RSS may not be isolated."
    )


# ---------------------------------------------------------------------------
# Test 3: worker exception is surfaced as failure dict, not a crash
# ---------------------------------------------------------------------------


def test_isolate_handles_worker_exception():
    """If the worker raises (or pool.apply fails), run_iteration_in_worker
    returns a failure dict and does not crash the caller.

    We trigger the failure by passing a corrupted corpus file (zero bytes),
    which causes format detection to raise ``UnsupportedFormatError`` inside
    the worker subprocess.  The parent catches the remote exception and
    wraps it in the standard failure shape.
    """
    import tempfile

    case = _small_jpeg_case()

    # Create a temporary file with garbage bytes to force a format-detection error.
    with tempfile.NamedTemporaryFile(suffix=".jpeg", delete=False) as tf:
        tf.write(b"\x00" * 32)
        bad_path = Path(tf.name)

    bad_case = Case(
        case_id=case.case_id,
        name=case.name,
        bucket=case.bucket,
        fmt=case.fmt,
        preset=case.preset,
        quality=case.quality,
        file_path=bad_path,
        input_size=32,
    )

    try:
        result = run_iteration_in_worker(bad_case)
    finally:
        bad_path.unlink(missing_ok=True)

    assert "error" in result, f"expected failure dict, got: {result}"
    assert isinstance(result["error"], str)
    assert len(result["error"]) > 0
    assert result["case_id"] == case.case_id


# ---------------------------------------------------------------------------
# Test 4: --isolate flag appears in output JSON config
# ---------------------------------------------------------------------------


def test_isolate_flag_in_cli(tmp_path: Path):
    """Invoking CLI with --isolate sets config.isolate=True in output JSON."""
    from bench.runner.cli import main

    out_path = tmp_path / "out.json"
    exit_code = main(
        [
            "run",
            "--mode",
            "timing",
            "--isolate",
            "--manifest",
            "core",
            "--fmt",
            "jpeg",
            "--bucket",
            "small",
            "--warmup",
            "0",
            "--repeat",
            "1",
            "--out",
            str(out_path),
        ]
    )

    assert exit_code == 0, f"CLI exited with code {exit_code}"
    assert out_path.exists(), "output JSON was not written"

    run_data = json.loads(out_path.read_text())
    assert (
        run_data["config"].get("isolate") is True
    ), f"expected config.isolate=True, got config={run_data['config']}"
    # Sanity: there should be at least one iteration
    assert len(run_data["iterations"]) >= 1
