"""Tests for LSM source-quality estimation in estimation/jpeg_header.py.

Uses Pillow to generate JPEG fixtures, then parses them with parse_jpeg_header()
and feeds the DQT tables to estimate_source_quality_lsm().
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from estimation.jpeg_header import estimate_source_quality_lsm, parse_jpeg_header


def _make_jpeg(
    width: int = 128,
    height: int = 96,
    mode: str = "RGB",
    quality: int = 75,
    qtables: list | None = None,
) -> bytes:
    img = Image.new(mode, (width, height), color=100)
    buf = io.BytesIO()
    kwargs: dict = {"format": "JPEG", "quality": quality}
    if qtables is not None:
        kwargs["qtables"] = qtables
    img.save(buf, **kwargs)
    return buf.getvalue()


def _dqt_for_quality(q: int) -> tuple[list[int], list[int]]:
    """Return (luma, chroma) DQT tables by parsing a freshly-encoded JPEG."""
    data = _make_jpeg(quality=q)
    hdr = parse_jpeg_header(data)
    assert hdr is not None, f"parse_jpeg_header returned None for q={q}"
    assert hdr.dqt_luma, "missing luma table"
    assert hdr.dqt_chroma is not None, "missing chroma table"
    return hdr.dqt_luma, hdr.dqt_chroma


# ---------------------------------------------------------------------------
# Standard libjpeg qualities
# ---------------------------------------------------------------------------


class TestStandardQualities:
    def test_q50_within_2(self):
        luma, chroma = _dqt_for_quality(50)
        q_est, nse = estimate_source_quality_lsm(luma, chroma)
        assert abs(q_est - 50) <= 2, f"expected 50 ±2, got {q_est}"
        assert nse > 0.99, f"expected NSE > 0.99 for q=50, got {nse:.4f}"

    @pytest.mark.parametrize("q", [10, 30, 70, 90, 95])
    def test_various_qualities_within_5(self, q: int):
        luma, chroma = _dqt_for_quality(q)
        q_est, nse = estimate_source_quality_lsm(luma, chroma)
        assert abs(q_est - q) <= 5, f"q={q}: expected ±5, got {q_est}"
        assert nse > 0.95, f"q={q}: expected NSE > 0.95, got {nse:.4f}"


# ---------------------------------------------------------------------------
# Custom quantization tables (not Annex K — NSE should be low)
# ---------------------------------------------------------------------------


class TestCustomQuantization:
    def test_custom_uniform_qtables_low_nse(self):
        """Uniform Q-table (not Annex K) → NSE < 0.95 (fallback signal)."""
        # Uniform table with all 16 — definitely not Annex K scaled
        custom_qtable = [16] * 64
        data = _make_jpeg(qtables=[custom_qtable, custom_qtable])
        hdr = parse_jpeg_header(data)
        assert hdr is not None
        if hdr.dqt_luma and hdr.dqt_chroma:
            _q_est, nse = estimate_source_quality_lsm(hdr.dqt_luma, hdr.dqt_chroma)
            assert nse < 0.95, f"expected NSE < 0.95 for custom tables, got {nse:.4f}"


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_all_zero_dqt_no_crash(self):
        """All-zero DQT → nse=0, no crash."""
        luma = [0] * 64
        q_est, nse = estimate_source_quality_lsm(luma, None)
        assert nse == pytest.approx(0.0)
        assert 1 <= q_est <= 100

    def test_constant_dqt_no_crash(self):
        """All-99 DQT → degenerate but no crash."""
        luma = [99] * 64
        chroma = [99] * 64
        q_est, nse = estimate_source_quality_lsm(luma, chroma)
        assert 1 <= q_est <= 100
        assert 0.0 <= nse <= 1.0

    def test_grayscale_luma_only(self):
        """dqt_chroma=None (grayscale) still works."""
        luma, _ = _dqt_for_quality(75)
        q_est, nse = estimate_source_quality_lsm(luma, None)
        assert 1 <= q_est <= 100
        assert 0.0 <= nse <= 1.0
        # Luma-only estimation should still produce reasonable NSE
        assert nse > 0.5, f"grayscale NSE surprisingly low: {nse:.4f}"

    def test_nse_in_unit_interval(self):
        """NSE is always clamped to [0, 1]."""
        luma, chroma = _dqt_for_quality(85)
        _, nse = estimate_source_quality_lsm(luma, chroma)
        assert 0.0 <= nse <= 1.0
