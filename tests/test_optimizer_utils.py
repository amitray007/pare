"""Tests for shared optimizer utilities."""

from optimizers.utils import clamp_quality


def test_clamp_quality_default():
    """Default offset=10, lo=30, hi=90."""
    assert clamp_quality(40) == 50
    assert clamp_quality(80) == 90
    assert clamp_quality(15) == 30  # clamped to lo
    assert clamp_quality(95) == 90  # clamped to hi


def test_clamp_quality_custom_range():
    """JXL uses hi=95."""
    assert clamp_quality(85, hi=95) == 95
    assert clamp_quality(90, hi=95) == 95  # 90+10=100, clamped to 95


def test_clamp_quality_custom_offset():
    """Custom offset."""
    assert clamp_quality(50, offset=0) == 50
    assert clamp_quality(50, offset=20) == 70
