"""Tests for shared optimizer utilities."""

from optimizers.utils import binary_search_quality, clamp_quality


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


def test_binary_search_finds_quality_within_cap():
    """Binary search finds quality where reduction stays under target."""

    def encode_fn(quality: int) -> bytes:
        # Lower quality -> smaller output (linear simulation)
        size = int(1000 * (quality / 100))
        return b"x" * max(1, size)

    original_size = 1000
    target_reduction = 30.0  # cap at 30% reduction

    result = binary_search_quality(encode_fn, original_size, target_reduction, lo=40, hi=100)
    assert result is not None
    reduction = (1 - len(result) / original_size) * 100
    assert reduction <= target_reduction + 1.0  # small tolerance


def test_binary_search_returns_none_when_q100_exceeds():
    """Returns None when even q=100 exceeds the cap."""

    def encode_fn(quality: int) -> bytes:
        # Even q=100 produces 50% reduction
        return b"x" * 500

    result = binary_search_quality(encode_fn, 1000, target_reduction=10.0, lo=40, hi=100)
    assert result is None


def test_binary_search_max_iterations():
    """Respects max_iters limit."""
    call_count = 0

    def encode_fn(quality: int) -> bytes:
        nonlocal call_count
        call_count += 1
        size = int(1000 * (quality / 100))
        return b"x" * max(1, size)

    binary_search_quality(encode_fn, 1000, target_reduction=30.0, lo=1, hi=100, max_iters=3)
    # 1 call for q=100 check + up to 3 iterations
    assert call_count <= 4
