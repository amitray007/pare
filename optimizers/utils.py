"""Shared utilities for format-specific optimizers."""


def clamp_quality(quality: int, *, offset: int = 10, lo: int = 30, hi: int = 90) -> int:
    """Map an abstract quality value (1-100) to a format-specific quality.

    Each format encoder has its own quality scale. This function applies a
    linear offset and clamps to the format's valid range.

    Args:
        quality: Input quality value on a 1-100 scale.
        offset: Added to quality before clamping.
        lo: Minimum output quality.
        hi: Maximum output quality.

    Returns:
        Clamped quality value for the format encoder.
    """
    return max(lo, min(hi, quality + offset))


def binary_search_quality(
    encode_fn,
    original_size: int,
    target_reduction: float,
    lo: int,
    hi: int,
    max_iters: int = 5,
) -> bytes | None:
    """Binary search for the lowest quality whose output stays within a reduction cap.

    Used by JPEG and WebP optimizers to enforce max_reduction. The search
    finds the lowest quality (= most compression) that doesn't exceed the
    target reduction percentage.

    Args:
        encode_fn: Callable(quality: int) -> bytes. Format-specific encoder.
        original_size: Size of the original file in bytes.
        target_reduction: Maximum allowed reduction percentage (0-100).
        lo: Lower bound of quality range.
        hi: Upper bound of quality range.
        max_iters: Maximum binary search iterations (default 5).

    Returns:
        Encoded bytes at the capped quality, or None if even q=hi exceeds the cap.
    """
    out_hi = encode_fn(hi)
    red_hi = (1 - len(out_hi) / original_size) * 100
    if red_hi > target_reduction:
        return None  # Even highest quality exceeds cap

    best_out = out_hi

    for _ in range(max_iters):
        if hi - lo <= 1:
            break
        mid = (lo + hi) // 2
        out_mid = encode_fn(mid)
        red_mid = (1 - len(out_mid) / original_size) * 100
        if red_mid > target_reduction:
            lo = mid
        else:
            hi = mid
            best_out = out_mid

    return best_out
