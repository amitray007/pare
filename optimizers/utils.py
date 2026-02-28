"""Shared utilities for format-specific optimizers."""


def clamp_quality(quality: int, *, offset: int = 10, lo: int = 30, hi: int = 90) -> int:
    """Map Pare quality (1-100, lower=aggressive) to format-specific quality.

    Each format encoder has its own quality scale. This function applies a
    linear offset and clamps to the format's valid range.

    Args:
        quality: Pare quality value (1-100).
        offset: Added to quality before clamping.
        lo: Minimum output quality.
        hi: Maximum output quality.

    Returns:
        Clamped quality value for the format encoder.
    """
    return max(lo, min(hi, quality + offset))
