"""Centralized threshold constants for bench quality/speed/accuracy gates.

These constants are the single source of truth used by:
- bench/runner/report/markdown.py (PR comment quality tables)
- bench/dashboard/scorecard.py (dashboard scorecard view)

Edit here to change thresholds globally.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# SSIM quality thresholds — per-preset floor.
# Cases below their preset's threshold are counted as failures.
# ---------------------------------------------------------------------------

SSIM_THRESHOLDS: dict[str, float] = {
    "high": 0.95,  # q=40, aggressive lossy
    "medium": 0.97,  # q=60, balanced
    "low": 0.99,  # q=75, near-lossless
}

# Default when preset can't be determined.
SSIM_DEFAULT: float = SSIM_THRESHOLDS["medium"]


# ---------------------------------------------------------------------------
# Latency p95 SLOs by size bucket (milliseconds)
# ---------------------------------------------------------------------------

LATENCY_P95_SLOS_MS: dict[str, int] = {
    "tiny": 100,
    "small": 500,
    "medium": 2000,
    "large": 8000,
    "xlarge": 20000,
}

# Format-specific relaxations for known-slow encoders.
# Multiplier applied to the bucket SLO.
# HEIC and JXL get +50% on large/xlarge buckets.
LATENCY_FORMAT_RELAX: dict[tuple[str, str], float] = {
    ("heic", "large"): 1.5,
    ("heic", "xlarge"): 1.5,
    ("jxl", "large"): 1.5,
    ("jxl", "xlarge"): 1.5,
}


# ---------------------------------------------------------------------------
# Estimation accuracy targets (absolute error matters, sign does not)
# ---------------------------------------------------------------------------

ESTIMATION_SIZE_REL_ERROR: dict[str, float] = {
    "median_max": 5.0,  # percent
    "p95_max": 15.0,  # percent
}

ESTIMATION_REDUCTION_ERROR: dict[str, float] = {
    "median_max": 5.0,  # percentage points
    "p95_max": 10.0,  # percentage points
}
