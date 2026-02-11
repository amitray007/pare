"""Shared constants for the benchmark suite.

Single source of truth for sizes, quality levels, generator mappings,
and quality presets used across cases, runner, and report modules.
"""

from dataclasses import dataclass

from schemas import OptimizationConfig

# ---------------------------------------------------------------------------
# Size table — all raster dimensions
# ---------------------------------------------------------------------------

SIZES = [
    ("tiny",     64,   64),
    ("small-l",  300,  200),
    ("small-p",  200,  300),
    ("medium-l", 800,  600),
    ("medium-p", 600,  800),
    ("square",   512,  512),
    ("large-l",  1920, 1080),
    ("large-p",  1080, 1920),
]


def sizes_matching(*prefixes: str) -> list[tuple[str, int, int]]:
    """Return sizes whose name starts with any of the given prefixes."""
    return [(s, w, h) for s, w, h in SIZES if any(s.startswith(p) for p in prefixes)]


def sizes_excluding(*prefixes: str) -> list[tuple[str, int, int]]:
    """Return all sizes except those whose name starts with any prefix."""
    return [(s, w, h) for s, w, h in SIZES if not any(s.startswith(p) for p in prefixes)]


# ---------------------------------------------------------------------------
# Content / format configuration
# ---------------------------------------------------------------------------

# Content types allowed at large sizes (others are skipped to keep runtime sane)
LARGE_ALLOWED_CONTENT = {"photo", "screenshot", "graphic"}

# JPEG quality levels and per-size filters
JPEG_QUALITIES = [98, 92, 85, 75, 60, 40]
JPEG_LARGE_QUALITIES = {98, 85, 60}
JPEG_TINY_QUALITIES = {98, 60}
JPEG_SCREENSHOT_QUALITY = 95

# WebP quality levels
WEBP_QUALITIES = [95, 80, 60]
WEBP_LARGE_QUALITIES = {80}

# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityPreset:
    name: str
    config: OptimizationConfig

    def __str__(self) -> str:
        return self.name


HIGH = QualityPreset(
    name="HIGH",
    config=OptimizationConfig(
        quality=40,
        png_lossy=True,
        strip_metadata=True,
        progressive_jpeg=True,
        max_reduction=70.0,
    ),
)

MEDIUM = QualityPreset(
    name="MEDIUM",
    config=OptimizationConfig(
        quality=60,
        png_lossy=True,
        strip_metadata=True,
        progressive_jpeg=False,
        max_reduction=50.0,
    ),
)

LOW = QualityPreset(
    name="LOW",
    config=OptimizationConfig(
        quality=80,
        png_lossy=False,
        strip_metadata=False,
        progressive_jpeg=False,
        max_reduction=25.0,
    ),
)

ALL_PRESETS = [HIGH, MEDIUM, LOW]

PRESETS_BY_NAME = {p.name: p for p in ALL_PRESETS}
