"""Shared constants for the benchmark suite.

Single source of truth for sizes, quality levels, and quality presets
used across cases, runner, and report modules.
"""

from dataclasses import dataclass

from schemas import OptimizationConfig

# ---------------------------------------------------------------------------
# Size table — 3 core sizes covering exact mode and sample estimation paths
# ---------------------------------------------------------------------------

CORE_SIZES = [
    ("small", 300, 200),  # 60K pixels — below EXACT_PIXEL_THRESHOLD (150K)
    ("medium", 800, 600),  # 480K pixels — above pixel threshold, exercises sample estimation
    ("large", 1920, 1080),  # 2M pixels — well above, tests large-file estimation paths
]

# ---------------------------------------------------------------------------
# Quality levels for lossy photo formats (JPEG, WebP, AVIF, HEIC, JXL)
# ---------------------------------------------------------------------------

LOSSY_QUALITIES = [95, 75, 50]

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
    ),
)

MEDIUM = QualityPreset(
    name="MEDIUM",
    config=OptimizationConfig(
        quality=60,
        png_lossy=True,
        strip_metadata=True,
        progressive_jpeg=True,
    ),
)

LOW = QualityPreset(
    name="LOW",
    config=OptimizationConfig(
        quality=75,
        png_lossy=False,
        strip_metadata=True,
        progressive_jpeg=True,
    ),
)

ALL_PRESETS = [HIGH, MEDIUM, LOW]

PRESETS_BY_NAME = {p.name: p for p in ALL_PRESETS}

# ---------------------------------------------------------------------------
# Corpus groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusGroup:
    key: str
    label: str
    badge: str  # Short label for UI badges
    description: str


CORPUS_GROUPS = {
    "high_res": CorpusGroup(
        key="high_res",
        label="High-Res",
        badge="Hi-Res",
        description="2000+ px, >500KB",
    ),
    "standard": CorpusGroup(
        key="standard",
        label="Standard",
        badge="Std",
        description="800-2000px, 100-500KB",
    ),
    "compact": CorpusGroup(
        key="compact",
        label="Compact",
        badge="Cmpct",
        description="<800px, <100KB",
    ),
    "deep_color": CorpusGroup(
        key="deep_color",
        label="Deep-Color",
        badge="Deep",
        description="10/12-bit native encoding",
    ),
}
