"""Preset -> OptimizationConfig mapping for the estimation API.

Source of truth for the preset definitions used by the /estimate
endpoint. The bench harness in `bench/` keeps its own quality mapping
(`bench.runner.case.PRESET_QUALITY`) tracking the same numeric values
so bench results reflect production /estimate behavior.
"""

from dataclasses import dataclass

from schemas import OptimizationConfig


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


def get_config_for_preset(preset: str) -> OptimizationConfig:
    """Convert a preset name to an OptimizationConfig.

    Args:
        preset: "high", "medium", or "low" (case-insensitive).

    Returns:
        OptimizationConfig with appropriate quality and flags.

    Raises:
        ValueError: If preset is not recognized.
    """
    key = preset.upper()
    if key not in PRESETS_BY_NAME:
        raise ValueError(f"Invalid preset: '{preset}'. Must be 'high', 'medium', or 'low'.")
    return PRESETS_BY_NAME[key].config
