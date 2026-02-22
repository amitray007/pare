"""Preset -> OptimizationConfig mapping for the estimation API."""

from schemas import OptimizationConfig

PRESET_CONFIGS = {
    "high": OptimizationConfig(quality=40, png_lossy=True, strip_metadata=True),
    "medium": OptimizationConfig(quality=60, png_lossy=True, strip_metadata=True),
    "low": OptimizationConfig(quality=80, png_lossy=False, strip_metadata=True),
}


def get_config_for_preset(preset: str) -> OptimizationConfig:
    """Convert a preset name to an OptimizationConfig.

    Args:
        preset: "high", "medium", or "low" (case-insensitive).

    Returns:
        OptimizationConfig with appropriate quality and flags.

    Raises:
        ValueError: If preset is not recognized.
    """
    key = preset.lower()
    if key not in PRESET_CONFIGS:
        raise ValueError(
            f"Invalid preset: '{preset}'. Must be 'high', 'medium', or 'low'."
        )
    return PRESET_CONFIGS[key]
