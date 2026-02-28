"""Preset -> OptimizationConfig mapping for the estimation API.

Delegates to benchmarks.constants as the single source of truth for preset
definitions. This module provides the get_config_for_preset() convenience
function used by the /estimate endpoint.
"""

from benchmarks.constants import PRESETS_BY_NAME


def get_config_for_preset(preset: str) -> "OptimizationConfig":
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
