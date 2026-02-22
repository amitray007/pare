"""Tests for preset -> OptimizationConfig mapping."""

from estimation.presets import get_config_for_preset


def test_high_preset_maps_to_quality_40():
    config = get_config_for_preset("high")
    assert config.quality == 40
    assert config.png_lossy is True


def test_medium_preset_maps_to_quality_60():
    config = get_config_for_preset("medium")
    assert config.quality == 60
    assert config.png_lossy is True


def test_low_preset_maps_to_quality_80():
    config = get_config_for_preset("low")
    assert config.quality == 80
    assert config.png_lossy is False


def test_preset_case_insensitive():
    assert get_config_for_preset("HIGH").quality == 40
    assert get_config_for_preset("High").quality == 40


def test_invalid_preset_raises():
    import pytest

    with pytest.raises(ValueError, match="Invalid preset"):
        get_config_for_preset("ultra")
