"""Tests for Settings / config.py behaviour."""

from unittest.mock import patch

import pytest


def test_compression_semaphore_floor_on_single_cpu():
    """When os.cpu_count() returns 1, semaphore must be at least 2."""
    with patch("os.cpu_count", return_value=1):
        from config import Settings

        s = Settings()

    assert (
        s.compression_semaphore_size >= 2
    ), f"Expected semaphore >= 2 on 1-cpu host, got {s.compression_semaphore_size}"


def test_fitted_estimator_mode_rejects_invalid_value():
    """fitted_estimator_mode must reject values other than 'off' or 'active' at boot."""
    from pydantic import ValidationError

    from config import Settings

    with pytest.raises(ValidationError, match="Input should be 'off' or 'active'"):
        Settings(fitted_estimator_mode="actve")  # type: ignore[arg-type]
