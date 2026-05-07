"""Tests for estimation/models/__init__.py and estimation/models/_artifact.py.

Covers:
- Missing file → LoadFailed("missing")
- Corrupt JSON → LoadFailed("parse_error")
- model_version mismatch → LoadFailed("version_mismatch")
- Happy path → Loaded(model) with correct PngModel / JpegHeaderModel fields
- load_png_model() / load_jpeg_header_model() never raise (including on all failure modes)
"""

import json
from pathlib import Path

import pytest

from estimation.models import (
    Loaded,
    LoadedHeader,
    LoadedJpeg,
    LoadFailed,
    load_jpeg_header_model,
    load_png_header_model,
    load_png_model,
)
from estimation.models._artifact import JpegHeaderModel as JpegHeaderModelDirect
from estimation.models._artifact import PngHeaderModel as PngHeaderModelDirect
from estimation.models._artifact import PngModel as PngModelDirect

# ---------------------------------------------------------------------------
# Minimal valid artifact payload
# ---------------------------------------------------------------------------

_VALID_ARTIFACT: dict = {
    "model_version": 2,
    "format": "png",
    "features": [
        "has_alpha",
        "log10_unique_colors",
        "mean_sobel",
        "edge_density",
        "quality",
        "log10_orig_pixels",
        "input_bpp",
    ],
    "supported_modes": ["RGB", "RGBA", "L", "LA", "P"],
    "scaler": {
        "mean": [0.0, 2.5, 15.0, 0.1, 70.0, 5.0, 8.0],
        "scale": [1.0, 1.2, 10.0, 0.05, 20.0, 0.8, 4.0],
    },
    "coefficients": {
        "intercept": 0.5,
        "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
        "knot_beta": 0.3,
        "knot_q50_beta": -0.01,
        "knot_q70_beta": 0.02,
    },
    "knot_log10_unique_colors": 3.3,
    "knot_q50": 50.0,
    "knot_q70": 70.0,
    "training_envelope": {"log10_unique_colors": [0.0, 5.7], "mean_sobel": [0.0, 80.0]},
    "training_corpus_sha256": "abc123def456",
    "git_sha": "cafebabe",
    "fit_environment": {
        "sklearn_version": "1.5.0",
        "numpy_version": "2.0.2",
        "openblas_threads": 1,
    },
    "created_at": "2026-05-07T00:00:00Z",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(tmp_path: Path, payload: dict | None = None, raw: str | None = None) -> Path:
    """Write a JSON artifact file and return its path."""
    p = tmp_path / "png_v1.json"
    if raw is not None:
        p.write_text(raw)
    else:
        p.write_text(json.dumps(payload or _VALID_ARTIFACT))
    return p


# ---------------------------------------------------------------------------
# PngModel.from_json tests (direct, via _artifact classmethod)
# ---------------------------------------------------------------------------


class TestPngModelFromJson:
    def test_missing_file_returns_load_failed(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        result = PngModelDirect.from_json(missing)
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_corrupt_json_returns_load_failed(self, tmp_path: Path):
        p = _write_artifact(tmp_path, raw="{not valid json")
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_version_mismatch_returns_load_failed(self, tmp_path: Path):
        bad = dict(_VALID_ARTIFACT, model_version=99)
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"

    def test_missing_required_field_returns_parse_error(self, tmp_path: Path):
        # Drop a required field — should produce parse_error, not crash.
        bad = {k: v for k, v in _VALID_ARTIFACT.items() if k != "training_corpus_sha256"}
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_happy_path_returns_loaded(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        model = result.model
        assert isinstance(model, PngModelDirect)
        assert model.model_version == 2
        assert model.format == "png"
        assert model.knot_log10_unique_colors == pytest.approx(3.3)
        assert model.knot_q50 == pytest.approx(50.0)
        assert model.knot_q70 == pytest.approx(70.0)
        assert model.git_sha == "cafebabe"
        assert "sklearn_version" in model.fit_environment

    def test_happy_path_features_preserved(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        assert result.model.features == _VALID_ARTIFACT["features"]
        assert result.model.supported_modes == _VALID_ARTIFACT["supported_modes"]

    def test_happy_path_training_envelope_forensic(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        # training_envelope is present but is forensic only
        assert "log10_unique_colors" in result.model.training_envelope

    def test_model_is_frozen(self, tmp_path: Path):
        p = _write_artifact(tmp_path)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, Loaded)
        with pytest.raises((AttributeError, TypeError)):
            result.model.format = "jpeg"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Schema validation negative tests (#3 — coefficients shape / bounds)
# ---------------------------------------------------------------------------


class TestPngModelSchemaValidation:
    """Negative tests for _validate_schema — structural and numeric bounds."""

    def test_from_json_rejects_unknown_coefficient_key(self, tmp_path: Path):
        """coefficients shape with 'coef' instead of 'betas' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "coef": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],  # wrong key
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_missing_intercept(self, tmp_path: Path):
        """coefficients dict without 'intercept' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            # 'intercept' intentionally missing
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_mismatched_betas_length(self, tmp_path: Path):
        """betas list with wrong length must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2],  # too short — features has 7 entries
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_non_finite_coefficients(self, tmp_path: Path):
        """Non-finite betas (inf/nan) must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [float("inf"), -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_beta_out_of_bounds(self, tmp_path: Path):
        """Beta value exceeding ±100 must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [999.0, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],  # 999 > 100
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_missing_knot_beta(self, tmp_path: Path):
        """coefficients dict without 'knot_beta' must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            # 'knot_beta' intentionally missing
            "knot_q50_beta": -0.01,
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_oob_knot_beta(self, tmp_path: Path):
        """knot_q50_beta value exceeding ±100 must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": 1e10,  # way out of bounds
            "knot_q70_beta": 0.02,
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_from_json_rejects_non_finite_knot_beta(self, tmp_path: Path):
        """Non-finite knot_q70_beta (inf) must return LoadFailed('parse_error')."""
        bad = dict(_VALID_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.5,
            "betas": [0.1, -0.2, 0.05, 0.3, -0.01, 0.15, -0.05],
            "knot_beta": 0.3,
            "knot_q50_beta": -0.01,
            "knot_q70_beta": float("inf"),
        }
        p = _write_artifact(tmp_path, payload=bad)
        result = PngModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"


# ---------------------------------------------------------------------------
# load_png_model() wrapper tests (goes through estimation/models/__init__.py)
# ---------------------------------------------------------------------------


class TestLoadPngModel:
    def _clear_cache(self):
        """Clear the lru_cache between tests to ensure isolation."""
        load_png_model.cache_clear()

    def test_does_not_raise_on_missing_artifact(self):
        """load_png_model() should return LoadFailed, not raise, when artifact is absent."""
        self._clear_cache()
        # The real png_v1.json is committed in this branch; this test loads it via the same
        # code path production uses.  Whether the file is present (Loaded) or absent
        # (LoadFailed), the key contract is: it never raises.
        result = load_png_model()
        assert isinstance(result, (Loaded, LoadFailed))

    def test_does_not_raise_when_called_multiple_times(self):
        """Calling load_png_model() multiple times must not raise."""
        self._clear_cache()
        for _ in range(5):
            result = load_png_model()
            assert isinstance(result, (Loaded, LoadFailed))

    def test_returns_load_failed_when_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Patch the models dir to a tmp_path without png_v1.json → LoadFailed."""
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        # Also need to clear cache again after the patch since the lru_cache already ran.
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_returns_loaded_on_valid_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With a valid artifact in tmp_path → Loaded."""
        _write_artifact(tmp_path)
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, Loaded)
        assert result.model.format == "png"

    def test_returns_load_failed_on_corrupt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Corrupt JSON → LoadFailed("parse_error") without raising."""
        (tmp_path / "png_v1.json").write_text("{bad json")
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_returns_load_failed_on_version_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """model_version=99 → LoadFailed("version_mismatch") without raising."""
        bad = dict(_VALID_ARTIFACT, model_version=99)
        (tmp_path / "png_v1.json").write_text(json.dumps(bad))
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"


# ===========================================================================
# PngHeaderModel tests
# ===========================================================================

_VALID_HEADER_ARTIFACT: dict = {
    "model_version": 1,
    "format": "png_header",
    "features": ["has_alpha", "quality", "log10_orig_pixels", "input_bpp"],
    "scaler": {
        "mean": [0.3, 65.0, 5.5, 9.0],
        "scale": [0.45, 18.0, 0.7, 5.0],
    },
    "coefficients": {
        "intercept": 0.8,
        "betas": [0.1, -0.05, -0.2, 0.3],
        "knot_q50_beta": -0.02,
        "knot_q70_beta": 0.01,
    },
    "knot_q50": 50.0,
    "knot_q70": 70.0,
    "training_envelope": {
        "has_alpha": [0.0, 1.0],
        "quality": [40.0, 85.0],
    },
    "training_corpus_sha256": "deadbeef1234",
    "git_sha": "abc1234",
    "fit_environment": {
        "numpy_version": "2.0.2",
        "scipy_version": "1.14.0",
    },
    "created_at": "2026-05-07T00:00:00+00:00",
}


def _write_header_artifact(
    tmp_path: Path, payload: dict | None = None, raw: str | None = None
) -> Path:
    p = tmp_path / "png_header_v1.json"
    if raw is not None:
        p.write_text(raw)
    else:
        p.write_text(json.dumps(payload or _VALID_HEADER_ARTIFACT))
    return p


class TestPngHeaderModelFromJson:
    def test_missing_file_returns_load_failed(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        result = PngHeaderModelDirect.from_json(missing)
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_corrupt_json_returns_load_failed(self, tmp_path: Path):
        p = _write_header_artifact(tmp_path, raw="{not valid json")
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_version_mismatch_returns_load_failed(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT, model_version=99)
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"

    def test_wrong_format_string_returns_parse_error(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT, format="png")  # must be "png_header"
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_wrong_features_returns_parse_error(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT, features=["has_alpha", "quality"])
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_missing_required_field_returns_parse_error(self, tmp_path: Path):
        bad = {k: v for k, v in _VALID_HEADER_ARTIFACT.items() if k != "training_corpus_sha256"}
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_happy_path_returns_loaded_header(self, tmp_path: Path):
        p = _write_header_artifact(tmp_path)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedHeader)
        model = result.model
        assert isinstance(model, PngHeaderModelDirect)
        assert model.model_version == 1
        assert model.format == "png_header"
        assert model.knot_q50 == pytest.approx(50.0)
        assert model.knot_q70 == pytest.approx(70.0)
        assert model.git_sha == "abc1234"
        assert "numpy_version" in model.fit_environment

    def test_happy_path_features_preserved(self, tmp_path: Path):
        p = _write_header_artifact(tmp_path)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedHeader)
        assert result.model.features == _VALID_HEADER_ARTIFACT["features"]

    def test_model_is_frozen(self, tmp_path: Path):
        p = _write_header_artifact(tmp_path)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedHeader)
        with pytest.raises((AttributeError, TypeError)):
            result.model.format = "jpeg"  # type: ignore[misc]

    def test_rejects_oob_intercept(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["intercept"] = 9999.0  # > 1000
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_oob_betas(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [999.0, 0.0, 0.0, 0.0]  # 999 > 100
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_non_finite_betas(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [float("nan"), 0.0, 0.0, 0.0]
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_mismatched_betas_length(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [0.1, 0.2]  # too short (need 4)
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_missing_knot_q50_beta(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = {
            "intercept": 0.8,
            "betas": [0.1, -0.05, -0.2, 0.3],
            # knot_q50_beta intentionally missing
            "knot_q70_beta": 0.01,
        }
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_oob_knot_beta(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["knot_q50_beta"] = 1e9  # way out of bounds
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_non_finite_scaler(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["scaler"] = {"mean": [float("inf"), 65.0, 5.5, 9.0], "scale": [0.45, 18.0, 0.7, 5.0]}
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_wrong_scaler_length(self, tmp_path: Path):
        bad = dict(_VALID_HEADER_ARTIFACT)
        bad["scaler"] = {"mean": [0.3, 65.0], "scale": [0.45, 18.0]}  # too short
        p = _write_header_artifact(tmp_path, payload=bad)
        result = PngHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"


class TestLoadPngHeaderModel:
    def _clear_cache(self):
        load_png_header_model.cache_clear()

    def test_does_not_raise_on_missing_artifact(self):
        """load_png_header_model() should return LoadFailed, not raise, when artifact absent."""
        self._clear_cache()
        result = load_png_header_model()
        assert isinstance(result, (LoadedHeader, LoadFailed))

    def test_does_not_raise_when_called_multiple_times(self):
        self._clear_cache()
        for _ in range(5):
            result = load_png_header_model()
            assert isinstance(result, (LoadedHeader, LoadFailed))

    def test_returns_loaded_header_on_valid_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With a valid header artifact in tmp_path → LoadedHeader."""
        _write_header_artifact(tmp_path)
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_header_model.cache_clear()

        result = load_png_header_model()
        assert isinstance(result, LoadedHeader)
        assert result.model.format == "png_header"

    def test_returns_load_failed_on_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Empty tmp_path has no png_header_v1.json → LoadFailed."""
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_header_model.cache_clear()

        result = load_png_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_returns_load_failed_on_corrupt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "png_header_v1.json").write_text("{bad json")
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_header_model.cache_clear()

        result = load_png_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_returns_load_failed_on_version_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bad = dict(_VALID_HEADER_ARTIFACT, model_version=99)
        (tmp_path / "png_header_v1.json").write_text(json.dumps(bad))
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_header_model.cache_clear()

        result = load_png_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"


# ===========================================================================
# JpegHeaderModel tests
# ===========================================================================

_JPEG_HEADER_FEATURES = [
    "target_quality",
    "source_quality",
    "nse",
    "subsampling_444",
    "subsampling_422",
    "subsampling_420",
    "progressive",
    "log10_orig_pixels",
    "input_bpp",
    "mean_dqt_luma",
    "std_dqt_luma",
    "mean_dqt_chroma",
    "std_dqt_chroma",
]

_VALID_JPEG_HEADER_ARTIFACT: dict = {
    "model_version": 1,
    "format": "jpeg_header",
    "features": _JPEG_HEADER_FEATURES,
    "scaler": {
        "mean": [62.5, 75.0, 0.98, 0.4, 0.2, 0.3, 0.1, 5.8, 3.5, 42.0, 20.0, 55.0, 25.0],
        "scale": [18.0, 15.0, 0.05, 0.49, 0.4, 0.46, 0.3, 0.6, 2.5, 18.0, 12.0, 22.0, 14.0],
    },
    "coefficients": {
        "intercept": 0.6,
        "betas": [
            -0.05,
            -0.03,
            0.02,
            0.01,
            -0.01,
            0.00,
            0.00,
            -0.10,
            0.08,
            0.02,
            0.01,
            0.01,
            0.00,
        ],
    },
    "training_envelope": {
        "target_quality": [40.0, 85.0],
        "source_quality": [1.0, 100.0],
    },
    "training_corpus_sha256": "abcdef1234567890",
    "git_sha": "deadbeef",
    "fit_environment": {
        "numpy_version": "2.0.2",
        "scipy_version": "1.14.0",
    },
    "created_at": "2026-05-07T00:00:00+00:00",
}


def _write_jpeg_header_artifact(
    tmp_path: Path, payload: dict | None = None, raw: str | None = None
) -> Path:
    p = tmp_path / "jpeg_header_v1.json"
    if raw is not None:
        p.write_text(raw)
    else:
        p.write_text(json.dumps(payload or _VALID_JPEG_HEADER_ARTIFACT))
    return p


class TestJpegHeaderModelFromJson:
    def test_missing_file_returns_load_failed(self, tmp_path: Path):
        missing = tmp_path / "does_not_exist.json"
        result = JpegHeaderModelDirect.from_json(missing)
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_corrupt_json_returns_load_failed(self, tmp_path: Path):
        p = _write_jpeg_header_artifact(tmp_path, raw="{not valid json")
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_version_mismatch_returns_load_failed(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT, model_version=99)
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"

    def test_wrong_format_string_returns_parse_error(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT, format="jpeg")  # must be "jpeg_header"
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_wrong_features_returns_parse_error(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT, features=["target_quality", "source_quality"])
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_missing_required_field_returns_parse_error(self, tmp_path: Path):
        bad = {
            k: v for k, v in _VALID_JPEG_HEADER_ARTIFACT.items() if k != "training_corpus_sha256"
        }
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_happy_path_returns_loaded_jpeg(self, tmp_path: Path):
        p = _write_jpeg_header_artifact(tmp_path)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedJpeg)
        model = result.model
        assert isinstance(model, JpegHeaderModelDirect)
        assert model.model_version == 1
        assert model.format == "jpeg_header"
        assert model.git_sha == "deadbeef"
        assert "numpy_version" in model.fit_environment

    def test_happy_path_features_preserved(self, tmp_path: Path):
        p = _write_jpeg_header_artifact(tmp_path)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedJpeg)
        assert result.model.features == _JPEG_HEADER_FEATURES

    def test_model_is_frozen(self, tmp_path: Path):
        p = _write_jpeg_header_artifact(tmp_path)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadedJpeg)
        with pytest.raises((AttributeError, TypeError)):
            result.model.format = "jpeg"  # type: ignore[misc]

    def test_rejects_oob_intercept(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_JPEG_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["intercept"] = 9999.0  # > 1000
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_oob_betas(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_JPEG_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [999.0] + [0.0] * 12  # first beta > 100
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_non_finite_betas(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_JPEG_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [float("nan")] + [0.0] * 12
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_mismatched_betas_length(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["coefficients"] = dict(_VALID_JPEG_HEADER_ARTIFACT["coefficients"])
        bad["coefficients"]["betas"] = [0.1, 0.2]  # too short (need 13)
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_missing_intercept(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["coefficients"] = {"betas": [0.0] * 13}  # intercept missing
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_non_finite_scaler(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["scaler"] = {
            "mean": [float("inf")] + [0.0] * 12,
            "scale": [1.0] * 13,
        }
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_rejects_wrong_scaler_length(self, tmp_path: Path):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT)
        bad["scaler"] = {"mean": [0.3, 65.0], "scale": [0.45, 18.0]}  # too short (need 13)
        p = _write_jpeg_header_artifact(tmp_path, payload=bad)
        result = JpegHeaderModelDirect.from_json(p)
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"


class TestLoadJpegHeaderModel:
    def _clear_cache(self):
        load_jpeg_header_model.cache_clear()

    def test_does_not_raise_on_missing_artifact(self):
        """load_jpeg_header_model() should return LoadFailed, not raise, when artifact absent."""
        self._clear_cache()
        result = load_jpeg_header_model()
        assert isinstance(result, (LoadedJpeg, LoadFailed))

    def test_does_not_raise_when_called_multiple_times(self):
        self._clear_cache()
        for _ in range(5):
            result = load_jpeg_header_model()
            assert isinstance(result, (LoadedJpeg, LoadFailed))

    def test_returns_loaded_jpeg_on_valid_artifact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """With a valid JPEG header artifact in tmp_path → LoadedJpeg."""
        _write_jpeg_header_artifact(tmp_path)
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_jpeg_header_model.cache_clear()

        result = load_jpeg_header_model()
        assert isinstance(result, LoadedJpeg)
        assert result.model.format == "jpeg_header"

    def test_returns_load_failed_on_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Empty tmp_path has no jpeg_header_v1.json → LoadFailed."""
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_jpeg_header_model.cache_clear()

        result = load_jpeg_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "missing"

    def test_returns_load_failed_on_corrupt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        (tmp_path / "jpeg_header_v1.json").write_text("{bad json")
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_jpeg_header_model.cache_clear()

        result = load_jpeg_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "parse_error"

    def test_returns_load_failed_on_version_mismatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bad = dict(_VALID_JPEG_HEADER_ARTIFACT, model_version=99)
        (tmp_path / "jpeg_header_v1.json").write_text(json.dumps(bad))
        self._clear_cache()
        import estimation.models as models_mod

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_jpeg_header_model.cache_clear()

        result = load_jpeg_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "version_mismatch"


# ---------------------------------------------------------------------------
# Defensive "other" exception handlers in the three loaders
# ---------------------------------------------------------------------------


class TestLoaderDefensiveExceptionHandler:
    """Cover the except Exception: → LoadFailed('other') path in each loader.

    These handlers exist as belt-and-suspenders: ``from_json`` is designed
    never to raise, but the loaders guard against future bugs.  We force a
    raise by patching ``from_json`` directly.
    """

    def test_load_png_model_defensive_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If PngModel.from_json raises unexpectedly, loader returns LoadFailed('other')."""
        import estimation.models as models_mod
        from estimation.models._artifact import PngModel

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_model.cache_clear()

        # Monkeypatch from_json to raise (circumvents the "never raises" contract)
        def _explode(path):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(PngModel, "from_json", staticmethod(_explode))

        result = load_png_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "other"

        load_png_model.cache_clear()

    def test_load_png_header_model_defensive_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If PngHeaderModel.from_json raises unexpectedly, loader returns LoadFailed('other')."""
        import estimation.models as models_mod
        from estimation.models._artifact import PngHeaderModel

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_png_header_model.cache_clear()

        def _explode(path):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(PngHeaderModel, "from_json", staticmethod(_explode))

        result = load_png_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "other"

        load_png_header_model.cache_clear()

    def test_load_jpeg_header_model_defensive_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If JpegHeaderModel.from_json raises unexpectedly, loader returns LoadFailed('other')."""
        import estimation.models as models_mod
        from estimation.models._artifact import JpegHeaderModel

        monkeypatch.setattr(models_mod, "_MODELS_DIR", tmp_path)
        load_jpeg_header_model.cache_clear()

        def _explode(path):
            raise RuntimeError("unexpected failure")

        monkeypatch.setattr(JpegHeaderModel, "from_json", staticmethod(_explode))

        result = load_jpeg_header_model()
        assert isinstance(result, LoadFailed)
        assert result.reason == "other"

        load_jpeg_header_model.cache_clear()
