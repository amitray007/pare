"""Versioned model artifact schema for the fitted BPP estimator.

Mirrors the ``bench/corpus/manifest.py`` precedent: ``@dataclass(frozen=True, slots=True)``
with a ``from_json()`` classmethod.  No Pydantic — frozen dataclasses have ~10× lower
instantiation overhead for a once-per-process load.

``optimizer_quality_logic_sha256`` is intentionally omitted from this MVP (Phase 1 scope cut).
It will be added in Phase 2 once ``bench/fit/png.py`` is implemented.

``Loaded``, ``LoadedHeader``, ``LoadedJpeg``, and ``LoadFailed`` live here (not in
``__init__.py``) so that model ``from_json`` methods can return them without a circular
import.  ``estimation/models/__init__.py`` re-exports them.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("pare.estimation.models")

# The only model_version this code understands.  Artifacts with a different version are
# rejected at load time so stale JSON from a previous fit does not silently corrupt predictions.
_SUPPORTED_MODEL_VERSION = 2

# Supported version for the header-only PNG model (separate artifact, separate version counter).
_SUPPORTED_PNG_HEADER_MODEL_VERSION = 1


@dataclass(frozen=True, slots=True)
class Loaded:
    """Successful model load result."""

    model: PngModel


@dataclass(frozen=True, slots=True)
class LoadFailed:
    """Failed model load result.

    ``reason`` values:

    - ``"missing"``          — artifact file not found.
    - ``"parse_error"``      — JSON is malformed or does not match the expected schema.
    - ``"version_mismatch"`` — ``model_version`` field differs from the supported version.
    - ``"other"``            — any other unexpected exception during load.
    """

    reason: Literal["missing", "parse_error", "version_mismatch", "other"]


@dataclass(frozen=True, slots=True)
class PngModel:
    """Fitted BPP curve artifact for PNG estimation.

    Fields
    ------
    model_version : int
        Schema version.  Must equal ``_SUPPORTED_MODEL_VERSION`` (currently 2);
        any other value → ``LoadFailed("version_mismatch")``.
    format : str
        Always ``"png"`` for this class.
    features : list[str]
        Ordered feature names in the order the ``coefficients`` vector expects them.
    supported_modes : list[str]
        Pillow image modes the model was trained on.  Modes outside this list → fallback.
    scaler : dict[str, list[float]]
        ``StandardScaler`` params: ``{"mean": [...], "scale": [...]}``.
    coefficients : dict[str, Any]
        Regression coefficients.  Shape is implementation-defined by ``bench/fit/png.py``.
    knot_log10_unique_colors : float
        Piecewise-linear knot position on the ``log10_unique_colors`` axis (~3.3 = 2000 colors).
    training_envelope : dict[str, list[float]]
        **Forensic only** — documents the corpus envelope used during training.
        Not used at runtime; routing clips are hardcoded in ``png_features.py``.
    training_corpus_sha256 : str
        SHA-256 of the training corpus manifest file.
    git_sha : str
        Git commit SHA at fit time.
    fit_environment : dict[str, Any]
        ``{"sklearn_version": "...", "numpy_version": "...", "openblas_threads": 1}``.
    created_at : str
        ISO-8601 timestamp of the fit run.
    """

    model_version: int
    format: str
    features: list[str]
    supported_modes: list[str]
    scaler: dict[str, list[float]]
    coefficients: dict[str, Any]
    knot_log10_unique_colors: float
    knot_q50: float
    knot_q70: float
    training_envelope: dict[str, list[float]]
    training_corpus_sha256: str
    git_sha: str
    fit_environment: dict[str, Any]
    created_at: str

    @classmethod
    def _validate_schema(cls, raw: dict[str, Any], features: list[str]) -> str | None:
        """Validate scaler/coefficients shapes and numeric bounds.

        Returns ``None`` on success, or a human-readable reason string on failure.
        Logs the reason at DEBUG level so test failures are debuggable.
        """
        # --- scaler shape ---
        scaler = raw.get("scaler")
        if not isinstance(scaler, dict) or "mean" not in scaler or "scale" not in scaler:
            reason = "scaler must be a dict with keys 'mean' and 'scale'"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        if not isinstance(scaler["mean"], list) or not isinstance(scaler["scale"], list):
            reason = "scaler['mean'] and scaler['scale'] must be lists"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        if len(scaler["mean"]) != len(features) or len(scaler["scale"]) != len(features):
            reason = (
                f"scaler mean/scale length {len(scaler['mean'])}/{len(scaler['scale'])} "
                f"!= features length {len(features)}"
            )
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        for val in scaler["mean"] + scaler["scale"]:
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                reason = f"non-finite value in scaler: {val!r}"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason

        # --- coefficients shape ---
        coefficients = raw.get("coefficients")
        if not isinstance(coefficients, dict):
            reason = "coefficients must be a dict"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        if "betas" not in coefficients:
            reason = "coefficients must have key 'betas'"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        if "intercept" not in coefficients:
            reason = "coefficients must have key 'intercept'"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason

        betas = coefficients["betas"]
        intercept = coefficients["intercept"]
        if not isinstance(betas, list):
            reason = "coefficients['betas'] must be a list"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        # betas has one entry per feature; knot terms are stored separately
        expected_betas_len = len(features)
        if len(betas) != expected_betas_len:
            reason = (
                f"coefficients['betas'] length {len(betas)} != "
                f"len(features) = {expected_betas_len}"
            )
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        if not isinstance(intercept, (int, float)) or not math.isfinite(float(intercept)):
            reason = f"coefficients['intercept'] is not finite: {intercept!r}"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason

        # --- numeric bounds (security: prevent adversarial artifacts) ---
        if abs(float(intercept)) > 1000.0:
            reason = f"intercept {intercept} exceeds ±1000 bound"
            logger.debug("PngModel schema validation failed: %s", reason)
            return reason
        for i, b in enumerate(betas):
            if not isinstance(b, (int, float)) or not math.isfinite(float(b)):
                reason = f"non-finite beta[{i}]: {b!r}"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason
            if abs(float(b)) > 100.0:
                reason = f"beta[{i}] = {b} exceeds ±100 bound"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason

        # --- knot beta bounds (security: prevent adversarial artifacts) ---
        for knot_key in ("knot_beta", "knot_q50_beta", "knot_q70_beta"):
            if knot_key not in coefficients:
                reason = f"coefficients missing required key '{knot_key}'"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason
            knot_val = coefficients[knot_key]
            if not isinstance(knot_val, (int, float)) or not math.isfinite(float(knot_val)):
                reason = f"coefficients['{knot_key}'] is not finite: {knot_val!r}"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason
            if abs(float(knot_val)) > 100.0:
                reason = f"coefficients['{knot_key}'] = {knot_val} exceeds ±100 bound"
                logger.debug("PngModel schema validation failed: %s", reason)
                return reason

        return None

    @classmethod
    def from_json(cls, path: Path) -> Loaded | LoadFailed:
        """Load and validate a ``PngModel`` from a JSON file at *path*.

        Returns ``Loaded(model)`` on success.  Returns ``LoadFailed(reason)`` on any of:

        - ``path`` does not exist → ``"missing"``
        - JSON parse error or schema mismatch → ``"parse_error"``
        - ``model_version != _SUPPORTED_MODEL_VERSION`` (currently 2) → ``"version_mismatch"``
        - Any other unexpected exception → ``"other"``

        Never raises.
        """
        if not path.exists():
            logger.warning("PngModel artifact not found: %s", path)
            return LoadFailed(reason="missing")

        try:
            raw: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("PngModel artifact parse error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        try:
            version = raw["model_version"]
        except (KeyError, TypeError) as exc:
            logger.warning("PngModel artifact missing model_version (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        if version != _SUPPORTED_MODEL_VERSION:
            logger.warning(
                "PngModel artifact version mismatch: expected %d, got %r (%s)",
                _SUPPORTED_MODEL_VERSION,
                version,
                path,
            )
            return LoadFailed(reason="version_mismatch")

        try:
            model = cls(
                model_version=int(raw["model_version"]),
                format=str(raw["format"]),
                features=list(raw["features"]),
                supported_modes=list(raw["supported_modes"]),
                scaler=dict(raw["scaler"]),
                coefficients=dict(raw["coefficients"]),
                knot_log10_unique_colors=float(raw["knot_log10_unique_colors"]),
                knot_q50=float(raw["knot_q50"]),
                knot_q70=float(raw["knot_q70"]),
                training_envelope=dict(raw.get("training_envelope") or {}),
                training_corpus_sha256=str(raw["training_corpus_sha256"]),
                git_sha=str(raw["git_sha"]),
                fit_environment=dict(raw["fit_environment"]),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("PngModel artifact schema error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        # Validate scaler/coefficients shapes and numeric bounds after construction
        validation_error = cls._validate_schema(raw, model.features)
        if validation_error is not None:
            logger.warning("PngModel artifact validation failed (%s): %s", path, validation_error)
            return LoadFailed(reason="parse_error")

        return Loaded(model=model)


# ---------------------------------------------------------------------------
# PngHeaderModel — header-only PNG estimator (Phase 1a)
# ---------------------------------------------------------------------------

# Expected feature list — validated exactly at load time.
_PNG_HEADER_FEATURES = ["has_alpha", "quality", "log10_orig_pixels", "input_bpp"]


@dataclass(frozen=True, slots=True)
class LoadedHeader:
    """Successful header-model load result."""

    model: "PngHeaderModel"


@dataclass(frozen=True, slots=True)
class PngHeaderModel:
    """Header-only PNG model: 4 features, no thumbnail decode required.

    Distinct from ``PngModel`` which uses thumbnail-derived features.

    Fields
    ------
    model_version : int
        Schema version.  Must equal ``_SUPPORTED_PNG_HEADER_MODEL_VERSION`` (1).
    format : str
        Always ``"png_header"``.
    features : list[str]
        Exactly ``["has_alpha", "quality", "log10_orig_pixels", "input_bpp"]``.
    scaler : dict[str, list[float]]
        ``StandardScaler`` params: ``{"mean": [...], "scale": [...]}``.
        Both lists have 4 elements, one per feature.
    coefficients : dict[str, Any]
        Regression coefficients:
        ``{"intercept": float, "betas": list[float],
           "knot_q50_beta": float, "knot_q70_beta": float}``.
        No ``knot_beta`` (no ``log10_unique_colors`` knot in this model).
    knot_q50 : float
        Quality knot at q=50.  Stored for reproducibility.
    knot_q70 : float
        Quality knot at q=70.  Stored for reproducibility.
    training_envelope : dict[str, list[float]]
        Forensic min/max per feature.  Not used at runtime.
    training_corpus_sha256 : str
        SHA-256 of the training corpus manifest file.
    git_sha : str
        Git commit SHA at fit time.
    fit_environment : dict[str, str]
        Build environment metadata (``numpy_version``, ``scipy_version``).
    created_at : str
        ISO-8601 timestamp of the fit run.
    """

    model_version: int
    format: str
    features: list[str]
    scaler: dict[str, list[float]]
    coefficients: dict[str, Any]
    knot_q50: float
    knot_q70: float
    training_envelope: dict[str, list[float]]
    training_corpus_sha256: str
    git_sha: str
    fit_environment: dict[str, str]
    created_at: str

    @classmethod
    def _validate_schema(cls, raw: dict[str, Any], features: list[str]) -> str | None:
        """Validate scaler/coefficients shapes and numeric bounds.

        Returns ``None`` on success, or a human-readable reason string on failure.
        """
        n_features = len(features)

        # --- scaler shape ---
        scaler = raw.get("scaler")
        if not isinstance(scaler, dict) or "mean" not in scaler or "scale" not in scaler:
            reason = "scaler must be a dict with keys 'mean' and 'scale'"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        if not isinstance(scaler["mean"], list) or not isinstance(scaler["scale"], list):
            reason = "scaler['mean'] and scaler['scale'] must be lists"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        if len(scaler["mean"]) != n_features or len(scaler["scale"]) != n_features:
            reason = (
                f"scaler mean/scale length {len(scaler['mean'])}/{len(scaler['scale'])} "
                f"!= features length {n_features}"
            )
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        for val in scaler["mean"] + scaler["scale"]:
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                reason = f"non-finite value in scaler: {val!r}"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason

        # --- coefficients shape ---
        coefficients = raw.get("coefficients")
        if not isinstance(coefficients, dict):
            reason = "coefficients must be a dict"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        for required_key in ("intercept", "betas", "knot_q50_beta", "knot_q70_beta"):
            if required_key not in coefficients:
                reason = f"coefficients missing required key '{required_key}'"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason

        betas = coefficients["betas"]
        intercept = coefficients["intercept"]

        if not isinstance(betas, list):
            reason = "coefficients['betas'] must be a list"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        if len(betas) != n_features:
            reason = f"coefficients['betas'] length {len(betas)} != features length {n_features}"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        if not isinstance(intercept, (int, float)) or not math.isfinite(float(intercept)):
            reason = f"coefficients['intercept'] is not finite: {intercept!r}"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason

        # --- numeric bounds (security: prevent adversarial artifacts) ---
        if abs(float(intercept)) > 1000.0:
            reason = f"intercept {intercept} exceeds ±1000 bound"
            logger.debug("PngHeaderModel schema validation failed: %s", reason)
            return reason
        for i, b in enumerate(betas):
            if not isinstance(b, (int, float)) or not math.isfinite(float(b)):
                reason = f"non-finite beta[{i}]: {b!r}"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason
            if abs(float(b)) > 100.0:
                reason = f"beta[{i}] = {b} exceeds ±100 bound"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason

        # --- knot beta bounds ---
        for knot_key in ("knot_q50_beta", "knot_q70_beta"):
            knot_val = coefficients[knot_key]
            if not isinstance(knot_val, (int, float)) or not math.isfinite(float(knot_val)):
                reason = f"coefficients['{knot_key}'] is not finite: {knot_val!r}"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason
            if abs(float(knot_val)) > 100.0:
                reason = f"coefficients['{knot_key}'] = {knot_val} exceeds ±100 bound"
                logger.debug("PngHeaderModel schema validation failed: %s", reason)
                return reason

        return None

    @classmethod
    def from_json(cls, path: Path) -> "LoadedHeader | LoadFailed":
        """Load and validate a ``PngHeaderModel`` from a JSON file at *path*.

        Returns ``LoadedHeader(model)`` on success.  Returns ``LoadFailed(reason)`` on:

        - ``path`` does not exist → ``"missing"``
        - JSON parse error or schema mismatch → ``"parse_error"``
        - ``model_version != _SUPPORTED_PNG_HEADER_MODEL_VERSION`` → ``"version_mismatch"``
        - Any other unexpected exception → ``"other"``

        Never raises.
        """
        if not path.exists():
            logger.warning("PngHeaderModel artifact not found: %s", path)
            return LoadFailed(reason="missing")

        try:
            raw: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("PngHeaderModel artifact parse error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        try:
            version = raw["model_version"]
        except (KeyError, TypeError) as exc:
            logger.warning("PngHeaderModel artifact missing model_version (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        if version != _SUPPORTED_PNG_HEADER_MODEL_VERSION:
            logger.warning(
                "PngHeaderModel artifact version mismatch: expected %d, got %r (%s)",
                _SUPPORTED_PNG_HEADER_MODEL_VERSION,
                version,
                path,
            )
            return LoadFailed(reason="version_mismatch")

        try:
            fmt = str(raw["format"])
            if fmt != "png_header":
                logger.warning(
                    "PngHeaderModel artifact format mismatch: expected 'png_header', got %r (%s)",
                    fmt,
                    path,
                )
                return LoadFailed(reason="parse_error")

            features = list(raw["features"])
            if features != _PNG_HEADER_FEATURES:
                logger.warning(
                    "PngHeaderModel features mismatch: expected %r, got %r (%s)",
                    _PNG_HEADER_FEATURES,
                    features,
                    path,
                )
                return LoadFailed(reason="parse_error")

            model = cls(
                model_version=int(raw["model_version"]),
                format=fmt,
                features=features,
                scaler=dict(raw["scaler"]),
                coefficients=dict(raw["coefficients"]),
                knot_q50=float(raw["knot_q50"]),
                knot_q70=float(raw["knot_q70"]),
                training_envelope=dict(raw.get("training_envelope") or {}),
                training_corpus_sha256=str(raw["training_corpus_sha256"]),
                git_sha=str(raw["git_sha"]),
                fit_environment=dict(raw["fit_environment"]),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("PngHeaderModel artifact schema error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        # Validate scaler/coefficients shapes and numeric bounds
        validation_error = cls._validate_schema(raw, model.features)
        if validation_error is not None:
            logger.warning(
                "PngHeaderModel artifact validation failed (%s): %s", path, validation_error
            )
            return LoadFailed(reason="parse_error")

        return LoadedHeader(model=model)


# ---------------------------------------------------------------------------
# JpegHeaderModel — header-only JPEG estimator (Phase 1b)
# ---------------------------------------------------------------------------

# Supported version for the header-only JPEG model.
_SUPPORTED_JPEG_HEADER_MODEL_VERSION = 1

# Expected feature list — validated exactly at load time.
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


@dataclass(frozen=True, slots=True)
class LoadedJpeg:
    """Successful JPEG header-model load result."""

    model: "JpegHeaderModel"


@dataclass(frozen=True, slots=True)
class JpegHeaderModel:
    """Header-only JPEG model. Decode-free inference from header parsing
    + LSM source-Q estimation.

    Fields
    ------
    model_version : int
        Schema version.  Must equal ``_SUPPORTED_JPEG_HEADER_MODEL_VERSION`` (1).
    format : str
        Always ``"jpeg_header"``.
    features : list[str]
        Exactly the 13 canonical feature names defined in ``_JPEG_HEADER_FEATURES``.
    scaler : dict[str, list[float]]
        ``StandardScaler`` params: ``{"mean": [...], "scale": [...]}``.
        Both lists have 13 elements, one per feature.
    coefficients : dict[str, Any]
        Regression coefficients:
        ``{"intercept": float, "betas": list[float]}``.
        ``betas`` has 13 elements (one per feature).
    training_envelope : dict[str, list[float]]
        Forensic min/max per feature.  Not used at runtime.
    training_corpus_sha256 : str
        SHA-256 of the training corpus manifest file.
    git_sha : str
        Git commit SHA at fit time.
    fit_environment : dict[str, str]
        Build environment metadata (``numpy_version``, ``scipy_version``).
    created_at : str
        ISO-8601 timestamp of the fit run.
    """

    model_version: int
    format: str
    features: list[str]
    scaler: dict[str, list[float]]
    coefficients: dict[str, Any]
    training_envelope: dict[str, list[float]]
    training_corpus_sha256: str
    git_sha: str
    fit_environment: dict[str, str]
    created_at: str

    @classmethod
    def _validate_schema(cls, raw: dict[str, Any], features: list[str]) -> str | None:
        """Validate scaler/coefficients shapes and numeric bounds.

        Returns ``None`` on success, or a human-readable reason string on failure.
        """
        n_features = len(features)

        # --- scaler shape ---
        scaler = raw.get("scaler")
        if not isinstance(scaler, dict) or "mean" not in scaler or "scale" not in scaler:
            reason = "scaler must be a dict with keys 'mean' and 'scale'"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        if not isinstance(scaler["mean"], list) or not isinstance(scaler["scale"], list):
            reason = "scaler['mean'] and scaler['scale'] must be lists"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        if len(scaler["mean"]) != n_features or len(scaler["scale"]) != n_features:
            reason = (
                f"scaler mean/scale length {len(scaler['mean'])}/{len(scaler['scale'])} "
                f"!= features length {n_features}"
            )
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        for val in scaler["mean"] + scaler["scale"]:
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                reason = f"non-finite value in scaler: {val!r}"
                logger.debug("JpegHeaderModel schema validation failed: %s", reason)
                return reason

        # --- coefficients shape ---
        coefficients = raw.get("coefficients")
        if not isinstance(coefficients, dict):
            reason = "coefficients must be a dict"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        for required_key in ("intercept", "betas"):
            if required_key not in coefficients:
                reason = f"coefficients missing required key '{required_key}'"
                logger.debug("JpegHeaderModel schema validation failed: %s", reason)
                return reason

        betas = coefficients["betas"]
        intercept = coefficients["intercept"]

        if not isinstance(betas, list):
            reason = "coefficients['betas'] must be a list"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        if len(betas) != n_features:
            reason = f"coefficients['betas'] length {len(betas)} != features length {n_features}"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        if not isinstance(intercept, (int, float)) or not math.isfinite(float(intercept)):
            reason = f"coefficients['intercept'] is not finite: {intercept!r}"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason

        # --- numeric bounds (security: prevent adversarial artifacts) ---
        if abs(float(intercept)) > 1000.0:
            reason = f"intercept {intercept} exceeds ±1000 bound"
            logger.debug("JpegHeaderModel schema validation failed: %s", reason)
            return reason
        for i, b in enumerate(betas):
            if not isinstance(b, (int, float)) or not math.isfinite(float(b)):
                reason = f"non-finite beta[{i}]: {b!r}"
                logger.debug("JpegHeaderModel schema validation failed: %s", reason)
                return reason
            if abs(float(b)) > 100.0:
                reason = f"beta[{i}] = {b} exceeds ±100 bound"
                logger.debug("JpegHeaderModel schema validation failed: %s", reason)
                return reason

        return None

    @classmethod
    def from_json(cls, path: Path) -> "LoadedJpeg | LoadFailed":
        """Load and validate a ``JpegHeaderModel`` from a JSON file at *path*.

        Returns ``LoadedJpeg(model)`` on success.  Returns ``LoadFailed(reason)`` on:

        - ``path`` does not exist → ``"missing"``
        - JSON parse error or schema mismatch → ``"parse_error"``
        - ``model_version != _SUPPORTED_JPEG_HEADER_MODEL_VERSION`` → ``"version_mismatch"``
        - Any other unexpected exception → ``"other"``

        Never raises.
        """
        if not path.exists():
            logger.warning("JpegHeaderModel artifact not found: %s", path)
            return LoadFailed(reason="missing")

        try:
            raw: dict[str, Any] = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            logger.warning("JpegHeaderModel artifact parse error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        try:
            version = raw["model_version"]
        except (KeyError, TypeError) as exc:
            logger.warning("JpegHeaderModel artifact missing model_version (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        if version != _SUPPORTED_JPEG_HEADER_MODEL_VERSION:
            logger.warning(
                "JpegHeaderModel artifact version mismatch: expected %d, got %r (%s)",
                _SUPPORTED_JPEG_HEADER_MODEL_VERSION,
                version,
                path,
            )
            return LoadFailed(reason="version_mismatch")

        try:
            fmt = str(raw["format"])
            if fmt != "jpeg_header":
                logger.warning(
                    "JpegHeaderModel artifact format mismatch: expected 'jpeg_header', got %r (%s)",
                    fmt,
                    path,
                )
                return LoadFailed(reason="parse_error")

            features = list(raw["features"])
            if features != _JPEG_HEADER_FEATURES:
                logger.warning(
                    "JpegHeaderModel features mismatch: expected %r, got %r (%s)",
                    _JPEG_HEADER_FEATURES,
                    features,
                    path,
                )
                return LoadFailed(reason="parse_error")

            model = cls(
                model_version=int(raw["model_version"]),
                format=fmt,
                features=features,
                scaler=dict(raw["scaler"]),
                coefficients=dict(raw["coefficients"]),
                training_envelope=dict(raw.get("training_envelope") or {}),
                training_corpus_sha256=str(raw["training_corpus_sha256"]),
                git_sha=str(raw["git_sha"]),
                fit_environment=dict(raw["fit_environment"]),
                created_at=str(raw["created_at"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("JpegHeaderModel artifact schema error (%s): %s", path, exc)
            return LoadFailed(reason="parse_error")

        # Validate scaler/coefficients shapes and numeric bounds
        validation_error = cls._validate_schema(raw, model.features)
        if validation_error is not None:
            logger.warning(
                "JpegHeaderModel artifact validation failed (%s): %s", path, validation_error
            )
            return LoadFailed(reason="parse_error")

        return LoadedJpeg(model=model)
