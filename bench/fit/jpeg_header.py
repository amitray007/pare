"""Fit the JPEG header-only BPP regression model from the bench corpus.

Usage
-----
    python -m bench.fit.jpeg_header \\
        --manifest full \\
        --seed 42 \\
        --output estimation/models/jpeg_header_v1.json \\
        --n-min 60 \\
        --quality-presets 40,60,75,85

The script:

1. Loads the corpus and iterates every (JPEG entry × quality preset) pair.
2. Parses the JPEG header with ``parse_jpeg_header()``.
   If None or ``fallback_reason is not None``, skips the entry.
3. Runs LSM: ``q_source, nse = estimate_source_quality_lsm(dqt_luma, dqt_chroma)``.
   If ``nse < 0.85``, skips (custom quantization — not modellable).
4. Computes 13 features:
   - ``target_quality``        (preset value)
   - ``source_quality``        (LSM estimate)
   - ``nse``                   (LSM fit quality)
   - ``subsampling_444``       (1 if 4:4:4)
   - ``subsampling_422``       (1 if 4:2:2)
   - ``subsampling_420``       (1 if 4:2:0)
   - ``progressive``           (1 if progressive)
   - ``log10_orig_pixels``     (log10 of width × height)
   - ``input_bpp``             (file_size × 8 / (width × height))
   - ``mean_dqt_luma``         (mean of 64-element luma table)
   - ``std_dqt_luma``          (std of 64-element luma table)
   - ``mean_dqt_chroma``       (mean of chroma table, or 0.0 if grayscale)
   - ``std_dqt_chroma``        (std of chroma table, or 0.0 if grayscale)
5. Runs actual ``optimize_image()`` to compute ``actual_output_bpp``.
6. Accumulates ``(features_dict, actual_output_bpp)`` rows.
7. Asserts ``n >= n_min``.  Fits via ``numpy.linalg.lstsq`` with ``StandardScaler``.
   No knot terms (v1 linear model — add knots in a follow-up if accuracy is poor).
8. Computes training residual stats (median / p95 / max relative error).
9. Writes JSON artifact (plain write).
10. PII assertion on output.
11. Prints summary.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import re
import subprocess
import sys
from math import log10
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import scipy

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("bench.fit.jpeg_header")

import PIL.Image as _PIL_Image  # noqa: E402

_PIL_Image.MAX_IMAGE_PIXELS = 100_000_000

_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Feature names — must match JpegHeaderModel._JPEG_HEADER_FEATURES exactly.
_FEATURE_NAMES = [
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


def _manifest_path(name: str) -> Path:
    if name.endswith(".json"):
        return Path(name)
    return Path(__file__).parent.parent / "corpus" / "manifests" / f"{name}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit JPEG header-only BPP regression model from bench corpus.",
    )
    parser.add_argument("--manifest", default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="estimation/models/jpeg_header_v1.json",
    )
    parser.add_argument("--n-min", type=int, default=60)
    parser.add_argument("--quality-presets", default="40,60,75,85")
    return parser.parse_args()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
                cwd=_PROJECT_ROOT,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _compute_features(
    hdr,
    file_size: int,
    target_quality: int,
    q_source: int,
    nse: float,
) -> dict | None:
    """Compute the 13-feature dict from a parsed JpegHeader + LSM result."""
    width, height = hdr.width, hdr.height
    orig_pixels = width * height
    if orig_pixels == 0:
        return None

    try:
        log10_orig_pixels = log10(orig_pixels)
        input_bpp = (file_size * 8) / orig_pixels
    except (ValueError, ZeroDivisionError):
        return None

    subsampling_444 = 1 if hdr.subsampling == "4:4:4" else 0
    subsampling_422 = 1 if hdr.subsampling == "4:2:2" else 0
    subsampling_420 = 1 if hdr.subsampling == "4:2:0" else 0
    progressive = 1 if hdr.progressive else 0

    luma = hdr.dqt_luma
    mean_dqt_luma = float(mean(luma)) if luma else 0.0
    std_dqt_luma = float(stdev(luma)) if len(luma) > 1 else 0.0

    chroma = hdr.dqt_chroma
    if chroma:
        mean_dqt_chroma = float(mean(chroma))
        std_dqt_chroma = float(stdev(chroma)) if len(chroma) > 1 else 0.0
    else:
        mean_dqt_chroma = 0.0
        std_dqt_chroma = 0.0

    return {
        "target_quality": float(target_quality),
        "source_quality": float(q_source),
        "nse": float(nse),
        "subsampling_444": float(subsampling_444),
        "subsampling_422": float(subsampling_422),
        "subsampling_420": float(subsampling_420),
        "progressive": float(progressive),
        "log10_orig_pixels": log10_orig_pixels,
        "input_bpp": input_bpp,
        "mean_dqt_luma": mean_dqt_luma,
        "std_dqt_luma": std_dqt_luma,
        "mean_dqt_chroma": mean_dqt_chroma,
        "std_dqt_chroma": std_dqt_chroma,
    }


async def _run_cases(
    jpeg_entries: list,
    corpus_root: Path,
    quality_presets: list[int],
    nse_threshold: float = 0.85,
) -> list[dict]:
    """Run optimize + header parse for every (entry × quality) pair."""
    from bench.corpus.builder import file_path
    from estimation.jpeg_header import estimate_source_quality_lsm, parse_jpeg_header
    from optimizers.router import optimize_image
    from schemas import OptimizationConfig

    rows = []
    total = len(jpeg_entries) * len(quality_presets)
    done = 0
    skipped = 0

    for entry in jpeg_entries:
        path = file_path(corpus_root, entry, "jpeg")
        if not path.exists():
            logger.warning("corpus file missing: %s — skipping", path)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        data = path.read_bytes()
        file_size = len(data)

        # Parse JPEG header — pure-bytes, no Pillow
        hdr = parse_jpeg_header(data)
        if hdr is None or hdr.fallback_reason is not None:
            reason = hdr.fallback_reason if hdr is not None else "parse_failed"
            logger.debug("skip %s: %s", entry.name, reason)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        if not hdr.dqt_luma:
            logger.debug("skip %s: no luma DQT", entry.name)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        # LSM source-quality estimation
        q_source, nse = estimate_source_quality_lsm(hdr.dqt_luma, hdr.dqt_chroma)
        if nse < nse_threshold:
            logger.debug(
                "skip %s: NSE=%.3f < %.2f (custom quantization)", entry.name, nse, nse_threshold
            )
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        features_base = _compute_features(hdr, file_size, 0, q_source, nse)
        if features_base is None:
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        for target_quality in quality_presets:
            done += 1
            config = OptimizationConfig(quality=target_quality)

            try:
                result = await optimize_image(data, config)
            except Exception as exc:
                logger.warning(
                    "optimize_image failed for %s q=%d: %s", entry.name, target_quality, exc
                )
                skipped += 1
                continue

            orig_pixels = hdr.width * hdr.height
            actual_bpp = result.optimized_size * 8 / orig_pixels

            row = dict(features_base)
            row["target_quality"] = float(target_quality)
            row["actual_bpp"] = actual_bpp

            rows.append(row)

            if done % 20 == 0:
                print(
                    f"  [{done}/{total}] rows={len(rows)} skipped={skipped}",
                    file=sys.stderr,
                )

    print(
        f"  [{done}/{total}] rows={len(rows)} skipped={skipped}",
        file=sys.stderr,
    )
    return rows


def _fit_linear_model(rows: list[dict]) -> dict:
    """Fit a linear OLS model (no knots for v1) via numpy.linalg.lstsq."""
    X_raw = np.asarray(
        [[r[col] for col in _FEATURE_NAMES] for r in rows],
        dtype=np.float64,
    )
    y = np.asarray([r["actual_bpp"] for r in rows], dtype=np.float64)

    n, p = X_raw.shape

    # StandardScaler
    mean_ = X_raw.mean(axis=0)
    std_ = X_raw.std(axis=0, ddof=0)
    scale_ = np.where(std_ > 1e-9, std_, 1.0)
    X_scaled = (X_raw - mean_) / scale_

    # Design matrix: [1 | X_scaled]
    ones = np.ones((n, 1), dtype=np.float64)
    A = np.hstack([ones, X_scaled])

    coeffs, _res, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coeffs[0])
    betas = [float(c) for c in coeffs[1:]]

    # Training residuals
    y_pred = A @ coeffs
    rel_err = np.abs((y_pred - y) / np.clip(y, 1e-9, None))

    return {
        "scaler": {
            "mean": mean_.tolist(),
            "scale": scale_.tolist(),
        },
        "coefficients": {
            "intercept": intercept,
            "betas": betas,
        },
        "train_residuals": {
            "median_rel_err": round(float(np.median(rel_err)), 6),
            "p95_rel_err": round(float(np.percentile(rel_err, 95)), 6),
            "max_rel_err": round(float(rel_err.max()), 6),
        },
    }


def _build_training_envelope(rows: list[dict]) -> dict:
    envelope = {}
    for name in _FEATURE_NAMES:
        vals = [r[name] for r in rows]
        envelope[name] = [float(min(vals)), float(max(vals))]
    return envelope


_PII_PATTERN = re.compile(r"[/@]|http")


def _assert_no_pii(artifact_json: str) -> None:
    for match in _PII_PATTERN.finditer(artifact_json):
        ctx_start = max(0, match.start() - 20)
        ctx_end = min(len(artifact_json), match.end() + 20)
        context = artifact_json[ctx_start:ctx_end]
        raise AssertionError(
            f"Possible PII detected in artifact JSON at position {match.start()}: "
            f"matched {match.group()!r} in context: ...{context}..."
        )


def main() -> int:
    args = _parse_args()
    quality_presets = [int(q.strip()) for q in args.quality_presets.split(",")]
    np.random.default_rng(args.seed)

    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_file = _manifest_path(args.manifest)
    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}", file=sys.stderr)
        return 1

    manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from bench.corpus.manifest import Manifest

    manifest = Manifest.load(manifest_file)
    jpeg_entries = [e for e in manifest.entries if "jpeg" in e.output_formats]

    if not jpeg_entries:
        print("ERROR: no JPEG entries found in manifest", file=sys.stderr)
        return 1

    corpus_root = _PROJECT_ROOT / "tests" / "corpus"
    print(
        f"Fitting JPEG header-only model: {len(jpeg_entries)} entries × {len(quality_presets)} "
        f"presets = {len(jpeg_entries) * len(quality_presets)} max rows",
        file=sys.stderr,
    )

    rows = asyncio.run(_run_cases(jpeg_entries, corpus_root, quality_presets))

    n = len(rows)
    print(f"Collected {n} valid rows (after NSE filter)", file=sys.stderr)

    if n < args.n_min:
        print(
            f"ERROR: collected {n} rows but --n-min={args.n_min} required. "
            f"Try --n-min {max(1, n)} or use a larger manifest.",
            file=sys.stderr,
        )
        return 1

    result = _fit_linear_model(rows)
    training_envelope = _build_training_envelope(rows)

    artifact = {
        "model_version": 1,
        "format": "jpeg_header",
        "features": _FEATURE_NAMES,
        "scaler": result["scaler"],
        "coefficients": result["coefficients"],
        "training_envelope": training_envelope,
        "training_corpus_sha256": manifest_sha,
        "git_sha": _git_sha(),
        "fit_environment": {
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    artifact_json = json.dumps(artifact, indent=2)

    try:
        _assert_no_pii(artifact_json)
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(artifact_json)

    res = result["train_residuals"]
    print("\n=== JPEG Header-Only Model Fit Summary ===")
    print(f"n             : {n}")
    print(f"n_entries     : {len(jpeg_entries)}")
    print(f"n_presets     : {len(quality_presets)}")
    print(f"median rel err: {res['median_rel_err']:.4f}  ({res['median_rel_err']*100:.2f}%)")
    print(f"p95    rel err: {res['p95_rel_err']:.4f}  ({res['p95_rel_err']*100:.2f}%)")
    print(f"max    rel err: {res['max_rel_err']:.4f}  ({res['max_rel_err']*100:.2f}%)")
    print(f"output        : {output_path}")
    print(f"intercept     : {result['coefficients']['intercept']:.6f}")
    print(f"betas[:5]     : {[round(b, 6) for b in result['coefficients']['betas'][:5]]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
