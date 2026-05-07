"""Fit the PNG header-only BPP regression model from the bench corpus.

Usage
-----
    python -m bench.fit.png_header \\
        --manifest full \\
        --seed 42 \\
        --output estimation/models/png_header_v1.json \\
        --n-min 60 \\
        --quality-presets 40,60,75,85

The script:

1. Loads the corpus and iterates every (PNG entry × quality preset) pair.
2. Runs ``optimize_image()`` to get the actual optimized size → actual BPP.
3. Parses the PNG header with ``parse_png_header()`` for ``has_alpha``, dimensions.
4. Computes 4 features:
   - ``has_alpha``       (0 or 1 from IHDR color type)
   - ``quality``         (preset value)
   - ``log10_orig_pixels`` (log10 of width × height)
   - ``input_bpp``       (file_size × 8 / (width × height))
5. Skips rows where header parse fails or features are NaN.
6. Asserts ``n >= n_min``.
7. Fits via ``bench.fit.common.train_one()`` — uses a modified call without
   the ``log10_unique_colors`` knot (not available from header-only features).
8. Writes the model JSON to *output* (plain write — see ``bench/fit/png.py``).
9. Prints summary: n, fit residuals, output path.
10. Asserts no PII in the output JSON.

Reuses ``bench/fit/common.py``'s ``StandardScaler`` math and residual
computation via ``train_one()``.  The feature set (4 features, no
``log10_unique_colors``) means we skip the ``log10_unique_colors`` knot column —
``train_one`` requires that column by name.  We use a reduced version of
``train_one`` that only applies the two quality knots (q=50, q=70).
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
from pathlib import Path

import numpy as np
import scipy

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("bench.fit.png_header")

# Raise PIL decompression bomb limit to match project settings (100 MP).
import PIL.Image as _PIL_Image  # noqa: E402

_PIL_Image.MAX_IMAGE_PIXELS = 100_000_000

# Resolve paths relative to the project root (two levels up from bench/fit/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Feature names for this model — must match PngHeaderModel._PNG_HEADER_FEATURES exactly.
_FEATURE_NAMES = ["has_alpha", "quality", "log10_orig_pixels", "input_bpp"]


def _manifest_path(name: str) -> Path:
    if name.endswith(".json"):
        return Path(name)
    return Path(__file__).parent.parent / "corpus" / "manifests" / f"{name}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PNG header-only BPP regression model from bench corpus.",
    )
    parser.add_argument(
        "--manifest",
        default="full",
        help="Corpus manifest name or path (default: full).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        default="estimation/models/png_header_v1.json",
        help="Output path for the model JSON artifact.",
    )
    parser.add_argument(
        "--n-min",
        type=int,
        default=60,
        help="Minimum number of training rows required (default: 60).",
    )
    parser.add_argument(
        "--quality-presets",
        default="40,60,75,85",
        help="Comma-separated quality values to run (default: 40,60,75,85).",
    )
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


async def _run_cases(
    png_entries: list,
    corpus_root: Path,
    quality_presets: list[int],
) -> list[dict]:
    """Run optimize + header parse for every (entry × quality) pair."""
    from math import log10

    from bench.corpus.builder import file_path
    from estimation.png_header import parse_png_header
    from optimizers.router import optimize_image
    from schemas import OptimizationConfig

    rows = []
    total = len(png_entries) * len(quality_presets)
    done = 0
    skipped = 0

    for entry in png_entries:
        path = file_path(corpus_root, entry, "png")
        if not path.exists():
            logger.warning("corpus file missing: %s — skipping", path)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        data = path.read_bytes()
        file_size = len(data)

        # Parse PNG header — pure-bytes, no Pillow
        hdr = parse_png_header(data)
        if hdr is None:
            logger.warning("PNG header parse failed for %s — skipping all presets", entry.name)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        width, height = hdr.width, hdr.height
        orig_pixels = width * height
        if orig_pixels == 0:
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        has_alpha_f = float(hdr.has_alpha)
        try:
            log10_orig_pixels = log10(orig_pixels)
            input_bpp = (file_size * 8) / orig_pixels
        except (ValueError, ZeroDivisionError):
            logger.warning("feature computation failed for %s — skipping", entry.name)
            skipped += len(quality_presets)
            done += len(quality_presets)
            continue

        for quality in quality_presets:
            done += 1
            config = OptimizationConfig(quality=quality, png_lossy=True)

            try:
                result = await optimize_image(data, config)
            except Exception as exc:
                logger.warning("optimize_image failed for %s q=%d: %s", entry.name, quality, exc)
                skipped += 1
                continue

            actual_size = result.optimized_size
            actual_bpp = actual_size * 8 / orig_pixels

            rows.append(
                {
                    "name": entry.name,
                    "quality": float(quality),
                    "actual_bpp": actual_bpp,
                    "has_alpha": has_alpha_f,
                    "log10_orig_pixels": log10_orig_pixels,
                    "input_bpp": input_bpp,
                }
            )

            if done % 20 == 0:
                print(
                    f"  [{done}/{total}] rows collected={len(rows)} skipped={skipped}",
                    file=sys.stderr,
                )

    print(
        f"  [{done}/{total}] rows collected={len(rows)} skipped={skipped}",
        file=sys.stderr,
    )
    return rows


def _train_header_model(
    X: np.ndarray,
    targets: np.ndarray,
    knot_q50: float = 50.0,
    knot_q70: float = 70.0,
) -> dict:
    """Fit a piecewise-linear OLS regression with two quality knots.

    The header-only model has 4 features: has_alpha, quality, log10_orig_pixels,
    input_bpp.  No ``log10_unique_colors`` knot — that requires thumbnail decode.
    Knots are on the quality axis only (q=50 and q=70).

    Returns dict matching PngHeaderModel.from_json expectations.
    """
    y = np.asarray(targets, dtype=np.float64)
    X_raw = np.asarray(X, dtype=np.float64)
    n, p = X_raw.shape
    assert n == len(y), f"Row count mismatch: X has {n} rows, y has {len(y)}"
    assert p == len(
        _FEATURE_NAMES
    ), f"Column count mismatch: X has {p} columns, expected {len(_FEATURE_NAMES)}"

    # --- StandardScaler ---
    mean_ = X_raw.mean(axis=0)
    std_ = X_raw.std(axis=0, ddof=0)
    scale_ = np.where(std_ > 1e-9, std_, 1.0)
    X_scaled = (X_raw - mean_) / scale_

    # --- Piecewise-linear knot columns for quality ---
    # quality is at index 1 (feature order: has_alpha, quality, log10_orig_pixels, input_bpp)
    quality_col_idx = _FEATURE_NAMES.index("quality")
    quality_raw = X_raw[:, quality_col_idx]
    knot_q50_col = np.clip(quality_raw - knot_q50, 0.0, None)
    knot_q70_col = np.clip(quality_raw - knot_q70, 0.0, None)

    # Design matrix: [1 | X_scaled | knot_q50_col | knot_q70_col]
    ones = np.ones((n, 1), dtype=np.float64)
    A = np.hstack(
        [
            ones,
            X_scaled,
            knot_q50_col.reshape(-1, 1),
            knot_q70_col.reshape(-1, 1),
        ]
    )

    # --- OLS via lstsq ---
    coeffs, _residuals, _rank, _sv = np.linalg.lstsq(A, y, rcond=None)
    intercept = float(coeffs[0])
    betas = [float(c) for c in coeffs[1 : p + 1]]
    knot_q50_beta = float(coeffs[p + 1])
    knot_q70_beta = float(coeffs[p + 2])

    # --- Training residuals ---
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
            "knot_q50_beta": knot_q50_beta,
            "knot_q70_beta": knot_q70_beta,
        },
        "knot_q50": knot_q50,
        "knot_q70": knot_q70,
        "train_residuals": {
            "median_rel_err": round(float(np.median(rel_err)), 6),
            "p95_rel_err": round(float(np.percentile(rel_err, 95)), 6),
            "max_rel_err": round(float(rel_err.max()), 6),
        },
    }


def _build_training_envelope(rows: list[dict], feature_names: list[str]) -> dict:
    """Forensic min/max per feature for the training envelope."""
    envelope = {}
    for name in feature_names:
        vals = [r[name] for r in rows]
        envelope[name] = [float(min(vals)), float(max(vals))]
    return envelope


_PII_PATTERN = re.compile(r"[/@]|http")


def _assert_no_pii(artifact_json: str) -> None:
    """Regex sweep for user-derived strings that could leak PII."""
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
    np.random.default_rng(args.seed)  # seed numpy RNG; lstsq is deterministic

    # Resolve paths relative to project root
    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_file = _manifest_path(args.manifest)
    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}", file=sys.stderr)
        return 1

    # Compute manifest SHA256
    manifest_sha = hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    sys.path.insert(0, str(_PROJECT_ROOT))
    from bench.corpus.manifest import Manifest

    manifest = Manifest.load(manifest_file)
    png_entries = [e for e in manifest.entries if "png" in e.output_formats]

    if not png_entries:
        print("ERROR: no PNG entries found in manifest", file=sys.stderr)
        return 1

    corpus_root = _PROJECT_ROOT / "tests" / "corpus"
    print(
        f"Fitting PNG header-only model: {len(png_entries)} entries × {len(quality_presets)} "
        f"presets = {len(png_entries) * len(quality_presets)} max rows",
        file=sys.stderr,
    )

    rows = asyncio.run(_run_cases(png_entries, corpus_root, quality_presets))

    n = len(rows)
    print(f"Collected {n} valid rows", file=sys.stderr)

    if n < args.n_min:
        print(
            f"ERROR: collected {n} rows but --n-min={args.n_min} required. "
            f"Try --n-min {max(1, n)} or use a larger manifest.",
            file=sys.stderr,
        )
        return 1

    # Column keys map onto _FEATURE_NAMES order
    feature_row_keys = ["has_alpha", "quality", "log10_orig_pixels", "input_bpp"]

    X = np.asarray([[r[col] for col in feature_row_keys] for r in rows], dtype=np.float64)
    targets = np.asarray([r["actual_bpp"] for r in rows], dtype=np.float64)

    result = _train_header_model(X, targets, knot_q50=50.0, knot_q70=70.0)

    training_envelope = _build_training_envelope(rows, feature_row_keys)

    artifact = {
        "model_version": 1,
        "format": "png_header",
        "features": _FEATURE_NAMES,
        "scaler": result["scaler"],
        "coefficients": result["coefficients"],
        "knot_q50": result["knot_q50"],
        "knot_q70": result["knot_q70"],
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

    # PII assertion: check for paths, URLs, email-like patterns in user-derived strings.
    # We only check the string fields (git_sha, created_at come from controlled sources).
    # Scan the full JSON to be conservative.
    try:
        _assert_no_pii(artifact_json)
    except AssertionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(artifact_json)

    res = result["train_residuals"]
    print("\n=== PNG Header-Only Model Fit Summary ===")
    print(f"n             : {n}")
    print(f"n_entries     : {len(png_entries)}")
    print(f"n_presets     : {len(quality_presets)}")
    print(f"median rel err: {res['median_rel_err']:.4f}  ({res['median_rel_err']*100:.2f}%)")
    print(f"p95    rel err: {res['p95_rel_err']:.4f}  ({res['p95_rel_err']*100:.2f}%)")
    print(f"max    rel err: {res['max_rel_err']:.4f}  ({res['max_rel_err']*100:.2f}%)")
    print(f"output        : {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
