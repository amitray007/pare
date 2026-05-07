"""Fit the PNG BPP regression model from the bench corpus.

Usage
-----
    python -m bench.fit.png \\
        --manifest core \\
        --seed 42 \\
        --output estimation/models/png_v1.json \\
        --n-min 30 \\
        --quality-presets 40,60,75,85

The script:

1. Loads the corpus and iterates every (PNG entry × quality preset) pair.
2. Runs ``optimize_image()`` to get the actual optimized size → actual BPP.
3. Runs ``extract_png_features()`` on the decoded image.
4. Accumulates rows; skips entries where feature extraction returns None.
5. Asserts ``n >= n_min``.
6. Fits via ``bench.fit.common.train_one()``.
7. Writes the model JSON to *output* (plain write, no atomic dance — each
   uvicorn worker reads the JSON at boot, not on-the-fly; the file is only
   ever written by the offline fit script, not by a concurrent server).
8. Prints a summary: n, fit residuals, output path.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
logger = logging.getLogger("bench.fit.png")

# Raise PIL decompression bomb limit to match project settings (100 MP).
# Must be set before any PIL Image operations (including in threads).
import PIL.Image as _PIL_Image  # noqa: E402

_PIL_Image.MAX_IMAGE_PIXELS = 100_000_000

# Resolve paths relative to the project root (two levels up from bench/fit/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _manifest_path(name: str) -> Path:
    if name.endswith(".json"):
        return Path(name)
    return Path(__file__).parent.parent / "corpus" / "manifests" / f"{name}.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PNG BPP regression model from bench corpus.",
    )
    parser.add_argument(
        "--manifest",
        default="core",
        help="Corpus manifest name or path (default: core).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        default="estimation/models/png_v1.json",
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
    """Run optimize + feature extraction for every (entry × quality) pair."""
    import io

    from PIL import Image

    from bench.corpus.builder import file_path
    from estimation.png_features import extract_png_features
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

        # Decode image once per entry for feature extraction.
        # Disable decompression bomb check for this offline fit script
        # (corpus images can exceed 100 MP).
        def _open_png(d: bytes) -> Image.Image:
            import PIL.Image

            PIL.Image.MAX_IMAGE_PIXELS = None  # disable bomb check for offline script
            return PIL.Image.open(io.BytesIO(d))

        img = await asyncio.to_thread(_open_png, data)
        width, height = img.size
        # Ensure pixel data is loaded for feature extraction
        await asyncio.to_thread(img.load)

        for quality in quality_presets:
            done += 1
            config = OptimizationConfig(quality=quality, png_lossy=True)

            # Run actual optimizer to get true optimized size → BPP
            try:
                result = await optimize_image(data, config)
            except Exception as exc:
                logger.warning("optimize_image failed for %s q=%d: %s", entry.name, quality, exc)
                skipped += 1
                continue

            actual_size = result.optimized_size
            orig_pixels = width * height
            if orig_pixels == 0:
                skipped += 1
                continue

            actual_bpp = actual_size * 8 / orig_pixels

            # Extract features (pass file_size for input_bpp computation)
            features = await asyncio.to_thread(
                extract_png_features, img, width, height, quality, file_size
            )
            if features is None:
                logger.info("features=None for %s q=%d — skipping", entry.name, quality)
                skipped += 1
                continue

            rows.append(
                {
                    "name": entry.name,
                    "quality": quality,
                    "actual_bpp": actual_bpp,
                    "file_size": file_size,
                    "actual_size": actual_size,
                    # feature fields (matches PngFeatures field order)
                    "has_alpha": float(features.has_alpha),
                    "log10_unique_colors": features.log10_unique_colors,
                    "mean_sobel": features.mean_sobel,
                    "edge_density": features.edge_density,
                    "quality_feat": float(features.quality),
                    "log10_orig_pixels": features.log10_orig_pixels,
                    "input_bpp": features.input_bpp,
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


def _build_training_envelope(rows: list[dict], feature_names: list[str]) -> dict:
    """Forensic min/max per feature for the training envelope."""
    envelope = {}
    for name in feature_names:
        vals = [r[name] for r in rows]
        envelope[name] = [float(min(vals)), float(max(vals))]
    return envelope


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

    # Load manifest
    sys.path.insert(0, str(_PROJECT_ROOT))
    from bench.corpus.manifest import Manifest

    manifest = Manifest.load(manifest_file)
    png_entries = [e for e in manifest.entries if "png" in e.output_formats]

    if not png_entries:
        print("ERROR: no PNG entries found in manifest", file=sys.stderr)
        return 1

    corpus_root = _PROJECT_ROOT / "tests" / "corpus"
    print(
        f"Fitting PNG model: {len(png_entries)} entries × {len(quality_presets)} presets "
        f"= {len(png_entries) * len(quality_presets)} max rows",
        file=sys.stderr,
    )

    # Run async case collection
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

    # Feature names in PngFeatures order (must match model artifact's ``features`` list).
    # Rows use 'quality_feat' internally; we map to 'quality' for the model artifact.
    feature_names = [
        "has_alpha",
        "log10_unique_colors",
        "mean_sobel",
        "edge_density",
        "quality",
        "log10_orig_pixels",
        "input_bpp",
    ]
    # Column keys in the rows dict (quality stored as 'quality_feat' to avoid shadowing)
    feature_row_keys = [
        "has_alpha",
        "log10_unique_colors",
        "mean_sobel",
        "edge_density",
        "quality_feat",
        "log10_orig_pixels",
        "input_bpp",
    ]

    # Build numpy arrays from list-of-dicts (no pandas)
    X = np.asarray([[r[col] for col in feature_row_keys] for r in rows], dtype=np.float64)
    targets = np.asarray([r["actual_bpp"] for r in rows], dtype=np.float64)

    # Rename rows to use model feature names (for training_envelope)
    renamed_rows = [
        {fname: r[rkey] for fname, rkey in zip(feature_names, feature_row_keys)} for r in rows
    ]

    # Fit (three knots: log10_unique_colors at 3.3, quality at 50 and 70)
    from bench.fit.common import train_one

    result = train_one(X, feature_names, targets, knot=3.3, knot_q50=50.0, knot_q70=70.0)

    # Training envelope (forensic)
    training_envelope = _build_training_envelope(renamed_rows, feature_names)

    # Build model artifact JSON (model_version=2 — schema changed: added input_bpp + quality knots)
    artifact = {
        "model_version": 2,
        "format": "png",
        "features": feature_names,
        "supported_modes": ["RGB", "RGBA", "L", "LA", "P"],
        "scaler": result["scaler"],
        "coefficients": result["coefficients"],
        "knot_log10_unique_colors": result["knot_log10_unique_colors"],
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

    # Write model JSON (plain write — multi-process invariant: no in-place mutation
    # on a running instance; artifact updates require a new deploy revision)
    output_path.write_text(json.dumps(artifact, indent=2))

    # Summary
    res = result["train_residuals"]
    print("\n=== PNG Model Fit Summary ===")
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
