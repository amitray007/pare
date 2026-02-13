# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Fast compression prediction without running full optimization. Target: <50ms latency, <15% average estimation error (measured by benchmarks).

## Architecture

Three files forming a pipeline:

1. **`estimator.py`** — Entry point. Calls header analysis then heuristics, returns `EstimateResponse`. A thumbnail compression layer (layer 3) exists but is disabled — the heuristic model is accurate enough.

2. **`header_analysis.py`** — Parses image headers without full pixel decode. Extracts dimensions, color type, bit depth, JPEG quality (from quantization tables), PNG palette info, SVG bloat ratio. Computes `flat_pixel_ratio` (center crop content classification) for JPEG, TIFF, BMP, and JXL. For small files (<12KB), stores `raw_data` on `HeaderInfo` so heuristics can run exact probes.

3. **`heuristics.py`** — Per-format prediction functions dispatched via `predict_reduction()`. Each `_predict_<format>()` takes `HeaderInfo` + `OptimizationConfig` and returns a `Prediction` dataclass. Covers all 12 formats: PNG, APNG, JPEG, WebP, GIF, SVG, SVGZ, AVIF, HEIC, TIFF, BMP, JXL. AVIF/HEIC/JXL use bpp-based models; BMP uses content-aware RLE8 bonus scaled by `flat_pixel_ratio`.

## Critical Rule: Estimation Must Match Optimizer

When you change optimizer quality thresholds or add compression tiers, the corresponding `_predict_*` function must use the **same quality breakpoints**. For example, if `optimizers/bmp.py` uses `quality < 70` for palette mode, `_predict_bmp` must also switch at `quality < 70`.

## Probes

Some formats run lightweight probes for better accuracy on small files:
- **JPEG probe** (`_jpeg_probe`): Runs actual Pillow encode + jpegtran on files <12KB
- **PNG lossy probe** (`_png_lossy_probe`): Runs pngquant + oxipng on files <12KB with quality <70

Probes are gated by `info.raw_data is not None` (only set for small files).

## Verification

After changing heuristics, run:
```bash
python -m benchmarks.run --fmt <format>
```
Check the "ESTIMATION ACCURACY" section — Avg Err should stay under ~15%. The "Top 10 worst estimates" table shows which cases need attention.
