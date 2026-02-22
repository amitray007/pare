# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Sample-based compression estimation. Instead of heuristic prediction, this module compresses a downsized sample of the image using the actual optimizers and extrapolates BPP (bits per pixel) to the full image size.

## Architecture

Two files:

1. **`estimator.py`** — Entry point. Downloads/receives image, determines whether to use exact mode (small/SVG/animated) or extrapolation mode (large raster). Calls the real optimizers via `optimize_image()`.

2. **`presets.py`** — Maps preset names (high/medium/low) to `OptimizationConfig` instances.

## Key Design Property

Estimation calls the actual optimizers. When optimizer logic changes, estimation automatically adapts. No parallel heuristic system to maintain.

## Modes

- **Exact mode** (<150K pixels, SVG, animated): Compresses the full file with the real optimizer. 100% accurate.
- **Extrapolation mode** (>150K pixels): Downsamples to ~300px wide, compresses sample, measures output BPP, scales to original pixel count.

## Verification

After changes to optimizers, run:
```
python -m benchmarks.run --fmt <format>
```
Check that estimation accuracy (Avg Err) stays under ~10%.
