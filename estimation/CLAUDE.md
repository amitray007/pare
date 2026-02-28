# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Sample-based compression estimation. Instead of heuristic prediction, this module compresses a downsized sample of the image using the actual optimizers and extrapolates BPP (bits per pixel) to the full image size.

## Architecture

Two files:

1. **`estimator.py`** — Entry point. Downloads/receives image, determines whether to use exact mode (small/SVG/animated) or extrapolation mode (large raster).

2. **`presets.py`** — Maps preset names (high/medium/low) to `OptimizationConfig` instances. Delegates to `benchmarks.constants.PRESETS_BY_NAME` as the single source of truth.

## Modes

- **Exact mode** (<150K pixels, SVG, animated): Compresses the full file with the real optimizer. 100% accurate.
- **Direct-encode mode** (JPEG, HEIC, AVIF, JXL, WebP, PNG, APNG): Encodes a downsized sample directly at target quality using format-specific `_*_sample_bpp()` helpers, then extrapolates BPP to original pixel count. Bypasses the optimizer pipeline to avoid its output-never-larger gate breaking on small samples.
- **Generic fallback mode** (GIF, BMP, TIFF): Creates a minimally-compressed sample via `_create_sample()`, runs the actual optimizer on it, extrapolates BPP.

## Direct-Encode BPP Helpers

Each helper mirrors the corresponding optimizer's encoding settings. Quality clamping for HEIC/AVIF/JXL uses `clamp_quality()` from `optimizers/utils.py` — the same function used by the optimizers themselves.

| Helper | Optimizer match | Quality mapping |
|--------|----------------|-----------------|
| `_jpeg_sample_bpp` | `optimizers/jpeg.py` Pillow path | `config.quality` directly |
| `_heic_sample_bpp` | `optimizers/heic.py` `_reencode` | `clamp_quality(config.quality)` (offset=10, lo=30, hi=90) |
| `_avif_sample_bpp` | `optimizers/avif.py` `_reencode` | `clamp_quality(config.quality)` (offset=10, lo=30, hi=90), speed=6 |
| `_jxl_sample_bpp` | `optimizers/jxl.py` `_reencode` | `clamp_quality(config.quality, hi=95)` |
| `_webp_sample_bpp` | `optimizers/webp.py` Pillow path | `config.quality`, method=4 |
| `_png_sample_bpp` | `optimizers/png.py` pipeline | oxipng level + optional pngquant quantization |

**IMPORTANT:** When changing quality mappings or encoding parameters in an optimizer, update the corresponding `_*_sample_bpp()` helper to match. For HEIC/AVIF/JXL, both the optimizer and BPP helper use `clamp_quality()`, so changes to the quality range constants in the optimizer class are automatically reflected.

## Verification

After changes to optimizers or estimation helpers, run:
```
python -m benchmarks.run --fmt <format>
```
Check that estimation accuracy (Avg Err) stays under ~10%.
