# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Sample-based compression estimation. Instead of heuristic prediction, this module compresses a downsized sample of the image using the actual optimizers and extrapolates BPP (bits per pixel) to the full image size.

## Architecture

Two files:

1. **`estimator.py`** — Entry point. Downloads/receives image, determines whether to use exact mode (small/SVG/animated) or extrapolation mode (large raster).

2. **`presets.py`** — Maps preset names (high/medium/low) to `OptimizationConfig` instances.

## Modes

- **Exact mode** (<150K pixels, SVG, animated): Compresses the full file with the real optimizer. 100% accurate.
- **Direct-encode mode** (JPEG, HEIC, AVIF, JXL, WebP, PNG, APNG): Encodes a downsized sample directly at target quality using format-specific `_*_sample_bpp()` helpers, then extrapolates BPP to original pixel count. Bypasses the optimizer pipeline to avoid its output-never-larger gate breaking on small samples.
- **Generic fallback mode** (GIF, BMP, TIFF): Creates a minimally-compressed sample via `_create_sample()`, runs the actual optimizer on it, extrapolates BPP.

## Direct-Encode BPP Helpers

Each helper mirrors the corresponding optimizer's encoding settings via pyvips:

| Helper | Optimizer match | Quality mapping |
|--------|----------------|-----------------|
| `_jpeg_sample_bpp` | `optimizers/jpeg.py` jpegsave | `config.quality`, optimize_coding=True |
| `_heic_sample_bpp` | `optimizers/heic.py` heifsave | `max(30, min(90, quality + 10))`, compression=hevc |
| `_avif_sample_bpp` | `optimizers/avif.py` heifsave | `max(30, min(90, quality + 10))`, compression=av1, effort=4 |
| `_jxl_sample_bpp` | `optimizers/jxl.py` jxlsave | `max(30, min(95, quality + 10))`, effort=7 |
| `_webp_sample_bpp` | `optimizers/webp.py` webpsave | `config.quality`, effort=4 |
| `_png_sample_bpp` | `optimizers/png.py` pipeline | pyvips palette (libimagequant) + oxipng |

**IMPORTANT:** When changing quality mappings or encoding parameters in an optimizer, update the corresponding `_*_sample_bpp()` helper to match.

## Verification

After changes to optimizers or estimation helpers, run:
```
python -m benchmarks.run --fmt <format>
```
Check that estimation accuracy (Avg Err) stays under ~10%.
