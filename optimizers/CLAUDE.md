# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Format-specific image optimization engines. Each optimizer takes raw image bytes + `OptimizationConfig` and returns the smallest valid output.

## How to Add a New Optimizer

1. Create `optimizers/<format>.py` inheriting `BaseOptimizer`
2. Set `format = ImageFormat.<FORMAT>` class attribute
3. Implement `async optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult`
4. Register in `router.py`'s `OPTIMIZERS` dict
5. Add matching `_predict_<format>()` in `estimation/heuristics.py`

## Key Pattern: Try All, Pick Best

Optimizers try multiple compression methods and return the smallest result. See `tiff.py` (deflate vs LZW vs JPEG-in-TIFF) and `bmp.py` (24-bit vs palette vs RLE8) for the clearest examples. The `_build_result()` method in `base.py` enforces the output-never-larger-than-input guarantee automatically.

## Quality Thresholds

`config.quality` (1-100, lower = more aggressive) drives method selection. The standard breakpoints are:
- `quality < 50`: Aggressive lossy (HIGH preset, q=40)
- `quality < 70`: Moderate lossy (MEDIUM preset, q=60)
- `quality >= 70`: Lossless only (LOW preset, q=80)

Each optimizer defines its own thresholds — these are conventions, not hard rules.

## CLI Tools vs Libraries

- **CLI tools** (pngquant, cjpeg, jpegtran, gifsicle, cwebp): Invoked via `utils/subprocess_runner.py`'s `run_tool()` — bytes in via stdin, bytes out via stdout, no temp files
- **Python libraries** (oxipng, Pillow, pillow-heif, scour): Called directly in-process

## Conventions

- `config.strip_metadata` should be handled early (before optimization) using `utils/metadata.py`
- `config.max_reduction` caps lossy methods — lossless methods are never capped
- Method names reported in results should be descriptive (e.g., `"pngquant + oxipng"`, `"bmp-rle8"`, `"jpegli"`)
- Use `asyncio.gather()` for concurrent independent operations (e.g., PNG runs pngquant and oxipng-only in parallel)
