# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Format-specific image optimization engines. Each optimizer takes raw image bytes + `OptimizationConfig` and returns the smallest valid output.

## How to Add a New Optimizer

1. Create `optimizers/<format>.py` inheriting `BaseOptimizer`
2. Set `format = ImageFormat.<FORMAT>` class attribute
3. Implement `async optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult`
4. Register in `router.py`'s `OPTIMIZERS` dict
5. Estimation adapts automatically (sample-based — no heuristics to update)
6. Add format detection in `utils/format_detect.py` (magic bytes + enum + MIME type)
7. Add to Dockerfile if new system dependencies are needed
8. Add benchmark cases in `benchmarks/cases.py` and encoder in `benchmarks/generators.py`
9. Add tests in `tests/test_formats.py` and optionally a dedicated `tests/test_optimizer_<format>.py`

See `jxl.py` and `avif.py` for recent examples of the full end-to-end pattern.

## Key Pattern: Try All, Pick Best

Optimizers try multiple compression methods and return the smallest result. See `tiff.py` (deflate vs LZW vs JPEG-in-TIFF) and `bmp.py` (24-bit vs palette vs RLE8) for the clearest examples. The `_build_result()` method in `base.py` enforces the output-never-larger-than-input guarantee automatically.

## Quality Thresholds

`config.quality` (1-100, lower = more aggressive) drives method selection. The standard breakpoints are:
- `quality < 50`: Aggressive lossy (HIGH preset, q=40)
- `quality < 70`: Moderate lossy (MEDIUM preset, q=60)
- `quality >= 70`: Lossless only (LOW preset, q=80)

Each optimizer defines its own thresholds — these are conventions, not hard rules.

## CLI Tools vs Libraries

- **CLI tools** (pngquant, jpegtran, gifsicle, cwebp, cjxl/djxl): Invoked via `utils/subprocess_runner.py`'s `run_tool()` — bytes in via stdin, bytes out via stdout, no temp files
- **Python libraries** (oxipng, Pillow/jpegli, pillow-heif, pillow-avif-plugin, jxlpy, scour): Called directly in-process

Note: JPEG encoding uses Pillow with jpegli (libjpeg.so.62 from libjxl) in Docker, falling back to libjpeg-turbo locally. The `cjpeg` MozJPEG fallback is available via `JPEG_ENCODER=cjpeg` config.

## Conventions

- `config.strip_metadata` should be handled early (before optimization) using `utils/metadata.py`
- `config.max_reduction` caps lossy methods — lossless methods are never capped
- Method names reported in results should be descriptive (e.g., `"pngquant + oxipng"`, `"bmp-rle8"`, `"jpegli"`)
- Wrap CPU-bound Pillow ops in `asyncio.to_thread()` to avoid blocking the event loop
- Use `asyncio.gather()` for concurrent independent operations (e.g., PNG runs pngquant and oxipng-only in parallel, TIFF runs deflate/LZW/JPEG concurrently, JPEG runs jpegli and jpegtran concurrently)
