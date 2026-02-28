# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Format-specific image optimization engines. Each optimizer takes raw image bytes + `OptimizationConfig` and returns the smallest valid output.

## How to Add a New Optimizer

1. Create `optimizers/<format>.py` inheriting `BaseOptimizer` (or `PillowReencodeOptimizer` for Pillow-based formats — see below)
2. Set `format = ImageFormat.<FORMAT>` class attribute
3. Implement `async optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult`
4. Register in `router.py`'s `OPTIMIZERS` dict
5. Estimation adapts automatically (sample-based — no heuristics to update)
6. Add format detection in `utils/format_detect.py` (magic bytes + enum + MIME type)
7. Add to Dockerfile if new system dependencies are needed
8. Add benchmark cases in `benchmarks/cases.py` and encoder in `benchmarks/generators.py`
9. Add tests in `tests/test_formats.py` and optionally a dedicated `tests/test_optimizer_<format>.py`

For Pillow-based formats that use strip + re-encode, see `jxl.py` (cleanest example). For CLI-tool-based formats, see `tiff.py` and `bmp.py`.

## PillowReencodeOptimizer Base Class

`pillow_reencode.py` provides a shared base for formats that optimize via Pillow strip + re-encode (AVIF, HEIC, JXL). Subclasses set class attributes and override hooks:

- **Required attributes**: `format`, `pillow_format`, `strip_method_name`, `reencode_method_name`
- **Quality range**: `quality_min`, `quality_max`, `quality_offset` (defaults: 30, 90, 10)
- **Required override**: `_ensure_plugin()` — import/register the format's Pillow plugin (currently a no-op in all subclasses since plugins register at module level)
- **Optional overrides**: `_open_image(data)` (HEIC uses pillow-heif), `_strip_metadata_from_img(img, data)` (AVIF uses quality=100, HEIC uses quality=-1)
- **Decode-once pattern**: `optimize()` decodes via `_open_image()` once, passes `img.copy()` to strip and `img` to reencode. Override `_strip_metadata_from_img(img, data)` instead of `_strip_metadata(data)` for format-specific strip behavior.
- **Plugin imports**: Moved to module level — `_ensure_plugin()` is a no-op in all current subclasses.

## Shared Utilities (`utils.py`)

- **`clamp_quality(quality, offset, lo, hi)`**: Map Pare quality (1-100) to format-specific quality with offset and clamping. Used by `PillowReencodeOptimizer` and the estimator's BPP helpers.
- **`binary_search_quality(encode_fn, original_size, target_reduction, lo, hi)`**: Binary search for the lowest quality that stays within a `max_reduction` cap. Used by JPEG and WebP optimizers.

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
