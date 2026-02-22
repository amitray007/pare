# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is Pare?

Pare is a serverless image compression API built on FastAPI + Google Cloud Run. It optimizes 12 image formats (PNG, APNG, JPEG, WebP, GIF, SVG, SVGZ, AVIF, HEIC, TIFF, BMP, JXL) using format-specific pipelines that combine CLI tools (MozJPEG, pngquant, oxipng, gifsicle, cwebp, cjxl/djxl) with Python libraries (Pillow, jpegli, pillow-heif, pillow-avif-plugin, jxlpy, scour).

## Common Commands

```bash
# Run the server locally
uvicorn main:app --reload --port 8080

# Run all tests
pytest tests/

# Run tests for a specific format or area
pytest tests/ -k "bmp"
pytest tests/test_security.py

# Run benchmarks (all formats, all presets)
python -m benchmarks.run

# Run benchmarks filtered by format and/or preset
python -m benchmarks.run --fmt bmp --preset high

# Compare current benchmark against previous run
python -m benchmarks.run --compare

# Docker
docker-compose up          # Pare + Redis (local dev)
docker build -t pare .     # Full build with jpegli, MozJPEG, JXL tools
```

## Architecture

### Optimization Pipeline

Request flow: `routers/optimize.py` -> `optimizers/router.py` (format detection + dispatch) -> format-specific optimizer -> `BaseOptimizer._build_result()` (enforces output <= input guarantee).

Each optimizer in `optimizers/` inherits `BaseOptimizer` and implements `async optimize(data, config) -> OptimizeResult`. The router holds a singleton registry (`OPTIMIZERS` dict) mapping `ImageFormat` enum values to optimizer instances.

### Estimation Engine (sample-based)

`routers/estimate.py` -> `estimation/estimator.py` -> `optimizers/router.py`

Estimates compression by compressing a downsized sample (~300px wide) with the actual optimizer and extrapolating BPP to the full image size. For small images (<150K pixels), SVG, and animated formats, compresses the full file for exact results.

Accepts presets (HIGH/MEDIUM/LOW) mapped to quality levels in `estimation/presets.py`. For images >= 10MB, supports an optional `thumbnail_url` to avoid downloading the full original.

### Quality Controls

`OptimizationConfig.quality` (1-100) drives format-specific behavior. Lower quality = more aggressive compression. The benchmark presets map to: HIGH (q=40), MEDIUM (q=60), LOW (q=80). Each optimizer defines its own quality thresholds (e.g., `quality < 70` = lossy path, `quality < 50` = aggressive).

`max_reduction` caps how much the optimizer is allowed to shrink a file.

### Concurrency

`utils/concurrency.py` has a `CompressionGate` (semaphore + queue depth cap). When the queue is full, the API returns 503 immediately rather than buffering unbounded 32MB payloads.

All CPU-bound Pillow operations are wrapped in `asyncio.to_thread()` to avoid blocking the event loop. Some optimizers (TIFF, PNG, JPEG) use `asyncio.gather()` to run independent compression methods concurrently in separate threads.

## Key Conventions

- **Optimizer pattern**: Try multiple methods, pick the smallest output. See `optimizers/tiff.py` and `optimizers/bmp.py` for the clearest examples of this "try all, pick best" pattern.
- **Estimation automatically matches optimizers**: The sample-based estimator calls the actual optimizers, so estimation accuracy adapts automatically when optimizer logic changes.
- **Output guarantee**: `_build_result()` in `base.py` ensures the API never returns a file larger than the input. If optimization makes it bigger, it returns the original with method="none".
- **Format detection**: Done by magic bytes in `utils/format_detect.py`, never by file extension or Content-Type header.
- **Benchmark verification**: After changing optimizer or estimation logic, run `python -m benchmarks.run --fmt <format>` and check that preset differentiation exists (HIGH > MEDIUM > LOW reduction) and estimation accuracy (Avg Err column) stays under ~15%.
