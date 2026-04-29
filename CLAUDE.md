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

# Run a single test
pytest tests/test_sample_estimator.py::test_large_jpeg_extrapolation -v

# Lint and format check
python -m ruff check . && python -m black --check .

# Build the deterministic corpus (synthetic images, manifest-pinned)
python -m bench.corpus build --manifest core
python -m bench.corpus verify --manifest core             # Re-synthesize and check pixel hashes
python -m bench.corpus list --manifest core                # Inspect manifest contents

# Run benchmarks
python -m bench.run --mode quick                           # 1 iter per case, ~1 min
python -m bench.run --mode timing --repeat 5 --warmup 1    # p50/p95/p99 + MAD
python -m bench.run --mode memory                          # peak RSS + tracemalloc
python -m bench.run --mode timing --fmt png --bucket small # filter by format/bucket/tag
python -m bench.run --annotate "branch=staging" --out reports/bench.json

# Diff two runs with Welch's t-test + Cohen's d
python -m bench.compare reports/baseline.json reports/head.json --threshold-pct 10

# Render a run as Markdown (PR comment / step summary)
python -m bench.run report reports/bench.json --format markdown

# Docker
docker-compose up          # Pare + Redis (local dev)
docker build -t pare .     # Full build with jpegli, MozJPEG, JXL tools
```

## Architecture

### Request Flow

Four endpoints:

- **`GET /`** (in `main.py`): Service info — name, version, supported formats, and available endpoints. Also provides a structured 404 response for unmatched routes.
- **`POST /optimize`** (in `routers/`): Multipart file upload or JSON with URL -> `optimizers/router.py` (format detection + dispatch) -> format-specific optimizer -> binary response (or JSON with GCS storage URL). Acquires `CompressionGate` semaphore slot.
- **`POST /estimate`** (in `routers/`): Same input modes -> `estimation/estimator.py` (sample-based compression) -> JSON response. Does **not** acquire semaphore slot. Latency: ~50-500ms depending on format.
- **`GET /health`** (in `routers/`): Returns `"ok"` or `"degraded"` based on CLI tool availability.

Middleware chain (in `middleware.py`): request ID injection -> authentication -> rate limiting -> route handler.

### Optimization Pipeline

Each optimizer in `optimizers/` inherits `BaseOptimizer` and implements `async optimize(data, config) -> OptimizeResult`. The router holds a singleton registry (`OPTIMIZERS` dict) mapping `ImageFormat` enum values to optimizer instances.

`_build_result()` in `base.py` enforces the output-never-larger guarantee: if optimization produces a bigger file, it returns the original with `method="none"`.

### Estimation Engine (sample-based)

`estimation/estimator.py` has three modes:

1. **Exact mode** (<150K pixels, SVG, animated): Compresses the full file with the real optimizer.
2. **Direct-encode mode** (JPEG, HEIC, AVIF, JXL, WebP, PNG): Encodes a downsized sample at target quality using format-specific `_*_sample_bpp()` helpers, extrapolates BPP to full image. Each helper mirrors the corresponding optimizer's quality mapping — if you change an optimizer's quality logic, update the matching helper.
3. **Generic fallback mode** (GIF, BMP, TIFF): Creates a minimally-compressed sample via `_create_sample()`, runs the actual optimizer, extrapolates BPP.

Sample widths: JPEG 1200px, HEIC/AVIF/JXL/WebP/PNG 800px, GIF/BMP/TIFF 300px. Presets (HIGH/MEDIUM/LOW) mapped in `estimation/presets.py`.

### Quality Controls

`OptimizationConfig.quality` (1-100) drives format-specific behavior. Lower quality = more aggressive compression. Benchmark presets: HIGH (q=40), MEDIUM (q=60), LOW (q=75). Standard quality breakpoints across optimizers: `< 50` = aggressive lossy, `< 70` = moderate lossy, `>= 70` = lossless only.

`max_reduction` caps how much the optimizer is allowed to shrink a file (binary search for the right quality).

### Concurrency

`CompressionGate` in `utils/concurrency.py` is a semaphore (CPU count) + queue depth cap (2x semaphore). Returns 503 immediately when full to prevent OOM.

All CPU-bound Pillow operations are wrapped in `asyncio.to_thread()`. Many optimizers use `asyncio.gather()` to run independent compression methods concurrently (PNG: pngquant + oxipng, JPEG: jpegli + jpegtran, HEIC/AVIF/JXL: metadata-strip + re-encode).

### Security

Applied per-request via `SecurityMiddleware`. Auth (Bearer token, empty key = dev mode), Redis-backed rate limiting (fail-open design), SSRF validation on all URL fetches (DNS resolution + IP range blocking at each redirect hop), SVG sanitization (strips scripts, event handlers, foreignObject).

## Key Conventions

- **Optimizer pattern**: Try multiple methods, pick the smallest output. See `optimizers/tiff.py` and `optimizers/bmp.py` for the clearest examples.
- **Estimation mirrors optimizers**: Direct-encode BPP helpers must match their optimizer's encoding parameters. When changing quality mappings in an optimizer, update the corresponding `_*_sample_bpp()` helper in `estimation/estimator.py`.
- **Output guarantee**: `_build_result()` ensures the API never returns a file larger than the input.
- **Format detection**: Done by magic bytes in `utils/format_detect.py`, never by file extension or Content-Type header.
- **Benchmark verification**: After changing optimizer or estimation logic, run `python -m bench.run --mode timing --fmt <format>` and check that preset differentiation exists (HIGH > MEDIUM > LOW reduction) and that the new run does not regress vs baseline (`python -m bench.compare baseline.json head.json`).
- **Async discipline**: Wrap CPU-bound work in `asyncio.to_thread()`. Use `asyncio.gather()` for concurrent independent operations.
- **CLI tools via stdin/stdout**: `utils/subprocess_runner.py`'s `run_tool()` pipes bytes through CLI tools — no temp files. Use `allowed_exit_codes` for expected non-zero exits (e.g., pngquant exit 99).

## Code Style

- **Formatter**: Black, line-length 100, Python 3.12
- **Linter**: Ruff with E, F, W, I rules (E501 ignored — handled by Black)
- **Async test framework**: pytest with `pytest-asyncio` (strict mode)
