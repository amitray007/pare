# HEIC/AVIF/JXL Estimation + Optimizer Improvements

## Problem

HEIC, AVIF, and JXL estimation likely suffers from the same accuracy issue we fixed for JPEG: `_create_sample()` encodes at `quality=100`, leaving no room for the optimizer on small 300px samples. The optimizer's output-never-larger gate returns `method="none"`, reporting 0% reduction.

Additionally, these three optimizers run metadata-strip and re-encode serially, unlike PNG/JPEG/WebP which use `asyncio.gather()`.

Finally, GIF/HEIC/AVIF/JXL optimizers lack dedicated unit tests.

## Solution

### 1. Estimation: Direct Pillow encode for HEIC/AVIF/JXL

Same pattern as JPEG fix — bypass the optimizer pipeline, encode samples directly at target quality using Pillow plugins (pillow-heif, pillow-avif-plugin, jxlpy). Use larger sample widths (benchmark to find optimal per format).

**Files:** `estimation/estimator.py`
- Add format-specific sample BPP helpers
- Add format branches in `_estimate_by_sample()` before the generic path
- Benchmark and tune sample widths per format

### 2. Optimizer parallelization

Replace serial metadata+re-encode with `asyncio.gather()` in HEIC, AVIF, and JXL optimizers.

**Files:** `optimizers/heic.py`, `optimizers/avif.py`, `optimizers/jxl.py`

### 3. Unit tests

Add dedicated test files for GIF, HEIC, AVIF, JXL optimizers. Each tests: basic optimization, quality tiers, metadata stripping, mode conversion. Tests skip if required plugins aren't installed.

**Files:** `tests/test_optimizer_gif.py`, `tests/test_optimizer_heic.py`, `tests/test_optimizer_avif.py`, `tests/test_optimizer_jxl.py`

## Verification

- All tests pass: `pytest tests/ -q`
- Lint clean: `python -m ruff check . && python -m black --check .`
- Benchmark estimation accuracy for HEIC/AVIF/JXL < 15% avg error
