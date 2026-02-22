# PNG/WebP/JPEG Estimation Accuracy Improvements

## Problem

Benchmark shows poor estimation accuracy for three formats:
- **PNG**: 22.2% avg error, 71.3% max. Large screenshots/graphics estimated at 0% but actually compress 70%+. Root cause: 300px sample encoded with `compress_level=0` goes through optimizer pipeline which returns `method="none"` on tiny samples.
- **WebP**: 33.6% avg error, 88.2% max. 300px lossless sample causes wildly wrong BPP extrapolation for lossy encoding. VP8 block-based encoding at 300px doesn't predict 1920px behavior.
- **JPEG**: 13.6% avg error, 57.3% max. Already has 1200px direct-encode path but could improve with larger samples. Experiments showed 1600px achieves 5.2% error vs 9.5% at 1200px.

## Solution

### 1. PNG: Direct lossless encode with larger samples

Add `_png_sample_bpp()` helper using 800px samples (`LOSSY_SAMPLE_MAX_WIDTH`). For lossless mode (`quality >= 70` or `png_lossy=False`), encode with `compress_level=9`. For lossy mode, quantize to palette first (simulating pngquant), then encode with `compress_level=9`. This eliminates catastrophic 0% estimates without trying to perfectly predict pngquant/oxipng behavior.

**File:** `estimation/estimator.py`

### 2. WebP: Direct encode at target quality

Add `_webp_sample_bpp()` helper using 800px samples. Encode directly at target quality with Pillow `method=4`, matching the optimizer's primary Pillow path. Quality mapping: `webp_quality = max(20, min(90, quality))`.

**File:** `estimation/estimator.py`

### 3. JPEG: Increase sample width to 1600px

Change `JPEG_SAMPLE_MAX_WIDTH` from 1200 to 1600. One constant change, no logic changes.

**File:** `estimation/estimator.py`

### 4. Tests

Add large-image estimation tests for PNG and WebP to verify they no longer report 0% when actual reduction is significant.

**File:** `tests/test_sample_estimator.py`

## Verification

- All tests pass: `pytest tests/ -q`
- Lint clean: `python -m ruff check . && python -m black --check .`
- Benchmark all three formats under 15% avg error
