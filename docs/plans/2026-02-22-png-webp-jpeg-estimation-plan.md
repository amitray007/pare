# PNG/WebP/JPEG Estimation Accuracy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce estimation error for PNG (22% -> <15%), WebP (34% -> <15%), and JPEG (14% -> <10%) by adding direct-encode BPP paths and increasing sample sizes.

**Architecture:** Add format-specific `_png_sample_bpp()` and `_webp_sample_bpp()` helpers in `estimation/estimator.py` (same pattern as the existing JPEG/HEIC/AVIF/JXL helpers), route PNG and WebP through them in `_estimate_by_sample()`, and bump JPEG sample width from 1200 to 1600.

**Tech Stack:** Python, Pillow, asyncio, pytest

---

## Task 1: Increase JPEG sample width to 1600px

**Files:**
- Modify: `estimation/estimator.py:21`

**Step 1: Update constant**

In `estimation/estimator.py`, change line 21 from:

```python
JPEG_SAMPLE_MAX_WIDTH = 1200  # JPEG needs larger samples for accurate BPP scaling
```

To:

```python
JPEG_SAMPLE_MAX_WIDTH = 1600  # JPEG needs larger samples for accurate BPP scaling
```

**Step 2: Run tests**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: All pass

**Step 3: Commit**

```bash
git add estimation/estimator.py
git commit -m "perf: increase JPEG sample width to 1600px for better accuracy"
```

---

## Task 2: Add WebP direct-encode estimation

**Files:**
- Modify: `estimation/estimator.py:128-131` (max_width selection)
- Modify: `estimation/estimator.py:165-194` (add WebP branch)
- Modify: `estimation/estimator.py:343` (add helper after `_jxl_sample_bpp`)

**Step 1: Update max_width selection**

In `_estimate_by_sample()`, update the max_width selection block (currently lines 126-131) to include WebP:

Replace:

```python
    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH
```

With:

```python
    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL, ImageFormat.WEBP):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH
```

**Step 2: Add WebP branch**

After the HEIC/AVIF/JXL branch's `return` (after line 194), add:

```python
    # WebP: direct encode at target quality (same pattern as JPEG/HEIC/AVIF/JXL)
    if fmt == ImageFormat.WEBP:
        output_bpp, method = await asyncio.to_thread(
            _webp_sample_bpp, img, sample_width, sample_height, config
        )
        estimated_size = int(output_bpp * original_pixels / 8)
        estimated_size = min(estimated_size, file_size)

        reduction = round((file_size - estimated_size) / file_size * 100, 1)
        reduction = max(0.0, reduction)

        return EstimateResponse(
            original_size=file_size,
            original_format=fmt.value,
            dimensions={"width": width, "height": height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=estimated_size,
            estimated_reduction_percent=reduction,
            optimization_potential=_classify_potential(reduction),
            method=method,
            already_optimized=reduction == 0,
            confidence="high",
        )
```

**Step 3: Add `_webp_sample_bpp()` helper**

After `_jxl_sample_bpp()` (around line 343), add:

```python
def _webp_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a WebP sample at target quality and return output BPP.

    Matches the WebP optimizer's Pillow path: lossy encode at target quality
    with method=4 for good compression.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA", "L"):
        sample = sample.convert("RGB")

    buf = io.BytesIO()
    sample.save(buf, format="WEBP", quality=config.quality, method=4)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "pillow")
```

**Step 4: Lint and test**

Run: `python -m ruff check estimation/estimator.py && python -m black --check estimation/estimator.py`
Run: `pytest tests/test_sample_estimator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add estimation/estimator.py
git commit -m "fix: add direct-encode estimation for WebP"
```

---

## Task 3: Add PNG direct-encode estimation

**Files:**
- Modify: `estimation/estimator.py:128-131` (max_width selection)
- Modify: `estimation/estimator.py` (add PNG branch after WebP branch)
- Modify: `estimation/estimator.py` (add helper after `_webp_sample_bpp`)

**Step 1: Update max_width selection to include PNG**

Update the max_width selection block to include PNG:

Replace:

```python
    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL, ImageFormat.WEBP):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH
```

With:

```python
    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (
        ImageFormat.HEIC,
        ImageFormat.AVIF,
        ImageFormat.JXL,
        ImageFormat.WEBP,
        ImageFormat.PNG,
    ):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH
```

**Step 2: Add PNG branch**

After the WebP branch's `return`, add:

```python
    # PNG: direct encode to measure achievable compression
    if fmt in (ImageFormat.PNG, ImageFormat.APNG):
        output_bpp, method = await asyncio.to_thread(
            _png_sample_bpp, img, sample_width, sample_height, config
        )
        estimated_size = int(output_bpp * original_pixels / 8)
        estimated_size = min(estimated_size, file_size)

        reduction = round((file_size - estimated_size) / file_size * 100, 1)
        reduction = max(0.0, reduction)

        return EstimateResponse(
            original_size=file_size,
            original_format=fmt.value,
            dimensions={"width": width, "height": height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=estimated_size,
            estimated_reduction_percent=reduction,
            optimization_potential=_classify_potential(reduction),
            method=method,
            already_optimized=reduction == 0,
            confidence="high",
        )
```

**Step 3: Add `_png_sample_bpp()` helper**

After `_webp_sample_bpp()`, add:

```python
def _png_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a PNG sample and return output BPP.

    For lossy mode (quality < 70 with png_lossy=True): quantizes to palette
    first (simulating pngquant), then encodes with maximum compression.
    For lossless mode: encodes directly with maximum compression.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)

    # Lossy path: quantize to palette (simulates pngquant)
    if config.png_lossy and config.quality < 70:
        max_colors = 64 if config.quality < 50 else 256
        if sample.mode == "RGBA":
            # Quantize preserving alpha
            sample = sample.quantize(max_colors)
        elif sample.mode != "P":
            sample = sample.convert("RGB").quantize(max_colors)
        method = "pngquant + oxipng"
    else:
        method = "oxipng"

    buf = io.BytesIO()
    sample.save(buf, format="PNG", compress_level=9)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, method)
```

**Step 4: Lint and test**

Run: `python -m ruff check estimation/estimator.py && python -m black --check estimation/estimator.py`
Run: `pytest tests/test_sample_estimator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add estimation/estimator.py
git commit -m "fix: add direct-encode estimation for PNG"
```

---

## Task 4: Add estimation tests for large PNG and WebP

**Files:**
- Modify: `tests/test_sample_estimator.py`

**Step 1: Add large PNG estimation test**

Add after the existing `test_large_png_extrapolation` test:

```python
@pytest.mark.asyncio
async def test_large_png_screenshot_not_zero():
    """Large PNG screenshot should estimate meaningful reduction, not 0%."""
    # Create a screenshot-like image: large areas of solid color
    img = Image.new("RGB", (1000, 800))
    # Draw colored rectangles to simulate a screenshot
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 1000, 40], fill=(50, 50, 60))       # title bar
    draw.rectangle([0, 40, 200, 800], fill=(240, 240, 240))    # sidebar
    draw.rectangle([200, 40, 1000, 800], fill=(255, 255, 255)) # content
    draw.rectangle([200, 700, 1000, 800], fill=(230, 230, 230))# footer
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=6)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60, png_lossy=True))
    assert result.original_format == "png"
    assert result.estimated_reduction_percent > 0, (
        f"Large PNG screenshot should not estimate 0%, got method={result.method}"
    )


@pytest.mark.asyncio
async def test_large_png_lossless_estimation():
    """Large PNG in lossless mode should still produce a reasonable estimate."""
    img = Image.new("RGB", (800, 600), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)  # uncompressed
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=80, png_lossy=False))
    assert result.original_format == "png"
    # Uncompressed PNG should estimate significant compression
    assert result.estimated_reduction_percent > 0
```

**Step 2: Add large WebP estimation test**

```python
@pytest.mark.asyncio
async def test_large_webp_not_zero():
    """Large WebP should estimate meaningful reduction."""
    raw = os.urandom(800 * 600 * 3)
    img = Image.frombytes("RGB", (800, 600), raw)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=95)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "webp"
    assert result.method != "none", "WebP should not report 'none' method"
    assert result.estimated_reduction_percent > 0, (
        f"Large WebP at q=95 estimated at q=60 should show reduction"
    )
```

**Step 3: Add import for ImageDraw**

Add `from PIL import Image, ImageDraw` at the top of the test file if not already present. Actually, just use it inline as shown in the test (imported locally in the test function).

**Step 4: Run tests**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: All pass

**Step 5: Commit**

```bash
git add tests/test_sample_estimator.py
git commit -m "test: add large PNG and WebP estimation accuracy tests"
```

---

## Task 5: Benchmark and verify

**Step 1: Run full test suite and lint**

Run: `pytest tests/ -q`
Run: `python -m ruff check . && python -m black --check .`
Expected: All pass, lint clean

**Step 2: Run benchmark with comparison**

Run: `python -m benchmarks.run --compare`

Check:
- PNG avg error < 15%
- WebP avg error < 15%
- JPEG avg error < 10%
- No cases report 0% estimation when actual reduction > 20%

**Step 3: Commit and push**

```bash
git push
```

---

## Verification Checklist

```bash
# All tests pass
pytest tests/ -q

# Lint clean
python -m ruff check . && python -m black --check .

# Specific test files
pytest tests/test_sample_estimator.py -v

# Benchmark comparison
python -m benchmarks.run --compare
```
