# Sample-Based Estimation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the heuristic-based estimation engine (~1,650 lines) with a sample-based approach that compresses a downsized sample using the actual optimizers and extrapolates BPP.

**Architecture:** The new estimator downloads the image, downsamples to ~300px wide, runs the real optimizer on the sample, measures output BPP, and scales to the original pixel count. For small images (<150K pixels) and SVG/animated, it compresses the full file for exact results.

**Tech Stack:** Pillow (decode/resize), existing optimizers (compression), httpx (HEAD/Range for large-image thumbnail path)

**Design doc:** `docs/plans/2026-02-21-sample-based-estimation-design.md`

---

### Task 1: Add Preset Mapping

**Files:**
- Create: `estimation/presets.py`
- Test: `tests/test_presets.py`

**Step 1: Write the failing test**

```python
# tests/test_presets.py
"""Tests for preset -> OptimizationConfig mapping."""

from estimation.presets import PRESET_CONFIGS, get_config_for_preset


def test_high_preset_maps_to_quality_40():
    config = get_config_for_preset("high")
    assert config.quality == 40
    assert config.png_lossy is True


def test_medium_preset_maps_to_quality_60():
    config = get_config_for_preset("medium")
    assert config.quality == 60
    assert config.png_lossy is True


def test_low_preset_maps_to_quality_80():
    config = get_config_for_preset("low")
    assert config.quality == 80
    assert config.png_lossy is False


def test_preset_case_insensitive():
    assert get_config_for_preset("HIGH").quality == 40
    assert get_config_for_preset("High").quality == 40


def test_invalid_preset_raises():
    import pytest
    with pytest.raises(ValueError, match="Invalid preset"):
        get_config_for_preset("ultra")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_presets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'estimation.presets'`

**Step 3: Write minimal implementation**

```python
# estimation/presets.py
"""Preset -> OptimizationConfig mapping for the estimation API."""

from schemas import OptimizationConfig

PRESET_CONFIGS = {
    "high": OptimizationConfig(quality=40, png_lossy=True, strip_metadata=True),
    "medium": OptimizationConfig(quality=60, png_lossy=True, strip_metadata=True),
    "low": OptimizationConfig(quality=80, png_lossy=False, strip_metadata=True),
}


def get_config_for_preset(preset: str) -> OptimizationConfig:
    """Convert a preset name to an OptimizationConfig.

    Args:
        preset: "high", "medium", or "low" (case-insensitive).

    Returns:
        OptimizationConfig with appropriate quality and flags.

    Raises:
        ValueError: If preset is not recognized.
    """
    key = preset.lower()
    if key not in PRESET_CONFIGS:
        raise ValueError(
            f"Invalid preset: '{preset}'. Must be 'high', 'medium', or 'low'."
        )
    return PRESET_CONFIGS[key]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_presets.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add estimation/presets.py tests/test_presets.py
git commit -m "feat: add preset mapping for estimation API"
```

---

### Task 2: Write Core Sample-Based Estimator

This is the main task. It replaces the entire estimation pipeline with sample-based compression.

**Files:**
- Rewrite: `estimation/estimator.py`
- Create: `tests/test_sample_estimator.py`

**Step 1: Write the failing tests**

```python
# tests/test_sample_estimator.py
"""Tests for the sample-based estimation engine."""

import io

import pytest
from PIL import Image

from estimation.estimator import estimate, EXACT_PIXEL_THRESHOLD, SAMPLE_MAX_WIDTH
from schemas import OptimizationConfig


def _make_image(fmt: str, width: int, height: int, quality: int = 95, **kwargs) -> bytes:
    """Helper: create a synthetic image in the given format."""
    mode = "RGB"
    if fmt == "PNG" and kwargs.get("rgba"):
        mode = "RGBA"
    img = Image.new(mode, (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality
    if fmt == "PNG":
        save_kwargs["compress_level"] = kwargs.get("compress_level", 6)
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


# --- Exact mode: small images ---


@pytest.mark.asyncio
async def test_small_png_exact_result():
    """Images under EXACT_PIXEL_THRESHOLD are compressed fully (exact)."""
    data = _make_image("PNG", 100, 100)  # 10K pixels, well under threshold
    result = await estimate(data, OptimizationConfig(quality=40, png_lossy=True))
    assert result.original_format == "png"
    assert result.original_size == len(data)
    assert result.estimated_reduction_percent >= 0
    assert result.confidence == "high"
    assert result.dimensions["width"] == 100
    assert result.dimensions["height"] == 100


@pytest.mark.asyncio
async def test_small_jpeg_exact_result():
    """Small JPEG compressed fully."""
    data = _make_image("JPEG", 200, 200, quality=95)  # 40K pixels
    result = await estimate(data, OptimizationConfig(quality=40))
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent > 0  # q95 -> q40 should reduce


@pytest.mark.asyncio
async def test_exact_mode_uses_actual_optimizer():
    """In exact mode, estimated_optimized_size matches what optimizer produces."""
    data = _make_image("JPEG", 100, 100, quality=95)
    config = OptimizationConfig(quality=60)
    result = await estimate(data, config)
    # The estimate should match exactly (it ran the full optimizer)
    # We can't easily verify the exact number, but confidence should be high
    assert result.confidence == "high"
    assert result.estimated_optimized_size <= result.original_size


# --- Extrapolate mode: large images ---


@pytest.mark.asyncio
async def test_large_jpeg_extrapolation():
    """Large JPEG uses sample-based extrapolation."""
    data = _make_image("JPEG", 1000, 1000, quality=95)  # 1M pixels
    result = await estimate(data, OptimizationConfig(quality=40))
    assert result.original_format == "jpeg"
    assert result.estimated_reduction_percent > 0
    assert result.confidence == "high"
    assert result.estimated_optimized_size < result.original_size


@pytest.mark.asyncio
async def test_large_png_extrapolation():
    """Large PNG uses sample-based extrapolation."""
    data = _make_image("PNG", 800, 600)  # 480K pixels
    result = await estimate(data, OptimizationConfig(quality=40, png_lossy=True))
    assert result.original_format == "png"
    assert result.estimated_reduction_percent >= 0
    assert result.estimated_optimized_size <= result.original_size


@pytest.mark.asyncio
async def test_extrapolation_bpp_consistency():
    """BPP should be roughly consistent: estimate for a large image should
    be proportional to the small-image result scaled by pixel count."""
    small_data = _make_image("JPEG", 300, 300, quality=95)
    large_data = _make_image("JPEG", 900, 900, quality=95)
    config = OptimizationConfig(quality=60)

    small_result = await estimate(small_data, config)
    large_result = await estimate(large_data, config)

    # The BPP should be similar (within ~20% for synthetic images)
    small_bpp = small_result.estimated_optimized_size * 8 / (300 * 300)
    large_bpp = large_result.estimated_optimized_size * 8 / (900 * 900)
    assert abs(small_bpp - large_bpp) / max(small_bpp, large_bpp) < 0.25


# --- SVG special case ---


@pytest.mark.asyncio
async def test_svg_compresses_full_file(sample_svg):
    """SVG always compresses the full file (no pixel sampling)."""
    result = await estimate(sample_svg, OptimizationConfig(quality=60))
    assert result.original_format == "svg"
    assert "scour" in result.method
    assert result.confidence == "high"


# --- Default config ---


@pytest.mark.asyncio
async def test_estimate_none_config_uses_defaults():
    """estimate() with config=None uses default OptimizationConfig."""
    data = _make_image("PNG", 100, 100)
    result = await estimate(data, None)
    assert result.original_format == "png"
    assert result.estimated_reduction_percent >= 0


@pytest.mark.asyncio
async def test_estimate_default_config():
    """estimate() with no config uses defaults."""
    data = _make_image("JPEG", 100, 100, quality=95)
    result = await estimate(data)
    assert result.original_format == "jpeg"


# --- Response fields ---


@pytest.mark.asyncio
async def test_response_has_all_fields():
    """EstimateResponse has all required fields."""
    data = _make_image("PNG", 200, 200)
    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_size > 0
    assert result.original_format == "png"
    assert "width" in result.dimensions
    assert "height" in result.dimensions
    assert isinstance(result.estimated_optimized_size, int)
    assert isinstance(result.estimated_reduction_percent, float)
    assert result.optimization_potential in ("high", "medium", "low")
    assert isinstance(result.method, str)
    assert isinstance(result.already_optimized, bool)
    assert result.confidence in ("high", "medium", "low")


# --- Animated images (exact mode) ---


@pytest.mark.asyncio
async def test_animated_gif_uses_exact_mode():
    """Animated GIFs compress the full file, never sample."""
    # Create a 2-frame GIF
    frames = [
        Image.new("P", (400, 400), color=i) for i in range(2)
    ]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:])
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=60))
    assert result.original_format == "gif"
    assert result.confidence == "high"


# --- Edge cases ---


@pytest.mark.asyncio
async def test_already_optimized_image():
    """An image that can't be compressed further reports 0% reduction."""
    # Create a tiny, already-efficient JPEG
    img = Image.new("L", (8, 8), color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=20)
    data = buf.getvalue()

    result = await estimate(data, OptimizationConfig(quality=80))
    # quality=80 is higher than source quality=20, so little/no reduction expected
    assert result.estimated_reduction_percent >= 0
    assert result.estimated_optimized_size <= result.original_size
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: FAIL — the current estimator.py imports from heuristics, doesn't have `SAMPLE_MAX_WIDTH` or `EXACT_PIXEL_THRESHOLD`

**Step 3: Rewrite `estimation/estimator.py`**

```python
# estimation/estimator.py
"""Sample-based estimation engine.

Instead of heuristic prediction, this module compresses a downsized sample
of the image using the actual optimizers and extrapolates BPP (bits per pixel)
to the full image size.

For small images (<150K pixels), SVG, and animated formats, it compresses the
full file for an exact result.
"""

import asyncio
import io

from PIL import Image

from optimizers.router import optimize_image
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

SAMPLE_MAX_WIDTH = 300
EXACT_PIXEL_THRESHOLD = 150_000  # ~390x390 pixels


async def estimate(
    data: bytes,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate compression savings by compressing a sample.

    For small images, SVGs, and animated images: compresses the full file.
    For large raster images: downsamples to ~300px wide, compresses the
    sample, and extrapolates output BPP to the original pixel count.
    """
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(data)
    file_size = len(data)

    # SVG/SVGZ: no pixel data — compress the whole file
    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return await _estimate_exact(data, fmt, config, file_size)

    # Decode image for dimensions and animation detection
    img = await asyncio.to_thread(_open_image, data)
    width, height = img.size
    original_pixels = width * height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Animated images: compress full file (inter-frame redundancy matters)
    frame_count = getattr(img, "n_frames", 1)
    if frame_count > 1:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Small images: compress fully for exact result
    if original_pixels <= EXACT_PIXEL_THRESHOLD:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Large raster images: downsample + compress sample + extrapolate BPP
    return await _estimate_by_sample(
        data, img, fmt, config, file_size, width, height, color_type, bit_depth
    )


def _open_image(data: bytes) -> Image.Image:
    """Open image in Pillow (lazy decode — reads header only)."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


async def _estimate_exact(
    data: bytes,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int = 0,
    height: int = 0,
    color_type: str | None = None,
    bit_depth: int | None = None,
) -> EstimateResponse:
    """Compress the full image with the actual optimizer. Returns exact result."""
    result = await optimize_image(data, config)
    already_optimized = result.method == "none"
    reduction = result.reduction_percent if not already_optimized else 0.0

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=result.optimized_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=already_optimized,
        confidence="high",
    )


async def _estimate_by_sample(
    data: bytes,
    img: Image.Image,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
) -> EstimateResponse:
    """Downsample to ~300px wide, compress sample, extrapolate BPP."""
    original_pixels = width * height

    # Proportional resize
    ratio = SAMPLE_MAX_WIDTH / width
    sample_width = SAMPLE_MAX_WIDTH
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

    # Create sample encoded with minimal compression
    sample_data = await asyncio.to_thread(
        _create_sample, img, sample_width, sample_height, fmt
    )

    # Compress sample with the actual optimizer
    result = await optimize_image(sample_data, config)

    # If optimizer says "already optimized", propagate that
    if result.method == "none":
        return EstimateResponse(
            original_size=file_size,
            original_format=fmt.value,
            dimensions={"width": width, "height": height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="high",
        )

    # Extrapolate output BPP to original pixel count
    sample_output_bpp = result.optimized_size * 8 / sample_pixels
    estimated_size = int(sample_output_bpp * original_pixels / 8)
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
        method=result.method,
        already_optimized=reduction == 0,
        confidence="high",
    )


def _create_sample(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    fmt: ImageFormat,
) -> bytes:
    """Resize image and encode with minimal compression.

    Minimal compression ensures the optimizer always has room to work,
    preventing false "already optimized" results on the sample.
    """
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    buf = io.BytesIO()

    if fmt == ImageFormat.JPEG:
        if sample.mode not in ("RGB", "L"):
            sample = sample.convert("RGB")
        sample.save(buf, format="JPEG", quality=100)
    elif fmt in (ImageFormat.PNG, ImageFormat.APNG):
        sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.WEBP:
        sample.save(buf, format="WEBP", lossless=True)
    elif fmt == ImageFormat.GIF:
        if sample.mode != "P":
            sample = sample.quantize(256)
        sample.save(buf, format="GIF")
    elif fmt == ImageFormat.TIFF:
        sample.save(buf, format="TIFF", compression="raw")
    elif fmt == ImageFormat.BMP:
        if sample.mode not in ("RGB", "L", "P"):
            sample = sample.convert("RGB")
        sample.save(buf, format="BMP")
    elif fmt == ImageFormat.AVIF:
        try:
            sample.save(buf, format="AVIF", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.HEIC:
        try:
            sample.save(buf, format="HEIF", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    elif fmt == ImageFormat.JXL:
        try:
            sample.save(buf, format="JXL", quality=100)
        except Exception:
            sample.save(buf, format="PNG", compress_level=0)
    else:
        sample.save(buf, format="PNG", compress_level=0)

    return buf.getvalue()


def _classify_potential(reduction: float) -> str:
    """Classify reduction percentage into potential category."""
    if reduction >= 30:
        return "high"
    elif reduction >= 10:
        return "medium"
    return "low"


def _get_color_type(img: Image.Image) -> str | None:
    """Map Pillow mode to color type string."""
    return {
        "RGB": "rgb",
        "RGBA": "rgba",
        "P": "palette",
        "L": "grayscale",
        "LA": "grayscale",
        "1": "grayscale",
    }.get(img.mode)


def _get_bit_depth(img: Image.Image) -> int | None:
    """Extract bit depth from Pillow image."""
    if img.mode == "1":
        return 1
    return img.info.get("bits") or 8
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sample_estimator.py -v`
Expected: All tests PASS

**Step 5: Verify existing endpoint tests still pass**

Run: `pytest tests/test_estimate.py -v`
Expected: All tests PASS (response schema unchanged)

**Step 6: Commit**

```bash
git add estimation/estimator.py tests/test_sample_estimator.py
git commit -m "feat: replace heuristic estimation with sample-based compression"
```

---

### Task 3: Update Estimate Endpoint for JSON Preset + URL Fields

**Files:**
- Modify: `routers/estimate.py`
- Modify: `schemas.py`
- Test: `tests/test_estimate.py` (add new tests)

**Step 1: Write the failing tests**

Add to `tests/test_estimate.py`:

```python
# --- New tests for preset-based JSON estimation ---


def test_estimate_json_with_preset(client, sample_png):
    """JSON body with preset returns valid estimate."""
    import base64
    # For now, test with file upload + preset form field
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", sample_png, "image/png")},
        data={"preset": "high"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["original_format"] == "png"
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_preset_overrides_options(client, sample_jpeg):
    """When preset is provided, it takes precedence over options."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.jpg", sample_jpeg, "image/jpeg")},
        data={"preset": "high"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["estimated_reduction_percent"] >= 0


def test_estimate_invalid_preset(client, sample_png):
    """Invalid preset returns 400."""
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", sample_png, "image/png")},
        data={"preset": "ultra"},
    )
    assert resp.status_code == 400
```

**Step 2: Run to verify failures**

Run: `pytest tests/test_estimate.py::test_estimate_json_with_preset tests/test_estimate.py::test_estimate_invalid_preset -v`
Expected: FAIL — endpoint doesn't accept `preset` field yet

**Step 3: Update schemas and endpoint**

In `schemas.py`, add the `EstimateRequest` model after `OptimizeRequest`:

```python
class EstimateRequest(BaseModel):
    """JSON body for URL-based estimation."""

    url: str
    preset: str = Field(default="medium", pattern=r"^(high|medium|low)$")
    thumbnail_url: Optional[str] = None
    file_size: Optional[int] = Field(default=None, gt=0)
```

In `routers/estimate.py`, update the endpoint:

```python
import json

from fastapi import APIRouter, File, Form, Request, UploadFile

from config import settings
from estimation.estimator import estimate as run_estimate
from estimation.presets import get_config_for_preset
from exceptions import BadRequestError, FileTooLargeError
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import detect_format
from utils.url_fetch import fetch_image

router = APIRouter()

LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(
    request: Request,
    file: UploadFile | None = File(None),
    options: str | None = Form(None),
    preset: str | None = Form(None),
):
    """Estimate compression savings without full optimization.

    Accepts multipart file upload or JSON with URL.
    Optionally accepts a preset (high/medium/low) instead of options.
    """
    content_type = request.headers.get("content-type", "")

    # --- Determine config from preset or options ---
    config = OptimizationConfig()
    if preset:
        try:
            config = get_config_for_preset(preset)
        except ValueError as e:
            raise BadRequestError(str(e))
    elif options:
        try:
            config = OptimizationConfig(**json.loads(options))
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Get image data ---
    if file is not None:
        data = await file.read()
    elif "application/json" in content_type:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise BadRequestError(f"Invalid JSON body: {e}")

        url = body.get("url")
        if not url:
            raise BadRequestError("Missing 'url' field in JSON body")

        # Use preset from JSON body if provided
        if not preset and body.get("preset"):
            try:
                config = get_config_for_preset(body["preset"])
            except ValueError as e:
                raise BadRequestError(str(e))

        is_authenticated = getattr(request.state, "is_authenticated", False)
        data = await fetch_image(url, is_authenticated=is_authenticated)
    else:
        raise BadRequestError("Expected multipart/form-data or application/json")

    # Validate file size
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds limit of {settings.max_file_size_mb} MB",
            file_size=len(data),
            limit=settings.max_file_size_bytes,
        )

    # Validate format
    detect_format(data)

    return await run_estimate(data, config)
```

**Step 4: Run tests**

Run: `pytest tests/test_estimate.py -v`
Expected: All tests PASS (old + new)

**Step 5: Commit**

```bash
git add schemas.py routers/estimate.py tests/test_estimate.py
git commit -m "feat: add preset support to /estimate endpoint"
```

---

### Task 4: Delete Old Heuristic Code and Tests

Now that the new estimator is working, remove the old code.

**Files:**
- Delete: `estimation/header_analysis.py`
- Delete: `estimation/heuristics.py`
- Delete: `tests/test_heuristics.py`
- Delete: `tests/test_heuristics_extra.py`
- Delete: `tests/test_heuristics_probes.py`
- Delete: `tests/test_header_analysis.py`
- Delete: `tests/test_header_analysis_extra.py`
- Delete: `tests/test_header_probes.py`
- Delete: `tests/test_estimator.py` (old unit tests for heuristic-based estimator)
- Update: `estimation/CLAUDE.md`

**Step 1: Verify no imports remain**

Run: `grep -r "from estimation.header_analysis" --include="*.py" .`
Run: `grep -r "from estimation.heuristics" --include="*.py" .`

Expected: Only hits in the files being deleted. If the new `estimator.py` doesn't import them (it shouldn't), proceed.

**Step 2: Delete files**

```bash
git rm estimation/header_analysis.py
git rm estimation/heuristics.py
git rm tests/test_heuristics.py
git rm tests/test_heuristics_extra.py
git rm tests/test_heuristics_probes.py
git rm tests/test_header_analysis.py
git rm tests/test_header_analysis_extra.py
git rm tests/test_header_probes.py
git rm tests/test_estimator.py
```

**Step 3: Update `estimation/CLAUDE.md`**

Replace contents with:

```markdown
# CLAUDE.md

## Purpose

Sample-based compression estimation. Instead of heuristic prediction, this module compresses a downsized sample of the image using the actual optimizers and extrapolates BPP (bits per pixel) to the full image size.

## Architecture

Two files:

1. **`estimator.py`** — Entry point. Downloads/receives image, determines whether to use exact mode (small/SVG/animated) or extrapolation mode (large raster). Calls the real optimizers via `optimize_image()`.

2. **`presets.py`** — Maps preset names (high/medium/low) to `OptimizationConfig` instances.

## Key Design Property

Estimation calls the actual optimizers. When optimizer logic changes, estimation automatically adapts. No parallel heuristic system to maintain.

## Modes

- **Exact mode** (<150K pixels, SVG, animated): Compresses the full file with the real optimizer. 100% accurate.
- **Extrapolation mode** (>150K pixels): Downsamples to ~300px wide, compresses sample, measures output BPP, scales to original pixel count.

## Verification

After changes to optimizers, run:
```
python -m benchmarks.run --fmt <format>
```
Check that estimation accuracy (Avg Err) stays under ~10%.
```

**Step 4: Run all tests to verify nothing breaks**

Run: `pytest tests/ -v`
Expected: All remaining tests PASS. The deleted test files are gone.

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove heuristic estimation code (~1,650 lines)

Deletes header_analysis.py, heuristics.py, and their tests.
The sample-based estimator in estimator.py replaces all of this."
```

---

### Task 5: Add Large-Image Thumbnail Path (>=10MB)

For images >= 10MB where a `thumbnail_url` is provided, download only the thumbnail and a small Range of the original for dimensions.

**Files:**
- Modify: `routers/estimate.py`
- Modify: `estimation/estimator.py` (add `estimate_from_thumbnail`)
- Create: `tests/test_thumbnail_estimation.py`

**Step 1: Write the failing tests**

```python
# tests/test_thumbnail_estimation.py
"""Tests for the large-image thumbnail estimation path."""

import io

import pytest
from PIL import Image

from estimation.estimator import estimate_from_thumbnail
from schemas import OptimizationConfig


def _make_jpeg(width: int, height: int, quality: int = 95) -> bytes:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_thumbnail_estimation_returns_valid_result():
    """Thumbnail-based estimation returns a valid EstimateResponse."""
    # Simulate: original is 3000x2000 JPEG, thumbnail is 300x200
    original_data = _make_jpeg(3000, 2000, quality=95)
    thumbnail_data = _make_jpeg(300, 200, quality=95)

    result = await estimate_from_thumbnail(
        thumbnail_data=thumbnail_data,
        original_file_size=len(original_data),
        original_width=3000,
        original_height=2000,
        config=OptimizationConfig(quality=40),
    )
    assert result.original_format == "jpeg"
    assert result.original_size == len(original_data)
    assert result.dimensions == {"width": 3000, "height": 2000}
    assert result.estimated_reduction_percent > 0
    assert result.estimated_optimized_size < result.original_size


@pytest.mark.asyncio
async def test_thumbnail_estimation_accuracy():
    """Thumbnail estimate should be within 20% of actual compression."""
    from optimizers.router import optimize_image

    original_data = _make_jpeg(1000, 1000, quality=95)
    thumbnail_data = _make_jpeg(300, 300, quality=95)
    config = OptimizationConfig(quality=40)

    # Get actual compression result
    actual = await optimize_image(original_data, config)

    # Get thumbnail-based estimate
    estimate_result = await estimate_from_thumbnail(
        thumbnail_data=thumbnail_data,
        original_file_size=len(original_data),
        original_width=1000,
        original_height=1000,
        config=config,
    )

    # Check accuracy: within 20% of actual
    error = abs(
        estimate_result.estimated_optimized_size - actual.optimized_size
    ) / actual.optimized_size * 100
    assert error < 20, f"Estimation error {error:.1f}% exceeds 20% threshold"
```

**Step 2: Run to verify failure**

Run: `pytest tests/test_thumbnail_estimation.py -v`
Expected: FAIL — `estimate_from_thumbnail` not defined

**Step 3: Add `estimate_from_thumbnail` to estimator.py**

Add this function to `estimation/estimator.py`:

```python
async def estimate_from_thumbnail(
    thumbnail_data: bytes,
    original_file_size: int,
    original_width: int,
    original_height: int,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate using a pre-downsized thumbnail (for large images).

    Used when the original image is >= 10MB and a CDN thumbnail is available.
    The thumbnail is compressed with the actual optimizer and BPP is
    extrapolated to the original dimensions.

    Args:
        thumbnail_data: Raw bytes of the thumbnail image.
        original_file_size: Size of the original image in bytes.
        original_width: Original image width in pixels.
        original_height: Original image height in pixels.
        config: Optimization parameters.
    """
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(thumbnail_data)
    original_pixels = original_width * original_height

    # Decode thumbnail for pixel count
    img = await asyncio.to_thread(_open_image, thumbnail_data)
    thumb_width, thumb_height = img.size
    thumb_pixels = thumb_width * thumb_height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Compress thumbnail with actual optimizer
    result = await optimize_image(thumbnail_data, config)

    if result.method == "none":
        return EstimateResponse(
            original_size=original_file_size,
            original_format=fmt.value,
            dimensions={"width": original_width, "height": original_height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=original_file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="medium",
        )

    # Extrapolate BPP
    thumb_output_bpp = result.optimized_size * 8 / thumb_pixels
    estimated_size = int(thumb_output_bpp * original_pixels / 8)
    estimated_size = min(estimated_size, original_file_size)

    reduction = round(
        (original_file_size - estimated_size) / original_file_size * 100, 1
    )
    reduction = max(0.0, reduction)

    return EstimateResponse(
        original_size=original_file_size,
        original_format=fmt.value,
        dimensions={"width": original_width, "height": original_height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=reduction == 0,
        confidence="medium",  # CDN thumbnail may have re-compression artifacts
    )
```

**Step 4: Update endpoint to use thumbnail path**

In `routers/estimate.py`, update the JSON body handling to check for `thumbnail_url` and `file_size`:

```python
    elif "application/json" in content_type:
        try:
            body = await request.json()
        except (ValueError, json.JSONDecodeError) as e:
            raise BadRequestError(f"Invalid JSON body: {e}")

        url = body.get("url")
        if not url:
            raise BadRequestError("Missing 'url' field in JSON body")

        # Use preset from JSON body if provided
        if not preset and body.get("preset"):
            try:
                config = get_config_for_preset(body["preset"])
            except ValueError as e:
                raise BadRequestError(str(e))

        is_authenticated = getattr(request.state, "is_authenticated", False)

        # Large-image thumbnail path
        thumbnail_url = body.get("thumbnail_url")
        client_file_size = body.get("file_size")

        if thumbnail_url and client_file_size and client_file_size >= LARGE_FILE_THRESHOLD:
            from estimation.estimator import estimate_from_thumbnail

            thumbnail_data = await fetch_image(thumbnail_url, is_authenticated=is_authenticated)
            detect_format(thumbnail_data)

            # Get original dimensions via Range request (first 4KB)
            original_width, original_height = await _fetch_dimensions(
                url, is_authenticated
            )

            return await estimate_from_thumbnail(
                thumbnail_data=thumbnail_data,
                original_file_size=client_file_size,
                original_width=original_width,
                original_height=original_height,
                config=config,
            )

        data = await fetch_image(url, is_authenticated=is_authenticated)
```

Add the `_fetch_dimensions` helper at the bottom of `routers/estimate.py`:

```python
async def _fetch_dimensions(url: str, is_authenticated: bool) -> tuple[int, int]:
    """Fetch just enough of the image to parse dimensions.

    Downloads first 8KB via Range request, parses with Pillow.
    Falls back to full download if Range not supported.
    """
    import httpx
    from PIL import Image

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            resp = await client.get(url, headers={"Range": "bytes=0-8191"})
            partial = resp.content
            img = Image.open(io.BytesIO(partial))
            return img.size
    except Exception:
        # Fallback: download full image just for dimensions
        from utils.url_fetch import fetch_image
        data = await fetch_image(url, is_authenticated=is_authenticated)
        img = Image.open(io.BytesIO(data))
        return img.size
```

Add `import io` to the imports if not already present.

**Step 5: Run tests**

Run: `pytest tests/test_thumbnail_estimation.py tests/test_estimate.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
git add estimation/estimator.py routers/estimate.py tests/test_thumbnail_estimation.py
git commit -m "feat: add thumbnail-based estimation for large images (>=10MB)"
```

---

### Task 6: Run Full Test Suite and Fix Breakages

**Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`

Check for any failures from:
- Tests that imported `_thumbnail_compress` or `_combine_with_thumbnail` from old estimator
- Tests that imported `HeaderInfo` or `Prediction` from deleted modules
- Benchmark imports that reference deleted modules

**Step 2: Fix any remaining imports**

Check benchmark files for references to deleted modules:

Run: `grep -r "header_analysis\|heuristics\|HeaderInfo\|Prediction" benchmarks/ --include="*.py"`

If the benchmark runner imports from `estimation.header_analysis` or `estimation.heuristics`, update those imports. The benchmark likely needs updating to work with the new estimator (Task 7).

**Step 3: Run tests again**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve remaining imports after heuristic code removal"
```

---

### Task 7: Update Benchmarks for Sample-Based Estimation

The benchmark system needs to call the new `estimate()` instead of the old one. Since `estimate()` has the same signature (`data, config -> EstimateResponse`), the benchmarks should mostly work, but verify.

**Files:**
- Check: `benchmarks/runner.py` (imports from estimation)
- Check: `benchmarks/report.py`
- Check: `benchmarks/cases.py`

**Step 1: Check benchmark imports**

Run: `grep -r "from estimation" benchmarks/ --include="*.py"`

Update any references to `header_analysis`, `heuristics`, `HeaderInfo`, or `Prediction`.

**Step 2: Run benchmarks for a quick format**

Run: `python -m benchmarks.run --fmt bmp --preset high`

Check:
- Benchmark completes without errors
- "ESTIMATION ACCURACY" section shows Avg Err values
- Avg Err should be significantly lower than before

**Step 3: Run benchmarks for all formats**

Run: `python -m benchmarks.run`

Check:
- All formats complete
- Avg Err < 10% for most formats
- No crashes or import errors

**Step 4: Commit any benchmark updates**

```bash
git add benchmarks/
git commit -m "fix: update benchmarks for sample-based estimation"
```

---

### Task 8: Final Verification and Cleanup

**Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 2: Verify the server starts**

Run: `uvicorn main:app --port 8080 &`
Then: `curl -X POST http://localhost:8080/estimate -F "file=@tests/sample_images/sample.jpg" -F "preset=high"`

Check: Valid JSON response with estimate.

**Step 3: Check for dead code**

Run: `grep -r "flat_pixel_ratio\|unique_color_ratio\|svg_bloat_ratio\|analyze_header\|predict_reduction" --include="*.py" .`

Remove any remaining references to deleted concepts.

**Step 4: Update root CLAUDE.md**

Update the "Estimation Engine" section in `CLAUDE.md` to reflect the new architecture:

Replace the current description:
```
### Estimation Engine (3-layer, no optimization)
...
```

With:
```
### Estimation Engine (sample-based)

`routers/estimate.py` -> `estimation/estimator.py` -> `optimizers/router.py`

Estimates compression by compressing a downsized sample (~300px wide) with the actual optimizer and extrapolating BPP to the full image size. For small images (<150K pixels), SVG, and animated formats, compresses the full file for exact results.

Accepts presets (HIGH/MEDIUM/LOW) mapped to quality levels in `estimation/presets.py`. For images >= 10MB, supports an optional `thumbnail_url` to avoid downloading the full original.
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "docs: update CLAUDE.md for sample-based estimation architecture"
```
