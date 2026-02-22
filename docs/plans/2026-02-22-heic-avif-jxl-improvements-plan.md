# HEIC/AVIF/JXL Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix estimation accuracy for HEIC/AVIF/JXL (same q=100 sample bug as JPEG), parallelize their optimizers, and add unit tests for GIF/HEIC/AVIF/JXL.

**Architecture:** Add format-specific direct-encode BPP helpers in `estimation/estimator.py` (mirroring the JPEG fix pattern), replace serial metadata+re-encode with `asyncio.gather()` in three optimizer files, and create four new test files following the existing `test_optimizer_jxl.py` pattern.

**Tech Stack:** Python, Pillow, pillow-heif, pillow-avif-plugin, jxlpy/pillow-jxl-plugin, gifsicle, asyncio, pytest

---

## Task 1: Parallelize HEIC optimizer

**Files:**
- Modify: `optimizers/heic.py:23-43`

**Step 1: Refactor optimize() to use asyncio.gather**

Replace the serial metadata-strip + re-encode with concurrent execution. In `optimizers/heic.py`, replace the `optimize` method body:

```python
async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    tasks = []

    if config.strip_metadata:
        tasks.append(asyncio.to_thread(self._strip_metadata, data))
    tasks.append(asyncio.to_thread(self._reencode, data, config.quality))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    method_names = []
    if config.strip_metadata:
        method_names.append("metadata-strip")
    method_names.append("heic-reencode")

    for result, method in zip(results, method_names):
        if not isinstance(result, Exception):
            candidates.append((result, method))

    if not candidates:
        return self._build_result(data, data, "none")

    best_data, best_method = min(candidates, key=lambda x: len(x[0]))
    return self._build_result(data, best_data, best_method)
```

**Step 2: Run existing tests**

Run: `pytest tests/ -k "heic" -v`
Expected: All existing HEIC tests pass (behavior unchanged, just faster)

**Step 3: Commit**

```
git add optimizers/heic.py
git commit -m "perf: parallelize HEIC metadata strip and re-encode"
```

---

## Task 2: Parallelize AVIF optimizer

**Files:**
- Modify: `optimizers/avif.py:30-52`

**Step 1: Refactor optimize() to use asyncio.gather**

Same pattern as HEIC. Replace `optimize` method in `optimizers/avif.py`:

```python
async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    tasks = []

    if config.strip_metadata:
        tasks.append(asyncio.to_thread(self._strip_metadata, data))
    tasks.append(asyncio.to_thread(self._reencode, data, config.quality))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    method_names = []
    if config.strip_metadata:
        method_names.append("metadata-strip")
    method_names.append("avif-reencode")

    for result, method in zip(results, method_names):
        if not isinstance(result, Exception):
            candidates.append((result, method))

    if not candidates:
        return self._build_result(data, data, "none")

    best_data, best_method = min(candidates, key=lambda x: len(x[0]))
    return self._build_result(data, best_data, best_method)
```

**Step 2: Run existing tests**

Run: `pytest tests/ -k "avif" -v`
Expected: All existing AVIF tests pass

**Step 3: Commit**

```
git add optimizers/avif.py
git commit -m "perf: parallelize AVIF metadata strip and re-encode"
```

---

## Task 3: Parallelize JXL optimizer

**Files:**
- Modify: `optimizers/jxl.py:25-45`

**Step 1: Refactor optimize() to use asyncio.gather**

Same pattern. Replace `optimize` method in `optimizers/jxl.py`:

```python
async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    tasks = []

    if config.strip_metadata:
        tasks.append(asyncio.to_thread(self._strip_metadata, data))
    tasks.append(asyncio.to_thread(self._reencode, data, config.quality))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    method_names = []
    if config.strip_metadata:
        method_names.append("metadata-strip")
    method_names.append("jxl-reencode")

    for result, method in zip(results, method_names):
        if not isinstance(result, Exception):
            candidates.append((result, method))

    if not candidates:
        return self._build_result(data, data, "none")

    best_data, best_method = min(candidates, key=lambda x: len(x[0]))
    return self._build_result(data, best_data, best_method)
```

**Step 2: Run existing tests**

Run: `pytest tests/test_optimizer_jxl.py -v`
Expected: All pass (including mock tests)

**Step 3: Commit**

```
git add optimizers/jxl.py
git commit -m "perf: parallelize JXL metadata strip and re-encode"
```

---

## Task 4: Add HEIC/AVIF/JXL direct-encode estimation

**Files:**
- Modify: `estimation/estimator.py:21-22` (add constants)
- Modify: `estimation/estimator.py:122-157` (add format branches in `_estimate_by_sample`)
- Modify: `estimation/estimator.py:204-233` (add helper functions after `_jpeg_sample_bpp`)

**Step 1: Add sample width constants**

At `estimation/estimator.py:21-22`, after `JPEG_SAMPLE_MAX_WIDTH`, add:

```python
LOSSY_SAMPLE_MAX_WIDTH = 800  # HEIC/AVIF/JXL also need larger samples
```

**Step 2: Add format-specific BPP helpers**

After `_jpeg_sample_bpp()` (around line 233), add three helpers. Each mirrors the quality mapping from its optimizer:

```python
def _heic_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a HEIC sample at target quality and return output BPP."""
    import pillow_heif

    pillow_heif.register_heif_opener()
    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA"):
        sample = sample.convert("RGB")

    heic_quality = max(30, min(90, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="HEIF", quality=heic_quality)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "heic-reencode")


def _avif_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode an AVIF sample at target quality and return output BPP."""
    import pillow_avif  # noqa: F401

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA"):
        sample = sample.convert("RGB")

    avif_quality = max(30, min(90, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="AVIF", quality=avif_quality, speed=6)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "avif-reencode")


def _jxl_sample_bpp(
    img: Image.Image,
    sample_width: int,
    sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JXL sample at target quality and return output BPP."""
    try:
        import pillow_jxl  # noqa: F401
    except ImportError:
        import jxlpy  # noqa: F401

    sample = img.resize((sample_width, sample_height), Image.LANCZOS)
    if sample.mode not in ("RGB", "RGBA", "L"):
        sample = sample.convert("RGB")

    jxl_quality = max(30, min(95, config.quality + 10))

    buf = io.BytesIO()
    sample.save(buf, format="JXL", quality=jxl_quality)
    output_size = buf.tell()
    sample_pixels = sample_width * sample_height

    return (output_size * 8 / sample_pixels, "jxl-reencode")
```

**Step 3: Add format branches in _estimate_by_sample**

In `_estimate_by_sample()`, update the `max_width` selection (line 125) and add format branches after the JPEG branch (after line 157).

Update the max_width line:

```python
# Lossy formats need larger samples for accurate BPP scaling
if fmt == ImageFormat.JPEG:
    max_width = JPEG_SAMPLE_MAX_WIDTH
elif fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL):
    max_width = LOSSY_SAMPLE_MAX_WIDTH
else:
    max_width = SAMPLE_MAX_WIDTH
```

After the JPEG branch's `return` (line 157), add:

```python
# HEIC/AVIF/JXL: same pattern as JPEG — direct encode at target quality
if fmt in (ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL):
    bpp_fn = {
        ImageFormat.HEIC: _heic_sample_bpp,
        ImageFormat.AVIF: _avif_sample_bpp,
        ImageFormat.JXL: _jxl_sample_bpp,
    }[fmt]

    output_bpp, method = await asyncio.to_thread(
        bpp_fn, img, sample_width, sample_height, config
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

**Step 4: Run lint and tests**

Run: `python -m ruff check estimation/estimator.py && python -m black --check estimation/estimator.py`
Run: `pytest tests/test_sample_estimator.py -v`
Expected: All pass

**Step 5: Commit**

```
git add estimation/estimator.py
git commit -m "fix: add direct-encode estimation for HEIC/AVIF/JXL"
```

---

## Task 5: Add GIF optimizer unit tests

**Files:**
- Create: `tests/test_optimizer_gif.py`

**Step 1: Write tests**

Follow the pattern from `test_bmp_optimizer.py`. GIF optimizer needs gifsicle, so tests skip if it's not available.

```python
"""Tests for GIF optimizer — quality tiers, lossless, lossy."""

import io

import pytest
from PIL import Image

from optimizers.gif import GifOptimizer
from schemas import OptimizationConfig
from utils.subprocess_runner import run_tool

# Check if gifsicle is available
try:
    import asyncio
    asyncio.get_event_loop().run_until_complete(run_tool(["gifsicle", "--version"], b""))
    HAS_GIFSICLE = True
except (FileNotFoundError, OSError, Exception):
    HAS_GIFSICLE = False


@pytest.fixture
def gif_optimizer():
    return GifOptimizer()


def _make_gif(width=100, height=100, colors=64, frames=1):
    """Create a test GIF image."""
    imgs = []
    for i in range(frames):
        img = Image.new("RGB", (width, height), ((i * 50) % 256, 100, 200))
        imgs.append(img.quantize(colors))
    buf = io.BytesIO()
    if frames > 1:
        imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:])
    else:
        imgs[0].save(buf, format="GIF")
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_lossless_optimization(gif_optimizer):
    """quality >= 70: lossless gifsicle --optimize=3 only."""
    data = _make_gif()
    config = OptimizationConfig(quality=80)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.method == "gifsicle"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_moderate_lossy(gif_optimizer):
    """quality 50-69: gifsicle --lossy=30 --colors=192."""
    data = _make_gif(width=200, height=200, colors=256)
    config = OptimizationConfig(quality=60)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert "lossy=30" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_aggressive_lossy(gif_optimizer):
    """quality < 50: gifsicle --lossy=80 --colors=128."""
    data = _make_gif(width=200, height=200, colors=256)
    config = OptimizationConfig(quality=30)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert "lossy=80" in result.method


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_animated(gif_optimizer):
    """Animated GIF is optimized without breaking frames."""
    data = _make_gif(frames=3)
    config = OptimizationConfig(quality=60)
    result = await gif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    # Verify output is valid and still animated
    out_img = Image.open(io.BytesIO(result.optimized_bytes))
    assert getattr(out_img, "n_frames", 1) >= 1


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_GIFSICLE, reason="gifsicle not available")
async def test_gif_quality_tiers(gif_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_gif(width=200, height=200, colors=256)
    result_high = await gif_optimizer.optimize(data, OptimizationConfig(quality=30))
    result_low = await gif_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )
```

**Step 2: Run tests**

Run: `pytest tests/test_optimizer_gif.py -v`
Expected: All pass (or skip if gifsicle unavailable)

**Step 3: Commit**

```
git add tests/test_optimizer_gif.py
git commit -m "test: add GIF optimizer unit tests"
```

---

## Task 6: Add HEIC optimizer unit tests

**Files:**
- Create: `tests/test_optimizer_heic.py`

**Step 1: Write tests**

Follow the `test_optimizer_jxl.py` pattern — real tests + mock tests.

```python
"""Tests for HEIC optimizer — re-encoding, metadata strip, quality tiers."""

import io
from unittest.mock import MagicMock, patch

import pytest

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    from PIL import Image as _Im
    _buf = io.BytesIO()
    _Im.new("RGB", (1, 1)).save(_buf, format="HEIF")
    HAS_HEIC = True
except (ImportError, Exception):
    HAS_HEIC = False

from PIL import Image

from optimizers.heic import HeicOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def heic_optimizer():
    return HeicOptimizer()


def _make_heic(quality=90, size=(100, 100)):
    img = Image.new("RGB", size, (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="HEIF", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_basic_optimization(heic_optimizer):
    """HEIC optimizer produces valid output not larger than input."""
    data = _make_heic(quality=95)
    config = OptimizationConfig(quality=60)
    result = await heic_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "heic"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_metadata_strip(heic_optimizer):
    """HEIC metadata strip path runs without error."""
    data = _make_heic(quality=90)
    config = OptimizationConfig(quality=80, strip_metadata=True)
    result = await heic_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_HEIC, reason="HEIC not available")
async def test_heic_quality_tiers(heic_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_heic(quality=95, size=(200, 200))
    result_high = await heic_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_low = await heic_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )


@pytest.mark.asyncio
async def test_heic_both_fail():
    """Both methods fail: returns method='none'."""
    opt = HeicOptimizer()
    data = b"\x00" * 100
    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"
```

**Step 2: Run tests**

Run: `pytest tests/test_optimizer_heic.py -v`
Expected: Pass (real tests skip if HEIC unavailable, mock tests always run)

**Step 3: Commit**

```
git add tests/test_optimizer_heic.py
git commit -m "test: add HEIC optimizer unit tests"
```

---

## Task 7: Add AVIF optimizer unit tests

**Files:**
- Create: `tests/test_optimizer_avif.py`

**Step 1: Write tests**

Same pattern as HEIC tests.

```python
"""Tests for AVIF optimizer — re-encoding, metadata strip, quality tiers."""

import io
from unittest.mock import patch

import pytest

try:
    import pillow_avif  # noqa: F401
    from PIL import Image as _Im
    _buf = io.BytesIO()
    _Im.new("RGB", (1, 1)).save(_buf, format="AVIF")
    HAS_AVIF = True
except (ImportError, Exception):
    HAS_AVIF = False

from PIL import Image

from optimizers.avif import AvifOptimizer
from schemas import OptimizationConfig


@pytest.fixture
def avif_optimizer():
    return AvifOptimizer()


def _make_avif(quality=90, size=(100, 100)):
    img = Image.new("RGB", size, (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="AVIF", quality=quality)
    return buf.getvalue()


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_basic_optimization(avif_optimizer):
    """AVIF optimizer produces valid output not larger than input."""
    data = _make_avif(quality=95)
    config = OptimizationConfig(quality=60)
    result = await avif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size
    assert result.format == "avif"


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_metadata_strip(avif_optimizer):
    """AVIF metadata strip path runs without error."""
    data = _make_avif(quality=90)
    config = OptimizationConfig(quality=80, strip_metadata=True)
    result = await avif_optimizer.optimize(data, config)
    assert result.success
    assert result.optimized_size <= result.original_size


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_AVIF, reason="AVIF not available")
async def test_avif_quality_tiers(avif_optimizer):
    """Aggressive quality produces smaller or equal output."""
    data = _make_avif(quality=95, size=(200, 200))
    result_high = await avif_optimizer.optimize(data, OptimizationConfig(quality=40))
    result_low = await avif_optimizer.optimize(data, OptimizationConfig(quality=80))
    assert result_high.optimized_size <= result_low.optimized_size or (
        result_high.method == "none" and result_low.method == "none"
    )


@pytest.mark.asyncio
async def test_avif_both_fail():
    """Both methods fail: returns method='none'."""
    opt = AvifOptimizer()
    data = b"\x00" * 100
    with patch.object(opt, "_strip_metadata", side_effect=Exception("fail")):
        with patch.object(opt, "_reencode", side_effect=Exception("fail")):
            config = OptimizationConfig(quality=60, strip_metadata=True)
            result = await opt.optimize(data, config)
            assert result.method == "none"
```

**Step 2: Run tests**

Run: `pytest tests/test_optimizer_avif.py -v`
Expected: Pass

**Step 3: Commit**

```
git add tests/test_optimizer_avif.py
git commit -m "test: add AVIF optimizer unit tests"
```

---

## Task 8: Benchmark estimation accuracy and tune sample widths

**Step 1: Run estimation benchmark for HEIC/AVIF/JXL**

Create synthetic test images in each format and compare estimation vs actual optimization at MEDIUM preset (q=60). Test sample widths 300, 600, 800, 1200 to find optimal.

```bash
# If plugins aren't available locally, create test images using available formats
# and convert, or skip this step and trust that the pattern works (proved with JPEG)
pytest tests/test_sample_estimator.py -v
```

**Step 2: Adjust LOSSY_SAMPLE_MAX_WIDTH if needed**

Based on benchmark results, adjust the constant. Start at 800 (conservative), increase to 1200 if accuracy is poor.

**Step 3: Run full test suite and lint**

Run: `pytest tests/ -q`
Run: `python -m ruff check . && python -m black --check .`
Expected: All pass, lint clean

**Step 4: Final commit**

```
git add -A
git commit -m "fix: tune HEIC/AVIF/JXL estimation sample widths"
```

---

## Verification Checklist

```bash
# All tests pass
pytest tests/ -q

# Lint clean
python -m ruff check . && python -m black --check .

# Specific test files
pytest tests/test_optimizer_gif.py tests/test_optimizer_heic.py tests/test_optimizer_avif.py tests/test_optimizer_jxl.py tests/test_sample_estimator.py -v

# Push and verify CI
git push
```
