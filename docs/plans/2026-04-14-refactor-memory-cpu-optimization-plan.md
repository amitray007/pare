---
title: "refactor: Memory & CPU Optimization"
type: refactor
status: completed
date: 2026-04-14
origin: docs/brainstorms/2026-04-14-memory-cpu-optimization-brainstorm.md
---

# refactor: Memory & CPU Optimization

## Overview

Pare's image compression API (8GB memory, 8 cores on Cloud Run) OOM-kills on 10MB files due to memory amplification — a single request creates 5-20x the compressed file size in RAM from duplicate PIL Image copies, parallel compression branches, and subprocess buffers. This plan reduces peak per-request memory by ~60-70%, adds decompression bomb protection, and right-sizes concurrency to reliably handle files up to the 32MB limit.

## Problem Statement

A single 10MB image optimization consumes 86-184MB of RAM depending on format. With 4 Uvicorn workers x 8 concurrent slots = 32 possible simultaneous optimizations, total memory demand easily exceeds 8GB. The container is OOM-killed within ~5 seconds.

| Format | Current Peak (10MB input) | Primary Cause |
|--------|--------------------------|---------------|
| TIFF | ~184MB | 3x `img.copy()` for parallel compression |
| PNG | ~105MB | pngquant + oxipng parallel + second oxipng pass |
| JPEG | ~86MB | Pillow decode + encode + jpegtran stdin/stdout |
| WebP | ~80MB | Double `Image.open()` + temp files |
| AVIF/HEIC/JXL | ~80MB | `img.copy()` for strip + img for reencode |

Additionally: no decompression bomb protection, no pixel count limits, estimate endpoint has unbounded concurrency, and there is no memory-aware admission control.

## Proposed Solution

Four-phase implementation ordered by risk/impact ratio:

1. **Infrastructure & Safety** — Single worker, pixel limits, error handling (low risk, highest impact)
2. **Memory Guards** — Decompressed size validation, estimate semaphore, memory-aware gate (medium risk, high impact)
3. **Optimizer Memory Reduction** — Eliminate `img.copy()`, fix double-decode (medium risk, high impact)
4. **CPU Efficiency** — Skip redundant lossy branches (low risk, moderate impact)

**Dropped from brainstorm:** BytesIO zero-copy (`bytes(buf.getbuffer())` vs `buf.getvalue()`). SpecFlow analysis confirmed both allocate identically in CPython — no actual memory savings. (see brainstorm: Decision 6)

## Technical Approach

### Architecture

The changes touch three layers:

```
Request → [Pixel Limit] → [Decompressed Size Check] → [Memory-Aware Gate] → Optimizer → Response
                                                              ↓
                                                    [Format-specific memory reduction]
                                                    - TIFF: sequential compression
                                                    - AVIF/HEIC/JXL: sequential strip+reencode
                                                    - WebP: decode-once
```

### Validation Order (per request)

```
1. File size check          (existing — config.max_file_size_bytes)         → 413
2. Format detection         (existing — detect_format magic bytes)          → 415
3. Pixel count limit        (NEW — Image.MAX_IMAGE_PIXELS via Pillow)      → 413
4. Decompressed size check  (NEW — width*height*bpp after header parse)    → 413
5. Memory-aware gate        (NEW — estimated memory vs budget)             → 503
6. Optimization             (existing — optimizer pipeline)
7. Memory-aware gate release
```

Steps 3-4 happen inside the optimizer (after `Image.open()` header parse, before `img.load()`). Step 5 wraps the optimizer dispatch in the router.

### Implementation Phases

---

#### Phase 1: Infrastructure & Safety Guards

Lowest risk, biggest impact. Can be deployed independently.

##### 1a. Single Uvicorn Worker

**Files:** `Dockerfile`, `Dockerfile.jxl`, `config.py`

Change the default worker count from 4 to 1. Cloud Run handles horizontal scaling via container instances — multiple in-process workers fragment memory without benefit. (see brainstorm: Decision 1)

```python
# Dockerfile — change default
# Before:
CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} ..."]
# After:
CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-1} ..."]
```

```python
# config.py — update default to match
workers: int = 1
```

##### 1b. Cloud Run Concurrency

**Files:** `cloudbuild.yaml`

Add `--concurrency` to the deploy step. With 1 worker and 8 semaphore slots + 16 queue depth, the container can handle 24 concurrent requests. Set Cloud Run concurrency to match so it doesn't send more than the container can handle.

```yaml
# cloudbuild.yaml — deploy step, add:
- '--concurrency=24'
```

Also add `--timeout=300` to ensure Cloud Run allows up to 5 minutes per request (the default is 300s but making it explicit prevents surprises).

##### 1c. Pixel Count Limit

**Files:** `config.py`, `exceptions.py`, `middleware.py`

Set `Image.MAX_IMAGE_PIXELS` early — in `config.py` after imports, since this module is imported before any optimizer or estimator module.

```python
# config.py — add after imports, before Settings class
from PIL import Image
Image.MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels
```

Add a new exception for decompression bombs:

```python
# exceptions.py — add
class ImageTooLargeError(PareError):
    """Image dimensions exceed maximum allowed pixel count."""
    status_code = 413
    error_code = "image_too_large"
```

Catch Pillow's `DecompressionBombError` in the middleware so it returns a clean 413 instead of a 500:

```python
# middleware.py — add import and catch
from PIL import Image as _PIL_Image

# In dispatch(), expand the except block:
except _PIL_Image.DecompressionBombError:
    response = JSONResponse(
        status_code=413,
        content={
            "success": False,
            "error": "image_too_large",
            "message": "Image exceeds maximum pixel count (100 megapixels)",
        },
    )
except PareError as exc:
    # ... existing handler ...
```

##### 1d. Add max_image_pixels to config

**Files:** `config.py`

Make the pixel limit configurable:

```python
# config.py — add to Settings
max_image_pixels: int = 100_000_000  # 100MP

def model_post_init(self, __context) -> None:
    # ... existing post-init ...
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = self.max_image_pixels
```

**Phase 1 tests:**
- [x] `tests/test_memory_guards.py`: Test that a small file (e.g., 100x100) passes pixel limit
- [x] `tests/test_memory_guards.py`: Test that a crafted image exceeding 100MP returns 413 with `image_too_large` error code
- [x] `tests/test_memory_guards.py`: Verify error response format matches PareError JSON structure
- [ ] Verify `WORKERS=1` works locally via `docker-compose up`

**Phase 1 acceptance criteria:**
- [ ] Dockerfile defaults to `WORKERS=1`
- [ ] `cloudbuild.yaml` deploy sets `--concurrency=24`
- [ ] `Image.MAX_IMAGE_PIXELS` is set before any Pillow usage
- [ ] `DecompressionBombError` returns 413, not 500
- [ ] Pixel limit is configurable via `MAX_IMAGE_PIXELS` env var

---

#### Phase 2: Memory Guards

Adds pre-optimization validation and admission control.

##### 2a. Decompressed Size Validation

**Files:** `utils/image_validation.py` (new), `optimizers/router.py`

Add a validation utility that checks decompressed size after Pillow's lazy header parse (before `img.load()` forces full decompression). This is the defense-in-depth layer that catches images that pass the pixel limit but would still use too much memory.

```python
# utils/image_validation.py (new file)
import io
from PIL import Image
from config import settings
from exceptions import ImageTooLargeError

# Bytes per pixel by PIL mode
_BPP = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4,
        "YCbCr": 3, "LAB": 3, "I": 4, "F": 4, "LA": 2, "PA": 2,
        "RGBa": 4, "I;16": 2}

MAX_DECOMPRESSED_BYTES = 512 * 1024 * 1024  # 512MB
MAX_FRAME_COUNT = 500

def validate_image_dimensions(data: bytes) -> None:
    """Peek at image dimensions and reject if decompressed size is too large.

    Uses Pillow's lazy loading — Image.open() reads headers without
    decompressing pixel data. Safe to call before img.load().
    """
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        return  # Let format detection handle invalid files

    width, height = img.size
    bpp = _BPP.get(img.mode, 4)  # Default to 4 (RGBA) if unknown
    n_frames = getattr(img, "n_frames", 1)
    n_frames = min(n_frames, MAX_FRAME_COUNT + 1)  # Cap iteration

    decompressed = width * height * bpp * n_frames

    if n_frames > MAX_FRAME_COUNT:
        raise ImageTooLargeError(
            f"Animated image has {n_frames} frames, maximum is {MAX_FRAME_COUNT}",
            frames=n_frames,
            limit=MAX_FRAME_COUNT,
        )

    if decompressed > MAX_DECOMPRESSED_BYTES:
        raise ImageTooLargeError(
            f"Decompressed image size ({decompressed // (1024*1024)}MB) "
            f"exceeds limit ({MAX_DECOMPRESSED_BYTES // (1024*1024)}MB)",
            decompressed_size=decompressed,
            limit=MAX_DECOMPRESSED_BYTES,
        )
```

Call this in both routers, after format detection but before optimization:

```python
# routers/optimize.py — add after detect_format(data):
from utils.image_validation import validate_image_dimensions
validate_image_dimensions(data)

# routers/estimate.py — same location
validate_image_dimensions(data)
```

Note: SVG/SVGZ files are text-based and don't go through Pillow — `Image.open()` will fail on them, which is caught by the `except Exception: return` fallback. No special handling needed.

##### 2b. Estimate Endpoint Semaphore

**Files:** `utils/concurrency.py`, `routers/estimate.py`, `config.py`

Add a separate, lighter semaphore for `/estimate`. Since estimates are faster (~50-500ms vs seconds for optimization), the semaphore can be larger.

```python
# config.py — add
estimate_semaphore_size: int = 0  # 0 = 2x compression_semaphore_size
estimate_queue_depth: int = 0     # 0 = 2x estimate_semaphore_size

# model_post_init — add
if self.estimate_semaphore_size == 0:
    self.estimate_semaphore_size = 2 * self.compression_semaphore_size
if self.estimate_queue_depth == 0:
    self.estimate_queue_depth = 2 * self.estimate_semaphore_size
```

```python
# utils/concurrency.py — add second gate instance
estimate_gate = CompressionGate(
    semaphore_size=settings.estimate_semaphore_size,
    max_queue=settings.estimate_queue_depth,
)
```

This requires refactoring `CompressionGate.__init__` to accept parameters instead of reading from `settings` directly:

```python
class CompressionGate:
    def __init__(self, semaphore_size: int | None = None, max_queue: int | None = None):
        size = semaphore_size or settings.compression_semaphore_size
        self._semaphore = asyncio.Semaphore(size)
        self._queue_depth = 0
        self._max_queue = max_queue or settings.max_queue_depth
        self._semaphore_size = size
        self._lock = asyncio.Lock()
```

```python
# routers/estimate.py — wrap the estimate call
from utils.concurrency import estimate_gate

# In the estimate() handler, before `return await run_estimate(data, config)`:
await estimate_gate.acquire()
try:
    return await run_estimate(data, config)
finally:
    estimate_gate.release()
```

##### 2c. Memory-Aware Concurrency Gate

**Files:** `utils/concurrency.py`, `routers/optimize.py`, `config.py`

Augment `CompressionGate` with memory budget tracking. Instead of counting slots, track estimated memory per request. Use format-specific multipliers derived from the audit.

```python
# config.py — add
memory_budget_mb: int = 0  # 0 = auto (total_memory * 0.75, or 6144)
```

```python
# utils/concurrency.py — add memory tracking to CompressionGate

# Format-specific memory multipliers (compressed_size -> estimated peak memory)
# Derived from audit: TIFF worst at ~6x after sequential optimization,
# PNG at ~5x, JPEG at ~4.5x, others at ~4x
MEMORY_MULTIPLIERS = {
    "tiff": 6, "png": 5, "jpeg": 4, "webp": 4,
    "avif": 4, "heic": 4, "jxl": 4,
    "bmp": 3, "gif": 2, "svg": 1, "svgz": 1,
    "apng": 5,
}

class CompressionGate:
    def __init__(self, semaphore_size=None, max_queue=None, memory_budget_bytes=None):
        size = semaphore_size or settings.compression_semaphore_size
        self._semaphore = asyncio.Semaphore(size)
        self._queue_depth = 0
        self._max_queue = max_queue or settings.max_queue_depth
        self._semaphore_size = size
        self._lock = asyncio.Lock()
        self._memory_used = 0
        self._memory_budget = memory_budget_bytes or (settings.memory_budget_mb * 1024 * 1024)

    async def acquire(self, estimated_memory: int = 0):
        async with self._lock:
            if self._queue_depth >= self._max_queue:
                raise BackpressureError("Compression queue full.", retry_after=5)
            if estimated_memory > 0 and self._memory_used + estimated_memory > self._memory_budget:
                raise BackpressureError(
                    "Memory budget exceeded. Try again shortly.", retry_after=5
                )
            self._queue_depth += 1
            self._memory_used += estimated_memory
        await self._semaphore.acquire()

    def release(self, estimated_memory: int = 0):
        self._semaphore.release()
        self._queue_depth -= 1
        self._memory_used -= estimated_memory
```

Update the router to pass memory estimates:

```python
# routers/optimize.py — update compression_gate usage
from utils.concurrency import compression_gate, MEMORY_MULTIPLIERS

fmt = detect_format(data)
multiplier = MEMORY_MULTIPLIERS.get(fmt.value, 4)
estimated_memory = len(data) * multiplier

await compression_gate.acquire(estimated_memory=estimated_memory)
try:
    result = await optimize_image(data, opt_config)
finally:
    compression_gate.release(estimated_memory=estimated_memory)
```

Note: The format detection (`detect_format`) already runs before optimization — we reuse its result for the memory estimate and pass `data` directly to `optimize_image` (which calls `detect_format` again internally, but it's fast magic-byte checking).

**Phase 2 tests:**
- [x] `tests/test_memory_guards.py`: Decompressed size check rejects image with dimensions exceeding 512MB decompressed
- [x] `tests/test_memory_guards.py`: Frame count limit rejects animated image with >500 frames
- [x] `tests/test_memory_guards.py`: SVG passes decompressed size check (Image.open fails gracefully)
- [x] `tests/test_memory_guards.py`: Estimate endpoint returns 503 when estimate semaphore is full
- [x] `tests/test_memory_guards.py`: Memory-aware gate rejects when budget would be exceeded
- [x] `tests/test_memory_guards.py`: Memory tracking decrements correctly on release
- [x] `tests/test_memory_guards.py`: Backward-compatible — `acquire()` without `estimated_memory` still works

**Phase 2 acceptance criteria:**
- [ ] Images decompressing to >512MB are rejected with 413
- [ ] Animated images with >500 frames are rejected with 413
- [ ] `/estimate` has its own semaphore, returns 503 when full
- [ ] Memory-aware gate tracks estimated memory and rejects when budget exceeded
- [ ] Memory budget is configurable via `MEMORY_BUDGET_MB` env var

---

#### Phase 3: Per-Optimizer Memory Reduction

Reduces per-request peak memory by eliminating redundant data copies.

##### 3a. TIFF: Sequential for Large Images

**Files:** `optimizers/tiff.py`

Replace `asyncio.gather` with `img.copy()` for each method with a pixel-count threshold. Below the threshold, keep parallel (fast path for small images). Above it, run sequentially on the shared `img` object (no copies needed since single-threaded access). (see brainstorm: Decision 2)

```python
# optimizers/tiff.py — update optimize()
PARALLEL_PIXEL_THRESHOLD = 5_000_000  # 5MP

async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    if config.strip_metadata:
        data = strip_metadata_selective(data, ImageFormat.TIFF)

    img, exif_bytes, icc_profile = await asyncio.to_thread(self._decode, data)

    methods = ["tiff_adobe_deflate", "tiff_lzw"]
    if config.quality < 70 and img.mode in ("RGB", "L"):
        methods.append("tiff_jpeg")

    pixel_count = img.size[0] * img.size[1]

    if pixel_count < PARALLEL_PIXEL_THRESHOLD:
        # Small image: parallel with copies (fast, memory is negligible)
        results = await asyncio.gather(
            *[
                asyncio.to_thread(
                    self._try_compression, img.copy(), compression, config,
                    exif_bytes, icc_profile
                )
                for compression in methods
            ]
        )
    else:
        # Large image: sequential on shared img (no copies, saves memory)
        results = []
        for compression in methods:
            result = await asyncio.to_thread(
                self._try_compression, img, compression, config,
                exif_bytes, icc_profile
            )
            results.append(result)

    best, best_method = data, "none"
    for candidate, method in results:
        if candidate is not None and len(candidate) < len(best):
            best, best_method = candidate, method

    return self._build_result(data, best, best_method)
```

**Thread safety note:** In the sequential path, `img.save()` in `_try_compression` reads from the internal pixel array but does not mutate pixel data. Each call writes to its own `BytesIO` buffer. Sequential execution on the same `img` is safe because only one thread accesses the Image at a time.

**Memory impact:** For a 12MP RGB TIFF (10MB compressed):
- Before: 3 x `img.copy()` = 3 x 36MB = 108MB in copies alone
- After: 0 copies, shared img = 36MB total
- Savings: ~72MB per request

##### 3b. PillowReencode (AVIF/HEIC/JXL): Sequential for Large Images

**Files:** `optimizers/pillow_reencode.py`

Same threshold pattern. The strip task currently gets `img.copy()`, reencode gets `img`. For large images, run sequentially.

```python
# optimizers/pillow_reencode.py — update optimize()
PARALLEL_PIXEL_THRESHOLD = 5_000_000  # 5MP

async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    img = await asyncio.to_thread(self._open_image, data)

    pixel_count = img.size[0] * img.size[1]
    use_parallel = pixel_count < PARALLEL_PIXEL_THRESHOLD

    if use_parallel:
        # Small image: parallel with copy (current behavior)
        tasks = []
        method_names = []
        if config.strip_metadata:
            tasks.append(asyncio.to_thread(self._strip_metadata_from_img, img.copy(), data))
            method_names.append(self.strip_method_name)
        tasks.append(asyncio.to_thread(self._reencode_from_img, img, config.quality))
        method_names.append(self.reencode_method_name)
        results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Large image: sequential, no copy needed
        results = []
        method_names = []
        if config.strip_metadata:
            try:
                strip_result = await asyncio.to_thread(
                    self._strip_metadata_from_img, img, data
                )
                results.append(strip_result)
            except Exception as e:
                results.append(e)
            method_names.append(self.strip_method_name)

        try:
            reencode_result = await asyncio.to_thread(
                self._reencode_from_img, img, config.quality
            )
            results.append(reencode_result)
        except Exception as e:
            results.append(e)
        method_names.append(self.reencode_method_name)

    candidates = []
    for result, method in zip(results, method_names):
        if isinstance(result, BaseException):
            logger.warning("%s method %s failed: %s: %s",
                          self.pillow_format, method, type(result).__name__, result)
        else:
            candidates.append((result, method))

    if not candidates:
        return self._build_result(data, data, "none")

    best_data, best_method = min(candidates, key=lambda x: len(x[0]))
    return self._build_result(data, best_data, best_method)
```

**Thread safety note:** `_strip_metadata_from_img` calls `img.save()` which reads pixel data. `_reencode_from_img` also calls `img.save()`. In sequential execution, these never overlap. Pillow's `.save()` does not mutate the pixel buffer — it reads from it and writes to the output BytesIO. The `img.info` dict is read but not modified by either method. Sequential execution on the shared `img` is safe.

**Memory impact:** For a 12MP RGBA HEIC (10MB compressed):
- Before: `img.copy()` = 48MB extra
- After: 0 copies
- Savings: ~48MB per request

##### 3c. WebP: Decode Once

**Files:** `optimizers/webp.py`

Currently `_pillow_optimize` (line 69) and `_find_capped_quality` (line 53) each call `Image.open()`. Decode once in `optimize()` and pass the Image to both.

```python
# optimizers/webp.py — refactor to decode once
async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
    # Decode once, share across all paths
    img, is_animated = await asyncio.to_thread(self._decode_image, data)

    pillow_task = asyncio.to_thread(
        self._encode_webp, img, config.quality, is_animated
    )
    cwebp_task = self._cwebp_fallback(data, config.quality)

    pillow_result, cwebp_result = await asyncio.gather(pillow_task, cwebp_task)

    best = pillow_result
    method = "pillow"
    if cwebp_result and len(cwebp_result) < len(best):
        best = cwebp_result
        method = "cwebp"

    if config.max_reduction is not None:
        reduction = (1 - len(best) / len(data)) * 100
        if reduction > config.max_reduction:
            capped = await asyncio.to_thread(
                self._find_capped_quality_from_img, img, is_animated, data, config
            )
            if capped is not None:
                best = capped
                method = "pillow"

    return self._build_result(data, best, method)

def _decode_image(self, data: bytes) -> tuple[Image.Image, bool]:
    """Decode WebP once. Returns (img, is_animated)."""
    img = Image.open(io.BytesIO(data))
    is_animated = getattr(img, "n_frames", 1) > 1
    return img, is_animated

def _find_capped_quality_from_img(
    self, img: Image.Image, is_animated: bool,
    data: bytes, config: OptimizationConfig
) -> bytes | None:
    """Binary search quality using pre-decoded image."""
    def encode_fn(quality: int) -> bytes:
        return self._encode_webp(img, quality, is_animated)
    return binary_search_quality(
        encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
    )
```

Remove the now-unused `_pillow_optimize` and `_find_capped_quality` methods.

**Thread safety note:** `_encode_webp` is a `@staticmethod` that calls `img.save()`. The `pillow_task` and `cwebp_task` run concurrently, but `cwebp_task` only uses raw `data` bytes (not the Image object). The `_find_capped_quality_from_img` runs after `gather` completes, so there's no concurrent access to `img`. Safe.

**Memory impact:**
- Before: 2x `Image.open()` = 2x decompressed pixel buffer
- After: 1x decode
- Savings: ~36-48MB per request (when max_reduction triggers binary search)

**Phase 3 tests:**
- [ ] `tests/test_optimizer_tiff.py`: TIFF optimization of large image (>5MP) produces same quality results as parallel path
- [ ] `tests/test_optimizer_tiff.py`: TIFF optimization of small image (<5MP) still runs parallel
- [ ] `tests/test_pillow_reencode.py`: PillowReencode sequential path produces same results as parallel
- [ ] `tests/test_optimizer_webp.py`: WebP decode-once produces identical output to current behavior
- [ ] `tests/test_formats.py`: All existing format tests still pass (regression check)
- [ ] Benchmark: `python -m benchmarks.run` — verify no regression in compression quality

**Phase 3 acceptance criteria:**
- [ ] TIFF optimizer uses 0 `img.copy()` calls for images >5MP
- [ ] PillowReencode uses 0 `img.copy()` calls for images >5MP
- [ ] WebP calls `Image.open()` exactly once per request
- [ ] All existing tests pass
- [ ] Benchmark results show equivalent compression ratios

---

#### Phase 4: CPU Efficiency

Skip unnecessary work for lossless presets.

##### 4a. PillowReencode: Skip Lossy Reencode for quality >= 70

**Files:** `optimizers/pillow_reencode.py`

For `quality >= 70` (lossless preset), the reencode path produces a lossy output at mapped quality ~80-90. If `strip_metadata` is true, the strip path (lossless re-encode) is almost always better. Skip the lossy reencode entirely for `quality >= 70` when strip is enabled.

```python
# In optimize(), add before building tasks:
skip_reencode = config.quality >= 70 and config.strip_metadata
```

If `skip_reencode`, only run the strip task. If `strip_metadata` is false and `quality >= 70`, still run reencode (it's the only method available).

**Note:** Most other optimizers already skip lossy branches for `quality >= 70`:
- TIFF: skips `tiff_jpeg` at `tiff.py:39`
- PNG: skips pngquant, uses oxipng-only at `png.py:55`
- BMP: skips palette quantization at `bmp.py:65`
- GIF: drops `--lossy` and `--colors` flags

No changes needed for those — they already implement this pattern.

##### 4b. JPEG: Consider Skipping Pillow Encode for quality >= 70

**Files:** `optimizers/jpeg.py`

For `quality >= 70`, jpegtran (lossless Huffman optimization) is almost always the winner since it doesn't re-encode at all. The Pillow/jpegli lossy encode at q=70+ usually produces a larger file than the original with slightly better Huffman tables. Skipping the Pillow encode saves CPU time and one encode buffer.

However, this is a product decision — there are cases where jpegli at q=80 produces a smaller file than the original. **Flag this as optional** and measure via benchmarks before implementing.

**Phase 4 tests:**
- [ ] `tests/test_pillow_reencode.py`: For quality=80 + strip_metadata=True, verify only strip method runs
- [ ] `tests/test_pillow_reencode.py`: For quality=80 + strip_metadata=False, verify reencode still runs

**Phase 4 acceptance criteria:**
- [ ] PillowReencode formats skip lossy reencode for quality >= 70 when strip is available
- [ ] Benchmark shows no regression in compression ratios for lossless presets

---

## System-Wide Impact

### Interaction Graph

Request → SecurityMiddleware (auth, rate limit) → Router (file validation, format detection, **NEW: decompressed size validation**, **NEW: memory-aware gate acquire**) → optimizer_router.optimize_image → format-specific optimizer (**MODIFIED: sequential for large images**) → _build_result → Router (**NEW: memory-aware gate release**) → Response

The estimate endpoint follows the same path but uses a separate, lighter semaphore. Estimate exact mode calls `optimize_image` internally — it should contribute to the memory budget via the memory-aware gate inside the optimizer router (not in the estimate router, since the gate wraps the optimizer dispatch, not the endpoint).

### Error Propagation

New error class `ImageTooLargeError` (413) propagates through:
1. `utils/image_validation.py` raises it
2. `middleware.py` catches `PareError` subclass, formats JSON response
3. `DecompressionBombError` from Pillow is caught separately in middleware (not a PareError subclass)

New `BackpressureError` paths from memory-aware gate use the existing `BackpressureError` class (503), same error flow.

### State Lifecycle Risks

- **Memory tracking drift:** If `acquire()` is called with `estimated_memory=X` but `release()` is called with a different value (or 0), the memory tracker drifts. Mitigation: the router must store the estimated value and pass it to both `acquire` and `release`. A context manager pattern would be safer but the existing acquire/release pattern is established.
- **Estimate exact mode memory:** When `estimate_from_thumbnail` or `_estimate_exact` calls `optimize_image`, it bypasses the memory-aware gate in the router. To address this, move the gate wrapping into `optimize_image` itself or accept that estimate-exact is rare (only for small files <150K pixels) and its memory impact is negligible.

### API Surface Parity

No API changes. All new rejections use existing error response format. New error codes: `image_too_large` (413). New `BackpressureError` message for memory budget ("Memory budget exceeded") uses existing 503 format.

### Integration Test Scenarios

1. **Large TIFF through full pipeline:** Upload a 10MB TIFF via `/optimize`, verify it completes without OOM and returns valid compressed output
2. **Concurrent load test:** Send 8 simultaneous 5MB JPEG requests, verify all complete (no 503 from memory gate under normal load)
3. **Memory gate rejection:** Configure low `MEMORY_BUDGET_MB=100`, send a 10MB file, verify 503 response
4. **Estimate with backpressure:** Fill estimate semaphore, verify 503 response, verify recovery after completion
5. **Decompression bomb:** Craft a small PNG with extreme pixel dimensions (e.g., 10000x10000 pixels in a 1MB file), verify 413 rejection

## Acceptance Criteria

### Functional Requirements

- [ ] 10MB files of all supported formats optimize successfully on 8GB/8-core Cloud Run
- [ ] 32MB files optimize successfully under low concurrency (1-2 concurrent requests)
- [ ] Decompression bombs (small file, huge pixel dimensions) are rejected with 413
- [ ] Animated images with excessive frames are rejected with 413
- [ ] Memory-aware gate prevents OOM under concurrent load
- [ ] Estimate endpoint has bounded concurrency
- [ ] All existing tests pass without modification
- [ ] API response format unchanged for all successful and error cases

### Non-Functional Requirements

- [ ] Peak per-request memory for 10MB TIFF: < 80MB (was ~184MB)
- [ ] Peak per-request memory for 10MB JPEG: < 50MB (was ~86MB)
- [ ] Compression quality: no regression (verified via benchmarks)
- [ ] Latency: < 20% increase for large images due to sequential execution
- [ ] No new Python dependencies

### Quality Gates

- [ ] `pytest tests/` passes (including new tests)
- [ ] `python -m ruff check . && python -m black --check .` passes
- [ ] `python -m benchmarks.run` shows no compression regression
- [ ] Manual test: upload 10MB TIFF, 10MB JPEG, 10MB PNG to deployed service

## Dependencies & Prerequisites

- No external dependencies
- No database changes
- No API contract changes
- Requires Cloud Run deployment config update (`--concurrency`, `WORKERS`)

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Sequential execution is too slow | Low | Medium | 5MP threshold keeps parallel for small images; benchmarks verify latency |
| Memory multipliers are inaccurate | Medium | Low | Multipliers are conservative (over-estimate); budget leaves 2GB headroom |
| Pillow thread safety issue in sequential path | Low | High | `img.save()` does not mutate pixel buffer; sequential access ensures no concurrent access |
| Cloud Run concurrency mismatch | Medium | Medium | Explicit `--concurrency=24` in deploy config; matches semaphore + queue depth |
| Estimate exact mode bypasses memory gate | Low | Low | Only triggers for tiny files (<150K pixels); negligible memory impact |

## Future Considerations

- **Per-format memory budgets:** Track actual memory (via `tracemalloc`) instead of estimated multipliers. More accurate but more complex.
- **Streaming responses:** Return optimized bytes as a stream instead of buffering the full output. Reduces response memory but changes API semantics.
- **Temp file fallback:** For very large images (>20MP), decode to temp file instead of RAM. Adds disk I/O but eliminates memory pressure.
- **PIL memory-mapping:** Use Pillow's memory-mapped storage for large images. Requires Pillow configuration changes.

## Sources & References

### Origin

- **Brainstorm document:** [docs/brainstorms/2026-04-14-memory-cpu-optimization-brainstorm.md](docs/brainstorms/2026-04-14-memory-cpu-optimization-brainstorm.md) — Key decisions carried forward: single worker, eliminate img.copy(), pixel count limit, memory-aware concurrency, estimate endpoint guard, decompressed size validation, skip redundant lossy work. BytesIO zero-copy dropped after SpecFlow analysis found no actual memory savings.

### Internal References

- Concurrency gate: `utils/concurrency.py:7-51`
- TIFF img.copy: `optimizers/tiff.py:42-49`
- PillowReencode img.copy: `optimizers/pillow_reencode.py:78`
- WebP double decode: `optimizers/webp.py:53,69`
- Middleware error handling: `middleware.py:40-49`
- Exception hierarchy: `exceptions.py:1-80`
- Config settings: `config.py:6-59`
- Optimize router gate: `routers/optimize.py:75-79`
- Estimate no gate: `routers/estimate.py:106`
- Prior performance plan: `docs/plans/2026-02-28-performance-optimizations-design.md`

### Related Work

- Prior decode-once optimization: `docs/plans/2026-02-28-performance-optimizations-design.md`
- DRY cleanup (PillowReencodeOptimizer): `docs/plans/2026-02-28-dry-cleanup-design.md`
