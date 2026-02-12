# Phase 2: Event Loop & Concurrency Fixes

**Effort**: 1 day | **Risk**: Low | **Impact**: Medium-High (throughput, not latency)

These fixes don't make individual images faster, but they prevent one slow image from blocking all other concurrent requests.

---

## The Problem

Several optimizers perform CPU-bound Pillow operations directly in `async` methods without `asyncio.to_thread()`. This blocks the uvicorn event loop, preventing ANY other requests from being served while that operation runs.

**Currently blocking the event loop**:

| File | Line | Operation | Blocking Duration |
|------|------|-----------|-------------------|
| `jpeg.py` | 25 | `_decode_to_bmp()` — Pillow JPEG decode + BMP serialize | 100-500ms |
| `webp.py` | 27 | `_pillow_optimize()` — Pillow WebP decode + encode | 2-15s |
| `webp.py` | 41 | `_find_capped_quality()` — up to 7x Pillow encodes | 14-105s |
| `webp.py` | 58 | `_pillow_optimize()` inside capped quality search | 2-15s per call |
| `avif.py` | 28 | `_strip_metadata()` — pillow-heif decode + AVIF re-encode | 1-30s |
| `heic.py` | 22 | `_strip_metadata()` — pillow-heif decode + HEIC re-encode | 1-10s |
| `tiff.py` | 29-72 | Entire `optimize()` — all Pillow saves | 1-6s |
| `bmp.py` | 24-63 | Entire `optimize()` — all Pillow + RLE8 operations | 0.1-2s |

**Already correct** (for reference):
- `png.py` lines 51, 66, 71 — wraps oxipng in `asyncio.to_thread()` correctly

---

## 2.1 Wrap JPEG BMP decode in `to_thread`

**File**: `optimizers/jpeg.py`

**Current** (line 25):
```python
bmp_data = self._decode_to_bmp(data, config.strip_metadata)
```

**Proposed**:
```python
bmp_data = await asyncio.to_thread(self._decode_to_bmp, data, config.strip_metadata)
```

**Impact**: Unblocks event loop for 100-500ms during Pillow JPEG decode.

---

## 2.2 Wrap WebP Pillow operations in `to_thread`

**File**: `optimizers/webp.py`

**Current** (line 27):
```python
pillow_result = self._pillow_optimize(data, config.quality)
```

**Proposed**:
```python
pillow_result = await asyncio.to_thread(self._pillow_optimize, data, config.quality)
```

Also wrap the `_find_capped_quality` method (line 41) and each `_pillow_optimize` call inside it (lines 58, 70):
```python
# Line 41
capped = await asyncio.to_thread(self._find_capped_quality, data, config)

# Or better: make _find_capped_quality async and wrap each internal call
async def _find_capped_quality(self, data, config):
    out_100 = await asyncio.to_thread(self._pillow_optimize, data, 100)
    # ... binary search ...
    out_mid = await asyncio.to_thread(self._pillow_optimize, data, mid)
```

**Impact**: Unblocks event loop for 2-105 seconds during WebP encoding.

---

## 2.3 Wrap AVIF/HEIC metadata stripping in `to_thread`

**File**: `optimizers/avif.py` line 28, `optimizers/heic.py` line 22

**Current**:
```python
optimized = self._strip_metadata(data)
```

**Proposed**:
```python
optimized = await asyncio.to_thread(self._strip_metadata, data)
```

**Impact**: Unblocks event loop for 1-30s (AVIF lossless re-encode is slow). This becomes critical once Phase 3 adds real AVIF/HEIC encoding.

---

## 2.4 Wrap TIFF optimizer in `to_thread`

**File**: `optimizers/tiff.py`

The entire loop at lines 50-70 should be wrapped:
```python
async def optimize(self, data, config):
    # ... metadata stripping, img open ...
    best, best_method = await asyncio.to_thread(
        self._try_all_methods, img, data, methods, config
    )
    return self._build_result(data, best, best_method)

def _try_all_methods(self, img, data, methods, config):
    # Move the sequential for loop into a sync method
    ...
```

**Bonus**: The sequential loop over methods could be parallelized with `asyncio.gather()` + `to_thread()` for each method, similar to how PNG runs pngquant + oxipng concurrently. This would reduce TIFF latency from sequential to concurrent.

---

## 2.5 Wrap BMP optimizer in `to_thread`

**File**: `optimizers/bmp.py`

Similar pattern — wrap the main Pillow/RLE8 work in `asyncio.to_thread()`.

---

## 2.6 Fix estimation probes (lower priority)

**File**: `estimation/heuristics.py`

Lines 424-464 (`_jpeg_probe`) and 467-519 (`_png_lossy_probe`) use synchronous `subprocess.run` instead of `asyncio.create_subprocess_exec`. These block the event loop for ~50-100ms per probe.

**Proposed**: Change to async subprocess calls or wrap in `asyncio.to_thread()`. These only fire on small files (<12KB), so the impact is minor — but it's a correctness issue.

---

## Expected Impact

This phase doesn't reduce single-image latency. It fixes **throughput under concurrent load**:

| Scenario | Before | After |
|----------|--------|-------|
| 4 concurrent JPEG requests | All 4 blocked on one decode | 4 processed concurrently |
| WebP request + 3 small requests | 3 small requests wait 15s | Small requests served immediately |
| AVIF/HEIC + PNG concurrent | AVIF blocks PNG start | Both proceed concurrently |

**Throughput improvement estimate**: 2-4x for mixed concurrent workloads.

---

## Verification

There's no benchmark change expected — this is about concurrency, not single-request speed. Test by:

1. Running the existing test suite (all tests should pass identically)
2. Load-testing with multiple concurrent requests:
   ```bash
   # Use a tool like wrk or hey to send 10 concurrent requests
   hey -n 100 -c 10 -m POST -D test_image.jpg http://localhost:8080/optimize
   ```
3. Checking that small-image responses are fast even while a large image is processing
