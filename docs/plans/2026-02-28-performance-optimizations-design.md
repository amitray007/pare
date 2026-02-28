# Performance Optimizations Design

**Goal:** Eliminate redundant image decoding, reduce memory waste, and move one-time setup out of the request path across optimizers.

**Scope:** Optimizers (AVIF, HEIC, JXL, WebP, BMP, PNG) — no changes to estimation, API surface, or output behavior.

---

## Phase 1: Decode-Once for Pillow-Based Optimizers

### Problem

`PillowReencodeOptimizer.optimize()` dispatches `_strip_metadata(data)` and `_reencode(data, quality)` concurrently. Each independently calls `_open_image(data)`, decoding the image from raw bytes a second time. For HEIC this means 2 full x265 decodes; for AVIF, 2 libavif decodes.

AVIF's `_strip_metadata` override and HEIC's `_strip_metadata` override also each call `_open_image` independently — same redundancy.

### Solution

Decode once in `optimize()`, pass `img.copy()` to the concurrent tasks. Pillow releases the GIL during C-level encode/decode, so `asyncio.to_thread()` parallelism is real — we must give each thread its own Image object.

**Changes to `pillow_reencode.py`:**

```python
async def optimize(self, data, config):
    img = await asyncio.to_thread(self._open_image, data)
    tasks = []
    method_names = []

    if config.strip_metadata:
        tasks.append(asyncio.to_thread(self._strip_metadata_from_img, img.copy(), data))
        method_names.append(self.strip_method_name)

    tasks.append(asyncio.to_thread(self._reencode_from_img, img, config.quality))
    method_names.append(self.reencode_method_name)

    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... rest unchanged (pick smallest candidate)
```

New `_from_img` methods on base class:
- `_strip_metadata_from_img(img, original_data)` — lossless re-encode from pre-decoded Image
- `_reencode_from_img(img, quality)` — lossy re-encode from pre-decoded Image

Subclass overrides (AVIF `_strip_metadata`, HEIC `_strip_metadata`) become `_strip_metadata_from_img` overrides instead, receiving `img` rather than `data`.

### WebP Binary Search Decode-Once

`WebpOptimizer._find_capped_quality()` calls `_pillow_optimize(data, quality)` per iteration, which calls `Image.open(io.BytesIO(data))` each time — up to 6 decodes during binary search.

Fix: decode once in `_find_capped_quality`, pass Image to a new `_encode_at_quality(img, quality)` helper:

```python
def _find_capped_quality(self, data, config):
    img = Image.open(io.BytesIO(data))
    is_animated = getattr(img, "n_frames", 1) > 1

    def encode_fn(quality):
        buf = io.BytesIO()
        save_kwargs = {"format": "WEBP", "quality": quality, "method": 4}
        if is_animated:
            save_kwargs["save_all"] = True
            save_kwargs["minimize_size"] = True
        img.save(buf, **save_kwargs)
        return buf.getvalue()

    return binary_search_quality(encode_fn, len(data), config.max_reduction, ...)
```

This is safe without `.copy()` because `_find_capped_quality` runs in a single thread — the binary search iterations are sequential.

---

## Phase 2: BMP Early-Termination Palette Detection

### Problem

`BmpOptimizer._try_lossless_palette()` does:
1. `pixels = list(img.getdata())` — materializes entire pixel buffer as Python list
2. `unique_colors = list(set(pixels))` — creates a full set, then converts to list
3. `color_to_idx = {c: i for i, c in enumerate(unique_colors)}` — builds reverse index

For a 1920x1080 RGB image, step 1 creates ~2M tuples, step 2 creates a set of up to 2M tuples, then checks `> 256`. This allocates 3-4x the pixel data in Python objects.

### Solution

Single-pass with early termination, building the index simultaneously:

```python
@staticmethod
def _try_lossless_palette(img):
    unique = {}
    pixel_indices = bytearray()

    for pixel in img.getdata():
        idx = unique.get(pixel)
        if idx is None:
            idx = len(unique)
            if idx >= 256:
                return None  # Early exit — no allocation wasted
            unique[pixel] = idx
        pixel_indices.append(idx)

    # unique is now {color: index} — replaces both set() and color_to_idx
    w, h = img.size
    palette_img = Image.new("P", (w, h))
    palette_img.putdata(list(pixel_indices))
    # ... build flat_palette from unique.keys() ...
```

Benefits:
- Bails out as soon as color 257 is found (instead of scanning all pixels first)
- Builds `color_to_idx` map inline (no second pass)
- Uses `bytearray` for indices instead of list-of-ints
- For photographic BMPs (millions of colors): exits in <1000 iterations vs. scanning all pixels

---

## Phase 3: One-Time Setup at Import/Startup

### Problem 1: Plugin registration per request

`_ensure_plugin()` in AVIF/HEIC/JXL subclasses runs on every `_open_image()` call:
- `import pillow_avif` — Python's import system has a lock, but still overhead
- `pillow_heif.register_heif_opener()` — re-registers on every request
- `import pillow_jxl` / `import jxlpy` — import lock contention under load

### Solution 1

Move plugin imports/registration to module-level in each subclass file. The `_ensure_plugin()` hook in the base class becomes a no-op (or is removed). This is safe because Pillow plugin registration is idempotent and these are always available in the Docker image.

**Example for `avif.py`:**
```python
import pillow_avif  # noqa: F401 — registers AVIF plugin at import time

class AvifOptimizer(PillowReencodeOptimizer):
    # _ensure_plugin() no longer needed
    ...
```

### Problem 2: oxipng imported inside method

`PngOptimizer._run_oxipng()` has `import oxipng` inside the method body (line 139). Called on every PNG optimization.

### Solution 2

Move `import oxipng` to module-level in `png.py`:

```python
import oxipng  # at top of file

class PngOptimizer(BaseOptimizer):
    def _run_oxipng(self, data, level=2):
        return oxipng.optimize_from_memory(data, level=level)
```

---

## Rejected Proposals

These were considered during the audit but rejected after correctness verification:

1. **BILINEAR resampling in estimator** — Would invalidate empirically calibrated LANCZOS correction factors (TIFF log-correction, PNG lossless cap). Recalibration across all formats would be required.

2. **TIFF sequential saves** (removing `img.copy()` + `asyncio.gather()`) — Pillow releases the GIL during C-level compression, so the concurrent TIFF paths (deflate/LZW/JPEG) achieve real parallelism. Making them sequential would regress latency.

3. **Shared Image decode in PillowReencodeOptimizer without `.copy()`** — Pillow Image objects are not thread-safe for concurrent saves. Each thread needs its own copy.

---

## Impact Assessment

- **No API changes** — request/response formats unchanged
- **No estimation changes** — estimator code untouched
- **No output changes** — same encoding parameters, same quality, same methods
- **Correctness preserved** — `.copy()` ensures thread isolation; early-termination produces identical palette output
- **Testable** — existing test suite (399 tests) covers all affected paths; benchmarks validate compression ratios
