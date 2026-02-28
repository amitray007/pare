# DRY Cleanup Design

**Date:** 2026-02-28
**Status:** Approved
**Priority:** DRY violations first (user-selected from 6 identified cleanup categories)

## Problem

The codebase has significant copy-paste duplication across optimizers and the estimation engine. Three optimizers (AVIF, HEIC, JXL) share near-identical `optimize()`, `_strip_metadata()`, and `_reencode()` methods (~90 lines each). Binary search quality capping is triplicated across JPEG and WebP. Quality clamping formulas are copy-pasted in 6 locations. `EstimateResponse` is constructed with identical kwargs in 6 separate places.

## Approach: Intermediate Base Class + Shared Utilities

### 1. PillowReencodeOptimizer (new intermediate base class)

**File:** `optimizers/pillow_reencode.py`

Captures the shared optimize/strip/reencode pattern for AVIF, HEIC, and JXL. Subclasses define only format-specific constants.

Inheritance hierarchy:
```
BaseOptimizer
  └── PillowReencodeOptimizer  (shared optimize/strip/reencode)
        ├── AvifOptimizer       (format="AVIF", quality_max=90, speed=6)
        ├── HeicOptimizer       (format="HEIF", quality_max=90, uses pillow-heif opener)
        └── JxlOptimizer        (format="JXL", quality_max=95, jxlpy fallback)
```

The base class owns:
- `async optimize(data, config)` — runs strip + reencode concurrently, picks smallest
- `_strip_metadata(data)` — lossless re-encode without metadata, preserving ICC
- `_reencode(data, quality)` — lossy re-encode at clamped quality

Subclasses specify via class attributes:
- `pillow_format: str` — Pillow save format string ("AVIF", "HEIF", "JXL")
- `strip_method_name: str` — method name for results ("metadata-strip")
- `reencode_method_name: str` — method name for results ("avif-reencode", etc.)
- `quality_min: int`, `quality_max: int`, `quality_offset: int` — quality clamp params
- `extra_save_kwargs: dict` — format-specific save kwargs (e.g. `{"speed": 6}` for AVIF)

Subclasses override:
- `_ensure_plugin()` — format-specific plugin import
- `_open_image(data)` — override for HEIC's pillow-heif opener pattern

### 2. Shared Utilities (`optimizers/utils.py`)

**`clamp_quality(quality, offset=10, lo=30, hi=90) -> int`**

Replaces 6 copies of `max(lo, min(hi, quality + offset))`. Used by:
- `PillowReencodeOptimizer._reencode()` (3 optimizers)
- `_heic_sample_bpp()`, `_avif_sample_bpp()`, `_jxl_sample_bpp()` (3 estimator helpers)

**`binary_search_quality(encode_fn, original_size, target_reduction, lo, hi, max_iters=5) -> bytes | None`**

Replaces 3 near-identical binary search implementations:
- `jpeg.py::_cap_quality` (sync, Pillow encode)
- `jpeg.py::_cap_mozjpeg` (async, cjpeg subprocess)
- `webp.py::_find_capped_quality` (sync, Pillow encode)

The `encode_fn: Callable[[int], bytes]` parameter abstracts over format-specific encoding.

Note: `_cap_mozjpeg` is async (uses subprocess). Provide an async variant or make the JPEG optimizer wrap the async call in a sync adapter for the binary search.

### 3. EstimateResponse Factory

**In `estimation/estimator.py`**, private helper:

```python
def _build_estimate(
    file_size, fmt, width, height, color_type, bit_depth,
    estimated_size, reduction, method, confidence="high",
) -> EstimateResponse:
```

All 6 construction sites collapse to single-line calls:
- `_estimate_exact` (1 site)
- `_estimate_by_sample` (2 sites: optimizer-says-none + extrapolation)
- `_bpp_to_estimate` (1 site)
- `estimate_from_thumbnail` (2 sites: already-optimized + extrapolation)

### 4. Inline Import Cleanup

Move to module-level:
- `import math` (estimator.py line 319)
- `import subprocess` (estimator.py line 503)
- `import io` (benchmarks/corpus.py)
- `from PIL import Image` (benchmarks/corpus.py)

Keep method-level (plugin side effects, optional installs):
- `import pillow_avif` in AvifOptimizer._ensure_plugin()
- `import pillow_heif` in HeicOptimizer._ensure_plugin()
- `import pillow_jxl / jxlpy` in JxlOptimizer._ensure_plugin()

### 5. Preset Deduplication

`estimation/presets.py` and `benchmarks/constants.py` define the same HIGH/MEDIUM/LOW presets independently. Make `estimation/presets.py` reference `benchmarks/constants.py` as the single source of truth.

## What This Does NOT Touch

- PNG, GIF, TIFF, BMP, SVG optimizers (unique logic)
- JPEG optimizer pipeline (unique; only binary search extracted)
- WebP optimizer pipeline (unique; only binary search extracted)
- Schema modernization (`Optional[X]` → `X | None`) — separate future pass
- File structure / module boundaries — separate future pass
- Error handling patterns — separate future pass

## Impact

| Area | Before | After |
|------|--------|-------|
| AVIF optimizer | 95 lines | ~20 lines |
| HEIC optimizer | 91 lines | ~25 lines |
| JXL optimizer | 95 lines | ~20 lines |
| Shared base (new) | 0 lines | ~80 lines |
| Binary search copies | 3 × ~20 lines | 1 × ~20 lines |
| Quality clamp copies | 6 copies | 1 function |
| EstimateResponse construction | 6 × ~12 lines | 6 × 1 line |
| Preset definitions | 2 files | 1 source of truth |
| Net line delta | ~0 (trade duplication for structure) |

## Testing

- All existing tests must pass unchanged (behavior is preserved)
- Run `python -m benchmarks.run` to verify no regression in optimization results
- Run format-specific benchmarks for AVIF, HEIC, JXL, JPEG, WebP
