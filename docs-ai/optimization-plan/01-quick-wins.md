# Phase 1: Quick Wins (Settings & Flags)

**Effort**: 1-2 days | **Risk**: Low | **Impact**: High

These are configuration changes to existing tools — no new dependencies, no architectural changes.

---

## 1.1 Tune pngquant speed + oxipng level

**Problem**: `png.py:42-43` uses oxipng level=6 (180 filter trials) for quality < 50, and `png.py:57-58` uses pngquant `--speed 1` (brute force palette search). These are the slowest possible settings.

**Current** (`png.py:42-47, 56-61`):
```python
if config.quality < 50:
    oxipng_level = 6    # 180 trials — takes 10-30s on 1080p
elif config.quality < 70:
    oxipng_level = 4    # 24 trials
else:
    oxipng_level = 2    # 8 trials

if config.quality < 50:
    speed = 1           # brute force — takes 3-5s on 1080p
```

**Proposed**:
```python
if config.quality < 50:
    oxipng_level = 4    # 24 trials — 7-10x faster than level 6
elif config.quality < 70:
    oxipng_level = 3    # 16 trials
else:
    oxipng_level = 2    # 8 trials (unchanged)

if config.quality < 50:
    speed = 3           # good palette, 3-5x faster than speed 1
```

**Expected improvement**:
- PNG 1080p HIGH preset: **20-25s → 2-5s** (~5-10x faster)
- PNG 1080p MEDIUM preset: **10-15s → 3-7s** (~2-3x faster)
- Compression ratio impact: ~1-3% worse (negligible for most use cases)

**Files to change**:
- `optimizers/png.py` lines 42-47, 56-61
- `estimation/heuristics.py` — no change needed (estimation doesn't model oxipng speed)

**Verification**:
```bash
python -m benchmarks.run --fmt png
# Check: reduction % should stay within ~2-3% of current
# Check: times should drop dramatically for medium/large images
```

---

## 1.2 Add `--colors` flag to gifsicle

**Problem**: `gif.py:24-33` runs gifsicle with `--optimize=3` and `--lossy=N` but never reduces the color palette. GIF allows up to 256 colors per frame — reducing palette size dramatically improves LZW compression.

**Current** (`gif.py:23-35`):
```python
cmd = ["gifsicle", "--optimize=3"]
if config.quality < 50:
    cmd.append("--lossy=80")
elif config.quality < 70:
    cmd.append("--lossy=30")
```

**Proposed**:
```python
cmd = ["gifsicle", "--optimize=3"]
if config.quality < 50:
    cmd.extend(["--lossy=80", "--colors", "128"])
elif config.quality < 70:
    cmd.extend(["--lossy=30", "--colors", "192"])
# quality >= 70: no --colors (preserve original palette)
```

**Expected improvement**:
- GIF HIGH preset: **12% → 30-50%** average reduction
- GIF MEDIUM preset: **9% → 20-35%** average reduction
- GIF LOW preset: unchanged (lossless, no palette reduction)

**Files to change**:
- `optimizers/gif.py` lines 24-33
- `estimation/heuristics.py` — update `_predict_gif()` (line 615+) to account for `--colors` savings

**Verification**:
```bash
python -m benchmarks.run --fmt gif
# Check: HIGH/MEDIUM reduction % should increase significantly
# Check: estimation accuracy stays under 15%
```

---

## 1.3 WebP: reduce method, add `-mt` to cwebp

**Problem**: `webp.py:94` uses Pillow `method=6` (slowest WebP encoding). `webp.py:123` calls cwebp with `-m 6` and without the `-mt` (multi-thread) flag.

**Proposed changes to `webp.py`**:

a) Change Pillow method from 6 to 4:
```python
# Line 94: "method": 6 → "method": 4
"method": 4,  # Good compression, 2-3x faster than method=6
```

b) Add `-mt` flag and reduce method for cwebp fallback:
```python
# Line 123: add -mt, change -m 6 to -m 4
["cwebp", "-q", str(quality), "-m", "4", "-mt", in_path, "-o", out_path]
```

c) Run Pillow and cwebp concurrently instead of Pillow-first-then-fallback (optional, medium effort):
```python
# Current: pillow first, cwebp only if pillow >= 90% of input
# Proposed: run both in parallel via asyncio.gather(), pick smallest
```

**Expected improvement**:
- WebP encoding: **5-15s → 2-5s** per encode on 1080p (method=4 is 2-3x faster)
- cwebp with `-mt`: ~2.5x speedup from multi-threading
- Compression ratio impact: <5% worse (method=4 vs method=6)

**Files to change**:
- `optimizers/webp.py` lines 94, 123
- `estimation/heuristics.py` — no change needed

**Verification**:
```bash
python -m benchmarks.run --fmt webp
# Check: times should drop 2-3x
# Check: reduction % should stay within ~3-5% of current
```

---

## 1.4 Add MozJPEG speed flags (stopgap before Phase 4)

**Problem**: `jpeg.py:101` calls `cjpeg` with no speed-related flags. MozJPEG's trellis quantization is the biggest cost center.

**Proposed** (add to `jpeg.py:101-103`):
```python
cmd = ["cjpeg", "-quality", str(quality)]
if progressive:
    cmd.append("-progressive")
cmd.append("-notrellis")  # Save ~20% encoding time, ~10% larger files
```

**Expected improvement**:
- JPEG encoding per-call: **~20% faster** (reduces 2-8s to 1.6-6.4s per cjpeg call)
- With binary search (7 calls): saves 3-11 seconds total
- Compression ratio impact: ~10% worse (still much better than libjpeg-turbo)

**Note**: This is a stopgap. Phase 4 (Jpegli) replaces MozJPEG entirely with something that is both faster AND produces better compression. Consider skipping this if Phase 4 is starting soon.

**Files to change**:
- `optimizers/jpeg.py` line 101
- `estimation/heuristics.py` — update JPEG heuristics if `-notrellis` changes compression ratio significantly

**Verification**:
```bash
python -m benchmarks.run --fmt jpeg
# Check: times should drop ~20%
# Check: reduction % will be slightly lower — acceptable if still >5% better than jpegtran
```

---

## Summary: Phase 1 Expected Before/After

| Format | Metric | Before | After | Change |
|--------|--------|--------|-------|--------|
| PNG | 1080p HIGH latency | 20-25s | 2-5s | **~5-10x faster** |
| PNG | 1080p HIGH reduction | 78.9% | ~76-78% | ~1-3% less |
| GIF | Avg reduction (HIGH) | 12.2% | 30-50% | **2.5-4x better** |
| GIF | Avg reduction (MED) | 9.4% | 20-35% | **2-3.5x better** |
| WebP | 1080p latency | 5-15s | 2-5s | **2-3x faster** |
| WebP | Reduction | 49.9% | ~47-49% | ~1-3% less |
| JPEG | Per-cjpeg-call time | 2-8s | 1.6-6.4s | **~20% faster** |

All changes are minimal code edits to existing files with no new dependencies.
