# Phase 5: Future Enhancements

**Effort**: Ongoing | **Risk**: Varies | **Impact**: Moves Pare toward industry-leading performance

These are longer-term improvements to evaluate after Phases 1-4 are complete.

---

## 5.1 Migrate from Pillow to pyvips (libvips)

**Priority**: High (after Phase 4) | **Effort**: 1-2 weeks | **Impact**: 2-5x across all formats

### Why libvips?

libvips is what Sharp (Node.js, 30k+ GitHub stars) and imgproxy (Go) use under the hood. It is the industry standard for production image processing.

| Benchmark | Pillow | pyvips | Improvement |
|-----------|--------|--------|-------------|
| 10Kx10K crop+resize+sharpen+save | 1.51s / 1040MB | 0.69s / 109MB | **2.2x faster / 9.5x less memory** |
| Production (Criteo migration) | baseline | — | **3.3x faster** |
| JPEG resize (Jan 2025 bench) | baseline | — | **25x faster** |

**Key advantages**:
- **Streaming pipeline**: Never loads full image into memory (processes in tiles)
- **No BMP intermediate**: Can read JPEG and write JPEG directly with MozJPEG/jpegli
- **Replaces most subprocess calls**: Built-in support for JPEG, PNG (with libimagequant), WebP, AVIF, HEIC, TIFF, GIF
- **GIL released**: pyvips CFFI calls release the GIL, enabling true parallel processing
- **Zero-copy**: Image data stays in C land, no Python buffer copies

### What pyvips replaces

| Current | pyvips Replacement |
|---------|-------------------|
| Pillow JPEG decode + cjpeg subprocess | `pyvips.Image.new_from_buffer()` → `.jpegsave_buffer()` |
| Pillow WebP + cwebp subprocess | `pyvips.Image.new_from_buffer()` → `.webpsave_buffer()` |
| Pillow AVIF via pillow-heif | `pyvips.Image.new_from_buffer()` → `.heifsave_buffer()` |
| Pillow TIFF sequential saves | `pyvips.Image.tiffsave_buffer(compression=...)` |
| Pillow BMP processing | `pyvips.Image` for decode, keep custom RLE8 |

### What pyvips does NOT replace

| Current | Keep |
|---------|------|
| pyoxipng | Keep (Rust in-process, already optimal) |
| pngquant | Could replace with libvips+libimagequant, but pngquant is proven |
| gifsicle | Keep (specialized GIF optimization) |
| scour | Keep (specialized SVG optimization) |

### Migration approach

1. Add `pyvips` to requirements.txt
2. Add `libvips-dev` to Dockerfile
3. Migrate one optimizer at a time (start with JPEG — biggest impact)
4. Benchmark each migration step
5. Remove Pillow dependency only after all formats are migrated

### Docker changes

```dockerfile
RUN apt-get install -y --no-install-recommends \
    libvips-dev \
    libvips-tools
```

If libvips is compiled with MozJPEG/jpegli support, it handles JPEG encoding natively — no separate cjpeg/cjpegli binary needed.

---

## 5.2 JPEG XL Support

**Priority**: Medium | **Effort**: 3-5 days | **Impact**: New format + unique lossless JPEG recompression

### Browser Support (as of early 2026)

- **Safari**: Ships JPEG XL support
- **Chrome**: Merged JPEG XL decoding (Jan 2026) using a Rust decoder (jxl-rs)
- **Firefox**: Neutral/interested; security concerns about C++ libjxl

### Two value propositions

**a) Lossless JPEG recompression (unique, high-value)**

JPEG XL can transcode an existing JPEG to JXL with:
- ~20% size reduction
- **Perfect bit-exact round-trip** back to the original JPEG
- Zero quality loss (mathematically lossless)
- Fast (much faster than MozJPEG lossy encoding)

This is a feature NO other format offers. It's essentially "free" compression.

**b) Full JXL encoding (competing with AVIF)**

JPEG XL's general compression is competitive with AVIF:
- Better than AVIF at typical speed settings for 1080p images
- Supports progressive decoding
- Supports lossless mode

### Implementation

**Python bindings**: `pillow-jxl-plugin` (Rust-based, pip-installable) or `jxlpy` (Cython, requires libjxl)

**CLI**: `cjxl` (encoder) from libjxl. Since Phase 4 already builds libjxl for jpegli, `cjxl` comes for free.

```python
class JxlOptimizer(BaseOptimizer):
    format = ImageFormat.JXL

    async def optimize(self, data, config):
        # For JPEG input: try lossless JPEG-to-JXL transcoding
        if is_jpeg(data):
            jxl_lossless = await self._jpeg_to_jxl_lossless(data)
            # This is ~20% savings with zero quality loss

        # For JXL input: try re-encoding at target quality
        jxl_reencoded = await self._reencode(data, config.quality)

        # Pick best
        ...
```

### Prerequisites

- Phase 4 (libjxl is already built for jpegli)
- `ImageFormat.JXL` enum value
- `format_detect.py` magic bytes for JXL
- New estimation heuristic `_predict_jxl()`
- Benchmark cases for JXL

---

## 5.3 Content-Aware Quality Selection (SSIM/Butteraugli)

**Priority**: Medium | **Effort**: 1-2 weeks | **Impact**: Better quality-per-byte

### The concept

Instead of a fixed quality number, detect the optimal quality level that produces imperceptibly different output. Professional services call this "auto quality" (Cloudinary's `q_auto`).

### How it works

1. Encode at a target quality
2. Compute perceptual quality metric (SSIM, butteraugli, or VMAF) between original and encoded
3. If metric is above threshold → quality is sufficient, try lower
4. If metric is below threshold → quality is too aggressive, try higher
5. Binary search to find the optimal quality

### Available metrics

| Metric | Speed | Accuracy | Availability |
|--------|-------|----------|--------------|
| SSIM | Fast | Good | Pillow, scikit-image, pyvips |
| Butteraugli | Medium | Excellent | libjxl (built in Phase 4) |
| VMAF | Slow | Best (trained on human data) | Netflix open-source |

### Practical approach for Pare

Start with **butteraugli** since libjxl (Phase 4) includes it. Jpegli already supports butteraugli distance (`-d` flag):

```python
# Instead of: cjpegli -q 60 (fixed quality)
# Use:        cjpegli -d 2.0 (target perceptual distance)
#
# -d 1.0 = visually lossless
# -d 2.0 = high quality, small artifacts
# -d 4.0 = moderate quality, noticeable on close inspection
```

This eliminates the quality guessing problem entirely. Map Pare quality presets to butteraugli distances:

| Pare Preset | Butteraugli Distance | Meaning |
|------------|---------------------|---------|
| HIGH (q=40) | 3.0-4.0 | Aggressive, visible on inspection |
| MEDIUM (q=60) | 1.5-2.0 | Good quality, minor artifacts |
| LOW (q=80) | 0.5-1.0 | Near-lossless |

---

## 5.4 Format Conversion (Cross-Format Optimization)

**Priority**: Low-Medium | **Effort**: 1 week | **Impact**: Potential 30-50% better compression

### The opportunity

The biggest compression gains come from format conversion, not same-format re-encoding:

| Conversion | Typical Savings |
|-----------|----------------|
| JPEG → AVIF | 30-50% at same visual quality |
| JPEG → WebP | 25-35% at same visual quality |
| PNG → WebP (lossy) | 60-80% for photos |
| PNG → AVIF | 70-85% for photos |
| GIF → WebP (animated) | 40-60% |

### API design question

Format conversion changes the output format, which has client implications (the browser must support the output format). Two approaches:

**a) Accept header negotiation** (like CDNs):
```
POST /optimize
Accept: image/avif, image/webp, image/jpeg
→ Returns best format the client supports
```

**b) Explicit opt-in**:
```json
{
  "optimization": {
    "quality": 60,
    "allow_format_conversion": true,
    "preferred_formats": ["avif", "webp"]
  }
}
```

### Implementation considerations

- Requires `ImageFormat` to have output format selection
- API response must include the output format in headers/JSON
- Estimation engine needs cross-format prediction models
- Benchmark suite needs cross-format test cases

---

## 5.5 Estimation Model Improvements

**Priority**: Low | **Effort**: Ongoing

### Extract magic numbers to calibration file

`heuristics.py` (915 lines) has dozens of calibrated constants embedded in code:
```python
# Examples of buried constants:
encoder_bonus = 28.0
ratio = 0.668 * _exp(0.293 * (source_q - 90))
```

These should be extracted to a `estimation/calibration.json` or `estimation/constants.py` file with comments explaining where each number came from and which benchmark run calibrated it.

### BMP estimation fix

The benchmark showed BMP estimation errors of 25.7% for the RLE8 path. The heuristic doesn't predict how well RLE8 handles flat/screenshot content. Fix by adding a content-type check:

```python
# If screenshot/flat content and quality < 50: predict RLE8 ~95-99% reduction
# If photo content and quality < 50: predict palette ~60-70% reduction
```

### JPEG LOW preset estimation fix

The JPEG q=92 at LOW preset showed 15.5% estimation error (predicted 25%, actual 9.5%). The jpegtran prediction overestimates savings for medium-quality sources. Fix by adjusting the jpegtran baseline curve.

---

## 5.6 Parallel Method Trials for TIFF

**Priority**: Low | **Effort**: 0.5 days | **Impact**: TIFF 1.5-2x faster

`tiff.py:50` runs compression methods sequentially:
```python
for compression in methods:  # deflate, lzw, jpeg — one at a time
```

Change to concurrent:
```python
results = await asyncio.gather(*[
    asyncio.to_thread(self._try_compression, img, compression, config)
    for compression in methods
])
best = min(results, key=lambda r: len(r[0]))
```

---

## Phase 5 Prioritization

After Phases 1-4 are complete, evaluate these in order:

1. **pyvips migration** (5.1) — Highest general impact, but largest effort
2. **JPEG XL lossless recompression** (5.2a) — Unique feature, libjxl already built
3. **Butteraugli-driven quality** (5.3) — Leverages jpegli, replaces binary search
4. **BMP/JPEG estimation fixes** (5.5) — Quick win for estimation accuracy
5. **TIFF parallelization** (5.6) — Quick win for TIFF latency
6. **Format conversion** (5.4) — Biggest potential impact but biggest API design change
7. **Full JXL encoding** (5.2b) — Wait for browser support to mature

---

## Long-Term Vision

After all phases, Pare would:
- Process 1080p images in **<5 seconds** for any format
- Achieve **best-in-class compression** using jpegli, SVT-AV1, and libvips
- Offer **content-aware quality selection** via butteraugli
- Support **JPEG XL** with unique lossless JPEG recompression
- Run on the same Python/FastAPI stack with **no language rewrite needed**
- Compete with Cloudinary/imgix on compression quality while being self-hosted
