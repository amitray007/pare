# Sample-Based Estimation: Design Document

## Problem

The current estimation system uses ~1,650 lines of hand-tuned, per-format heuristic functions to predict compression results from image metadata. This approach:

- Requires parallel maintenance: every optimizer change needs a matching heuristic update
- Has inconsistent accuracy across formats (5-20% error depending on format and content)
- Needs corpus calibration and manual constant tuning
- Grows linearly with new format support (~100 lines per format predictor)

## Solution

Replace heuristics with **sample-based compression**: downsample the image to ~300px wide, run the actual optimizer on the sample, measure output BPP (bits per pixel), and extrapolate to the full image size.

The key insight: BPP is consistent across resolutions for the same content and quality setting. A 300x300 JPEG at quality 60 has nearly the same BPP as a 3000x3000 of the same content (<3% variation).

## Requirements

- **Accuracy:** Average error 5-10%, max 20%
- **Latency:** Average <500ms, up to 5s acceptable for edge cases
- **Use case:** Shopify product/collection/blog images, 500KB-5MB typical, up to 50MB
- **Workflow:** Customer selects image + preset (HIGH/MEDIUM/LOW), sees estimate, then decides to optimize

## API Design

```
POST /estimate
{
  "url": "https://cdn.shopify.com/.../product.jpg",
  "thumbnail_url": "https://cdn.shopify.com/.../product.jpg?width=300",   // optional, for >= 10MB
  "file_size": 2500000,                                                    // optional, skip HEAD
  "preset": "high"                                                         // "high" | "medium" | "low"
}

Response:
{
  "original_size": 2500000,
  "original_format": "jpeg",
  "dimensions": {"width": 3000, "height": 2000},
  "estimated_optimized_size": 625000,
  "estimated_reduction_percent": 75.0,
  "optimization_potential": "high",
  "method": "jpegli",
  "already_optimized": false,
  "confidence": "high"
}
```

### Request fields

| Field | Required | Purpose |
|-------|----------|---------|
| `url` | yes | Original image URL |
| `preset` | yes | Compression level: "high", "medium", "low" |
| `thumbnail_url` | no | CDN-resized thumbnail for >= 10MB images |
| `file_size` | no | Original file size in bytes (skips HTTP HEAD) |

### Preset mapping

| Preset | Quality | png_lossy | Description |
|--------|---------|-----------|-------------|
| HIGH | 40 | true | Maximum compression, lower file size |
| MEDIUM | 60 | true | Balanced compression |
| LOW | 80 | false | Light compression, near-original quality |

## Core Algorithm

```
1. ACQUIRE IMAGE DATA
   - Determine file_size (from client param or HTTP HEAD on url)
   - If file_size < 10MB:
       Download full original from url
       Decode with Pillow -> get (width, height, format, pixels)
   - If file_size >= 10MB AND thumbnail_url provided:
       Download thumbnail -> decode with Pillow
       Range request on original (first 4KB) -> parse (width, height, format)
   - If file_size >= 10MB AND no thumbnail_url:
       Download full original (slower but works)

2. PREPARE SAMPLE
   - original_pixels = width * height
   - If original_pixels <= 150,000 (~390x390):
       sample = full image (no resize needed)
       mode = "exact"
   - If original_pixels > 150,000:
       sample = resize to ~300px wide (proportional, Lanczos)
       mode = "extrapolate"

3. COMPRESS SAMPLE
   - Map preset -> OptimizationConfig (see preset table above)
   - Run: result = await optimizer.optimize(sample_bytes, config)
   - sample_output_size = result.optimized_size

4. EXTRAPOLATE
   - If mode == "exact":
       estimated_size = sample_output_size
       confidence = "high"
   - If mode == "extrapolate":
       sample_output_bpp = sample_output_size * 8 / sample_pixels
       estimated_size = sample_output_bpp * original_pixels / 8
       confidence = "high"

5. RESULT
   - reduction = (file_size - estimated_size) / file_size * 100
   - If reduction <= 0: reduction = 0, already_optimized = True
   - Return EstimateResponse
```

## Special Cases

### SVG/SVGZ
No pixels to sample. Download the full file (SVGs are small, typically <1MB) and run the SVG optimizer directly. Returns exact result.

### Animated GIF/APNG
Run gifsicle on the full file (gifsicle is fast even on multi-frame GIFs). For APNG, compress the full file with the APNG optimizer. Most animated files are under 5MB.

### Already optimized images
If the optimizer returns output >= input on the sample, report 0% reduction with `already_optimized: true`. Matches the existing `_build_result()` output-<=input guarantee.

### Timeout safety
If optimizer exceeds 3s on the sample (should almost never happen), abort and return a low-confidence fallback: assume 30% reduction for lossy presets, 5% for lossless. This is a safety net, not an expected path.

## What Changes

### Deleted (~1,650 lines)
- `estimation/heuristics.py` - all 12 format-specific prediction functions
- `estimation/header_analysis.py` - content probing, flat_pixel_ratio, JPEG quality reverse-lookup, thumbnail compression

### Rewritten (~80-100 lines)
- `estimation/estimator.py` - new sample-based logic

### Modified
- `routers/estimate.py` - accept new JSON fields (preset, thumbnail_url, file_size)
- `schemas.py` - add preset field, optional thumbnail_url/file_size to request schema

### Unchanged
- All optimizers in `optimizers/` - estimation calls them directly
- `utils/format_detect.py` - still needed for format detection
- `utils/url_fetch.py` - still needed for downloading
- `benchmarks/` - still works, measures extrapolation accuracy instead of heuristic accuracy

## Expected Accuracy

| Format | Expected avg error | Why |
|--------|-------------------|-----|
| JPEG | 2-3% | BPP extremely consistent across JPEG resolutions |
| WebP | 2-3% | Same block-based consistency as JPEG |
| AVIF/HEIC/JXL | 2-5% | Modern codecs, consistent BPP |
| PNG (lossy) | 3-5% | Palette quantization is content-dependent, not size-dependent |
| PNG (lossless) | 5-8% | Deflate effectiveness varies slightly with resolution |
| GIF/APNG | exact | Full-file compression, no extrapolation |
| SVG/SVGZ | exact | Full-file compression, no extrapolation |
| BMP/TIFF | 2-5% | Format conversion to target format, BPP consistent |

## Expected Latency

| Scenario | Download | Compress | Total |
|----------|----------|----------|-------|
| JPEG 2MB (typical) | ~30ms | ~30ms | ~60ms |
| PNG 5MB | ~80ms | ~100ms | ~180ms |
| WebP 3MB | ~50ms | ~20ms | ~70ms |
| AVIF 5MB | ~80ms | ~200ms | ~280ms |
| 50MB + CDN thumbnail | ~10ms | ~100ms | ~110ms |
| SVG 200KB | ~5ms | ~30ms | ~35ms |

## Key Design Property

Estimation is decoupled from optimizer internals. When optimizer logic changes (new methods, tuned thresholds, new formats), estimation automatically adapts because it calls the optimizer directly. Zero maintenance burden for estimation when optimizers evolve.
