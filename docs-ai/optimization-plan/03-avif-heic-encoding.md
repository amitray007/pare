# Phase 3: Real AVIF & HEIC Encoding

**Effort**: 2-3 days | **Risk**: Medium | **Impact**: High (AVIF/HEIC go from no-ops to real optimization)

---

## Current State

Both `avif.py` and `heic.py` only strip metadata. The docstrings explicitly say "no decode/re-encode to avoid generation loss." Current results:
- AVIF: ~5% reduction (metadata only)
- HEIC: ~5% reduction (metadata only)

This means users sending AVIF/HEIC files get almost nothing back.

## The Generation Loss Question

**Is re-encoding lossy AVIF/HEIC actually harmful?**

Yes — but manageable:
- Each decode/re-encode cycle adds quantization noise (similar to JPEG)
- At quality 75-85, the degradation is minimal and generally imperceptible
- The key insight: **only re-encode when the savings justify it** (e.g., >10% reduction)
- If re-encoding produces <10% savings, the generation loss isn't worth it — return original

This is the same tradeoff MozJPEG makes for JPEG (and users accept it).

## Strategy

**Two-tier approach**:
1. **Lossy re-encoding** (quality < 70): Decode → re-encode at target quality. Accept generation loss for meaningful compression.
2. **Conservative re-encoding** (quality >= 70): Only re-encode if savings > 15%. High quality minimizes generation loss.

Always compare against the original and return the original if savings are insufficient (existing `_build_result` guarantee handles this).

---

## 3.1 AVIF Real Encoding

### Encoder Choice: SVT-AV1 over libaom

| Encoder | Speed (1080p) | Quality | Availability |
|---------|--------------|---------|--------------|
| libaom (reference) | 30-120s | Best | Default in Pillow |
| SVT-AV1 (Intel/Netflix) | **3-15s** | Excellent | Pillow codec="svt" |
| rav1e (Rust) | 15-60s | Good | Pillow codec="rav1e" |

**SVT-AV1 is the clear choice for an API**: 5-20x faster than libaom with comparable quality. Netflix uses SVT-AV1 in production.

### Implementation Plan

**File**: `optimizers/avif.py`

Replace the current metadata-only approach:

```python
class AvifOptimizer(BaseOptimizer):
    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        # Try re-encoding (lossy) and metadata-strip, pick best
        candidates = []

        # Always try metadata strip (cheap, lossless)
        if config.strip_metadata:
            try:
                stripped = await asyncio.to_thread(self._strip_metadata, data)
                candidates.append((stripped, "metadata-strip"))
            except Exception:
                pass

        # Try lossy re-encoding at target quality
        try:
            reencoded = await asyncio.to_thread(
                self._reencode, data, config.quality
            )
            candidates.append((reencoded, "avif-svt"))
        except Exception:
            pass

        if not candidates:
            return self._build_result(data, data, "none")

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    def _reencode(self, data: bytes, quality: int) -> bytes:
        """Re-encode AVIF using SVT-AV1 via Pillow."""
        import pillow_heif
        pillow_heif.register_avif_opener()

        img = Image.open(io.BytesIO(data))
        icc_profile = img.info.get("icc_profile")

        # Map Pare quality (1-100, lower=aggressive) to AVIF quality
        # Pare q=40 (HIGH) → AVIF q=50 (aggressive)
        # Pare q=60 (MED)  → AVIF q=65 (moderate)
        # Pare q=80 (LOW)  → AVIF q=80 (conservative)
        avif_quality = max(30, min(90, quality + 10))

        output = io.BytesIO()
        save_kwargs = {
            "format": "AVIF",
            "quality": avif_quality,
            "speed": 6,          # Balanced speed/quality
            "codec": "svt",      # SVT-AV1: 5-20x faster than libaom
        }
        if icc_profile:
            save_kwargs["icc_profile"] = icc_profile

        img.save(output, **save_kwargs)
        return output.getvalue()
```

### Quality Thresholds

| Pare Quality | AVIF Quality | Behavior |
|-------------|-------------|----------|
| < 50 (HIGH) | 50 | Aggressive re-encode, accept generation loss |
| < 70 (MEDIUM) | 65 | Moderate re-encode |
| >= 70 (LOW) | 80 | Conservative, only keep if >15% savings |

### Dockerfile Changes

SVT-AV1 needs to be available. Check if `pillow-heif` bundles it, or add:
```dockerfile
RUN apt-get install -y --no-install-recommends libsvtav1enc-dev
```

Alternatively, if Pillow (v11.3+) has native AVIF support with SVT-AV1, `pillow-heif` may not be needed for AVIF at all.

### Requirements Changes

```
# requirements.txt — check Pillow version supports AVIF natively
Pillow>=11.3.0  # Native AVIF read/write with codec selection
```

If Pillow native AVIF is sufficient, the `pillow-heif` dependency could be removed for AVIF (still needed for HEIC).

---

## 3.2 HEIC Real Encoding

### Encoder: x265 via pillow-heif

`pillow-heif` bundles libheif + x265. HEIC encoding is 2-5x faster than AV1 encoding for equivalent quality.

### Implementation Plan

**File**: `optimizers/heic.py`

Same pattern as AVIF:

```python
def _reencode(self, data: bytes, quality: int) -> bytes:
    """Re-encode HEIC using x265 via pillow-heif."""
    import pillow_heif

    heif_file = pillow_heif.open_heif(data)
    img = heif_file.to_pillow()
    icc_profile = img.info.get("icc_profile")

    heic_quality = max(30, min(90, quality + 10))

    output = io.BytesIO()
    save_kwargs = {
        "format": "HEIF",
        "quality": heic_quality,
    }
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile

    img.save(output, **save_kwargs)
    return output.getvalue()
```

### x265 Preset Tuning

For still images, only three x265 presets matter (others give nearly identical results):
- `ultrafast`: Fastest, good for an API
- `superfast`: Slight quality improvement
- `slow`: Best quality, but much slower

Configure via `pillow-heif`'s `enc_params` if needed:
```python
save_kwargs["enc_params"] = {"preset": "ultrafast"}
```

---

## 3.3 Update Estimation Heuristics

**File**: `estimation/heuristics.py`

The current `_predict_metadata_only` (line 774) returns a flat 5% prediction. This must be replaced with format-aware heuristics.

### New `_predict_avif` function:

```python
def _predict_avif(info: HeaderInfo, config: OptimizationConfig) -> Prediction:
    """AVIF — predict re-encoding savings based on quality delta."""
    if config.quality >= 70:
        # Conservative: small savings from re-encoding
        reduction = 10.0 if info.file_size > 50_000 else 5.0
    elif config.quality >= 50:
        # Moderate: meaningful re-encoding
        reduction = 25.0
    else:
        # Aggressive: significant re-encoding
        reduction = 40.0

    # Already-small files have less room for improvement
    if info.file_size < 10_000:
        reduction *= 0.5

    return Prediction(
        estimated_size=int(info.file_size * (1 - reduction / 100)),
        reduction_percent=round(reduction, 1),
        potential="medium" if reduction > 15 else "low",
        method="avif-svt",
        already_optimized=reduction < 5,
        confidence="medium",
    )
```

### Update dispatch table (line 34):
```python
ImageFormat.AVIF: _predict_avif,    # was: _predict_metadata_only
ImageFormat.HEIC: _predict_heic,    # was: _predict_metadata_only
```

---

## 3.4 Update Benchmark Cases

**File**: `benchmarks/cases.py`

Add AVIF and HEIC test images at various quality levels to the benchmark suite. The generators in `benchmarks/generators.py` will need AVIF/HEIC image generation functions.

---

## Expected Results

| Format | Metric | Before | After |
|--------|--------|--------|-------|
| AVIF | Avg reduction (HIGH) | ~5% | **35-50%** |
| AVIF | Avg reduction (MED) | ~5% | **25-35%** |
| AVIF | Avg reduction (LOW) | ~5% | **10-20%** |
| AVIF | 1080p latency (SVT-AV1) | <1s (metadata only) | **3-15s** |
| HEIC | Avg reduction (HIGH) | ~5% | **30-45%** |
| HEIC | Avg reduction (MED) | ~5% | **20-30%** |
| HEIC | 1080p latency (x265) | <1s (metadata only) | **1-5s** |

**Tradeoff**: Latency increases because we're doing real work now. But the output is actually optimized, which is the whole point of the API.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Generation loss complaints | Only re-encode when savings > 10%; document in API response |
| SVT-AV1 not available in container | Test Dockerfile with `libsvtav1enc-dev`; fall back to libaom speed=8 |
| Pillow AVIF support incomplete | Keep `pillow-heif` as fallback encoder |
| Encoding too slow for API | Use SVT-AV1 speed=6-8; set per-format timeout |
| HEIC licensing (x265 is GPL) | `pillow-heif` already bundles x265; no new licensing exposure |

---

## Verification

```bash
# Run benchmarks for AVIF (will need test images added first)
python -m benchmarks.run --fmt avif

# Run benchmarks for HEIC (will need test images added first)
# Note: HEIC benchmark images require pillow-heif for generation

# Full suite to check nothing else broke
python -m benchmarks.run

# Check estimation accuracy
# AVIF/HEIC estimation will start rough — calibrate after first benchmark run
```
