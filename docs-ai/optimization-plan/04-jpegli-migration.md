# Phase 4: Jpegli Migration (Replace MozJPEG)

**Effort**: 2-3 days | **Risk**: Medium | **Impact**: Critical (fixes the #1 performance problem)

---

## Why Replace MozJPEG?

MozJPEG is the single biggest bottleneck in Pare. The numbers:

| Image | MozJPEG (current) | Expected with Jpegli |
|-------|-------------------|---------------------|
| 1920x1080 q=40 HIGH | **41.5 seconds** | **2-5 seconds** |
| 1920x1080 q=80 LOW | **12.2 seconds** | **1-3 seconds** |
| 800x600 q=40 HIGH | **3.4 seconds** | **0.5-1.5 seconds** |

**Why is MozJPEG so slow?**
1. MozJPEG's trellis quantization is 4-7x slower than libjpeg-turbo by design
2. Pare decodes JPEG → BMP (3-5x data inflation) before piping to `cjpeg`
3. `_cap_mozjpeg` binary search spawns up to 7 sequential `cjpeg` processes
4. No speed flags (`-notrellis`, `-fastcrush`) are used

## What is Jpegli?

Jpegli is Google's next-generation JPEG encoder from the libjxl project:

| Metric | MozJPEG | Jpegli | Winner |
|--------|---------|--------|--------|
| Encoding speed | Baseline | **33-40% faster** | Jpegli |
| Compression ratio | Good | **35% better** at same perceptual quality | Jpegli |
| Output compatibility | Standard JPEG | Standard JPEG (all decoders work) | Tie |
| API | libjpeg-turbo compatible | libjpeg62 ABI compatible (drop-in) | Tie |
| Human preference | — | Preferred 54% of time over MozJPEG at same filesize | Jpegli |

Jpegli is strictly better: faster encoding AND better compression AND fully backward-compatible output.

---

## 4.1 Build Jpegli in Dockerfile

Replace the MozJPEG build stage with a libjxl build stage:

**Current** (`Dockerfile` lines 1-19):
```dockerfile
FROM debian:bookworm-slim AS mozjpeg-builder
# ... builds MozJPEG v4.1.5 from source ...
```

**Proposed**:
```dockerfile
# ---- Stage 1: Build libjxl (includes jpegli) from source ----
FROM debian:bookworm-slim AS jpegli-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential curl ca-certificates git \
    libbrotli-dev libhwy-dev \
    && rm -rf /var/lib/apt/lists/*

ARG LIBJXL_VERSION=0.11.1
RUN curl -L https://github.com/libjxl/libjxl/archive/refs/tags/v${LIBJXL_VERSION}.tar.gz \
    | tar xz \
    && cd libjxl-${LIBJXL_VERSION} \
    && git init && git submodule update --init --recursive third_party \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/jpegli \
             -DBUILD_TESTING=OFF \
             -DJPEGXL_ENABLE_TOOLS=ON \
             -DJPEGXL_ENABLE_JPEGLI=ON \
             -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
             .. \
    && make -j$(nproc) \
    && make install
```

**Production stage** (replace lines 24-26):
```dockerfile
# Copy jpegli binaries (cjpegli replaces cjpeg, djpegli replaces djpeg)
COPY --from=jpegli-builder /opt/jpegli/bin/cjpegli /usr/local/bin/cjpegli
COPY --from=jpegli-builder /opt/jpegli/bin/djpegli /usr/local/bin/djpegli
# Keep jpegtran from MozJPEG for lossless optimization (jpegli doesn't replace jpegtran)
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/jpegtran /usr/local/bin/jpegtran
```

**Note**: We still need `jpegtran` for lossless Huffman optimization. Jpegli replaces `cjpeg` (lossy encoding) only. Options:
- Keep the MozJPEG build stage just for `jpegtran`
- Or use libjpeg-turbo's `jpegtran` (apt package `libjpeg-turbo-progs`)

---

## 4.2 Update JPEG Optimizer

**File**: `optimizers/jpeg.py`

### Key changes:

**a) Replace `cjpeg` with `cjpegli` (line 101)**

```python
async def _run_cjpegli(self, bmp_data: bytes, quality: int) -> bytes:
    """Run jpegli encoder on BMP input.

    cjpegli supports butteraugli distance (-d) and quality (-q).
    Using -q for compatibility with existing quality scale.
    """
    cmd = ["cjpegli", bmp_data_path, output_path, "-q", str(quality)]
    # Note: cjpegli CLI takes file paths, not stdin.
    # May need temp files or check if stdin is supported.
    ...
```

**Important**: Check `cjpegli` CLI interface — it may differ from `cjpeg`:
- `cjpeg` reads from stdin: `cjpeg -quality 80 < input.bmp > output.jpg`
- `cjpegli` may require file paths: `cjpegli input.bmp output.jpg -q 80`
- If `cjpegli` doesn't support stdin/stdout, temp files will be needed (like the cwebp fallback pattern)

**b) Alternative: Use jpegli as drop-in libjpeg replacement for Pillow**

Jpegli builds `libjpeg.so.62.3.0` which is ABI-compatible with libjpeg. If we set `LD_LIBRARY_PATH` to point to jpegli's library, Pillow will use jpegli transparently for JPEG encoding:

```dockerfile
# Copy jpegli shared library
COPY --from=jpegli-builder /opt/jpegli/lib/libjpeg.so* /usr/local/lib/
RUN ldconfig
```

This approach means **no code changes at all** — Pillow's `img.save(format="JPEG", quality=80)` would use jpegli automatically. However, it loses MozJPEG's specific optimizations and may not give the same compression ratio.

**Recommended approach**: Use `cjpegli` CLI (like current cjpeg), falling back to the Pillow-with-jpegli approach if CLI stdin/stdout isn't supported.

**c) Eliminate the BMP intermediate step (if using Pillow+jpegli)**

If we go the Pillow-with-jpegli-library route, the entire `_decode_to_bmp` step disappears:

```python
async def optimize(self, data, config):
    # Direct: JPEG → Pillow decode → Pillow JPEG encode (via jpegli)
    img = Image.open(io.BytesIO(data))
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=config.quality)
    reencoded = output.getvalue()

    # Also try jpegtran (lossless)
    jpegtran_out = await self._run_jpegtran(data, config.progressive_jpeg)

    # Pick smallest
    ...
```

This eliminates:
- The BMP inflation (6.2MB → gone)
- The subprocess pipe for lossy encoding
- The 3-5x data expansion that made MozJPEG process more bytes

**d) Simplify or remove the binary search**

With jpegli's butteraugli distance mode (`-d`), you can target a specific perceptual quality level directly instead of binary-searching quality values:
- `-d 1.0` = visually lossless
- `-d 2.0` = high quality
- `-d 4.0` = moderate quality

This could replace the entire `_cap_mozjpeg` binary search (7 sequential subprocess calls) with a single encoding call at the right distance value.

---

## 4.3 Update Estimation Heuristics

**File**: `estimation/heuristics.py`

Jpegli produces ~35% better compression than MozJPEG at the same perceptual quality. The JPEG prediction model needs recalibration:

- The encoder bonus constant (currently calibrated for MozJPEG's ~28% bonus at delta=0) needs to increase
- The piecewise linear curves need re-fitting from new benchmark data
- The screenshot detection path may change since jpegli handles flat areas differently

**Process**:
1. Build with jpegli
2. Run `python -m benchmarks.run --fmt jpeg`
3. Analyze the estimation accuracy table
4. Re-calibrate constants in `_predict_jpeg()` to match actual jpegli behavior

---

## 4.4 Update Tests

All JPEG tests that assert on specific file sizes or compression ratios will need updating since jpegli produces different (better) output than MozJPEG.

Tests that mock `cjpeg` subprocess calls will need to mock `cjpegli` instead (or the Pillow path if using the library approach).

---

## Expected Results

| Metric | MozJPEG (current) | Jpegli (expected) |
|--------|-------------------|-------------------|
| 1080p HIGH latency | 41.5s | **2-5s** |
| 1080p MED latency | 22s | **1-4s** |
| 1080p LOW latency | 12.2s | **0.5-2s** |
| Avg reduction (HIGH) | 62.4% | **65-70%** |
| Avg reduction (MED) | 39.8% | **42-48%** |
| Avg reduction (LOW) | 19.6% | **22-28%** |
| Binary search worst case | 7 x 2-8s = 56s | **Eliminated** (butteraugli distance) |

**Both faster AND better compression.** This is the highest-impact single change in the entire plan.

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| libjxl build complexity | Pin exact version; test in CI; cache Docker build layer |
| `cjpegli` CLI interface differs from `cjpeg` | Test stdin/stdout support; fall back to temp files if needed |
| Jpegli quality scale differs from MozJPEG | Map Pare quality to jpegli quality through benchmarking |
| `jpegtran` still needed for lossless path | Keep MozJPEG build stage or use libjpeg-turbo's jpegtran |
| Jpegli output differs from MozJPEG (tests break) | Expected; update test assertions after benchmarking |
| Pillow+jpegli library approach may not expose all features | Use CLI as primary, library as alternative |

---

## Migration Strategy

1. **Keep MozJPEG as fallback** during testing — don't remove until jpegli is proven
2. **Feature flag**: `JPEG_ENCODER=jpegli|mozjpeg` environment variable
3. **Parallel benchmarks**: Run same test suite with both encoders, compare
4. **Gradual rollout**: Deploy jpegli to staging first, monitor quality metrics
5. **Remove MozJPEG** once jpegli benchmarks confirm equal or better results

---

## Verification

```bash
# Build Docker image with jpegli
docker build -t pare-jpegli .

# Run JPEG benchmarks
python -m benchmarks.run --fmt jpeg

# Compare against current baseline
python -m benchmarks.run --fmt jpeg --compare

# Expected in comparison:
# - All latencies should drop dramatically
# - Reduction % should stay same or improve
# - Estimation accuracy will be poor (needs recalibration)

# Full suite regression check
python -m benchmarks.run
pytest tests/
```
