# pyvips Migration: Design Document

## Problem

The current optimization infrastructure uses 7 CLI tools, 7+ Python libraries, and 1,369 lines of optimizer code across 10 format-specific optimizers. This creates:

- **Speed bottleneck**: Pillow is 3-10x slower than libvips for decode/encode/resize
- **Infrastructure cost**: Multi-stage Dockerfile building jpegli + MozJPEG from source, subprocess overhead, temp files (cwebp)
- **Code complexity**: Each optimizer mixes Pillow calls, subprocess invocations, and format-specific quirks
- **Deployment friction**: 7 CLI tools to install, build, and health-check

## Solution

Replace Pillow + 5 CLI tools with **pyvips** (Python bindings for libvips), compiled with jpegli (the best JPEG encoder) and all optimal codecs. Keep gifsicle and scour for niches where no library alternative exists.

## Phase Plan

- **Phase 1** (this design): Migrate optimizers from Pillow + CLI tools to pyvips
- **Phase 2** (future): Add format conversion layer (e.g., PNG -> WebP, JPEG -> AVIF)

## Key Design Constraint

**Quality must not regress.** Every migration step is validated by benchmark comparison against the current baseline, per-format and per-preset (HIGH/MEDIUM/LOW).

## Architecture

### What Changes

```
BEFORE:                                    AFTER:
Pillow (decode/encode/resize)              pyvips (decode/encode/resize)
+ pngquant (subprocess)                    + gifsicle (subprocess, GIF only)
+ jpegtran (subprocess)                    + scour (Python, SVG only)
+ cjpeg (subprocess)                       + oxipng (Python, PNG enhancement)
+ cwebp (subprocess + temp files)
+ gifsicle (subprocess)
+ scour (Python)
+ oxipng (Python)
+ pillow-avif (Python)
+ pillow-heif (Python)
+ jxlpy (Python)
+ jpegli (built from libjxl source)

7 CLI tools, 7+ Python libraries           1 CLI tool, 3 Python libraries + pyvips
```

### What Stays Unchanged

- `BaseOptimizer` interface: `optimize(data, config) -> OptimizeResult`
- `_build_result()` output-never-larger guarantee
- `OPTIMIZERS` registry and router dispatch
- Quality breakpoints convention (`<50` aggressive, `<70` moderate, `>=70` lossless)
- `CompressionGate` concurrency model (semaphore + queue depth)
- All API endpoints, schemas, middleware
- Benchmark infrastructure (cases, runner, report)

### libvips Codec Stack

pyvips (libvips) compiled with:

| Codec | Purpose | Quality vs Current |
|-------|---------|-------------------|
| jpegli (libjpeg.so.62 from libjxl) | JPEG encode/decode | **Equal** (same encoder, 35% better than mozjpeg) |
| libimagequant | PNG palette quantization | **Identical** (same engine as pngquant) |
| libwebp | WebP encode/decode | **Identical** (same library) |
| libheif + libaom | AVIF encode/decode | **Identical** (same encoder) |
| libheif + x265 | HEIC encode/decode | **Identical** (same encoder) |
| libjxl | JXL encode/decode | **Identical** (same library) |
| cgif | GIF write | New (used for non-animated GIF output) |
| libpng + zlib | PNG lossless | Near-identical (oxipng adds 2-5%, kept as enhancement) |

Kept separately:
- **gifsicle**: Animated GIF inter-frame optimization (no library alternative)
- **scour**: SVG source optimization (libvips rasterizes SVG, can't optimize source)
- **oxipng**: Optional PNG lossless post-processing enhancement (+2-5% squeeze)

## Optimizer Redesign

### Pluggable Enhancement System

Each optimizer has a primary method (pyvips) and optional enhancement methods (specialized tools). All methods run in parallel; the optimizer picks the smallest result.

```python
class BaseOptimizer(ABC):
    def __init__(self):
        self._enhancements: list[Callable] = []

    def register_enhancement(self, fn):
        """Register an enhancement method (specialized tool)."""
        self._enhancements.append(fn)

    async def optimize(self, data, config) -> OptimizeResult:
        # Primary method (pyvips) always runs
        primary_result = await self._primary(data, config)

        # Enhancement methods run in parallel, failures ignored
        enhancement_results = await asyncio.gather(
            *[self._run_enhancement(fn, data, config) for fn in self._enhancements],
            return_exceptions=True,
        )

        # Pick smallest from all candidates
        candidates = [primary_result] + [r for r in enhancement_results if valid]
        best = min(candidates, key=lambda c: len(c.data))
        return self._build_result(data, best.data, best.method)
```

Adding a new tool = write a function, register it. Removing a tool = don't register. No core code changes needed.

### Per-Format Changes

#### JPEG (~30-40 lines, down from 242)

- **Primary**: `img.jpegsave_buffer(Q=quality, optimize_coding=True, strip=strip_metadata)`
  - jpegli under the hood (via libjpeg.so.62)
  - `optimize_coding=True` handles Huffman optimization (replaces jpegtran)
- **Eliminated**: jpegtran subprocess, cjpeg/MozJPEG fallback pipeline, BMP intermediate
- **Quality capping** (`max_reduction`): Binary search stays, single encoder simplifies it
- **Metadata**: pyvips `strip` parameter + ICC profile preservation

#### PNG (~40-50 lines, down from 141)

- **Primary lossy** (`quality < 70`): `img.pngsave_buffer(palette=True, Q=quality, effort=7, dither=1.0)`
  - libimagequant (same engine as pngquant)
  - Quality < 50: `colours=64, effort=10`
  - Quality < 70: `colours=256, effort=7`
- **Primary lossless** (`quality >= 70`): `img.pngsave_buffer(compression=9, effort=10)`
- **Enhancement**: oxipng post-processing on the primary output (+2-5% lossless squeeze)
- **Eliminated**: pngquant subprocess, exit code 99 handling
- **APNG**: pyvips handles multipage/animated natively

#### WebP (~20-30 lines, down from 140)

- **Primary**: `img.webpsave_buffer(Q=quality, effort=4)`
- **Animated**: `save_all=True` via pyvips multipage support
- **Eliminated**: cwebp subprocess, temp files, dual pipeline
- **Quality capping**: Binary search stays

#### GIF (~30 lines, similar to current 37)

- **Primary**: gifsicle subprocess (kept — no alternative for inter-frame optimization)
- Minimal change from current implementation

#### SVG (~50 lines, down from 77)

- **Primary**: scour (kept — no alternative for SVG source optimization)
- Simplify SVGZ detection/handling

#### TIFF (~25-30 lines, down from 91)

- **Primary**: `img.tiffsave_buffer(compression='deflate')` for lossless, `compression='jpeg', Q=quality` for lossy
- **Eliminated**: Manual `img.copy()` for thread safety, triple parallel method
- pyvips handles compression selection and thread safety natively

#### BMP (~20-30 lines, down from 253)

- **Primary**: pyvips decode, palette conversion for quality < 70, raw BMP save
- **Eliminated**: Hand-coded RLE8 encoder, multi-tier strategy, binary state machine
- Phase 2 (format conversion) makes BMP trivial: just convert to PNG/WebP

#### AVIF / HEIC / JXL (~20-30 lines each, down from ~90 each)

- **Primary**: `img.heifsave_buffer(Q=quality, compression='av1')` for AVIF, `compression='hevc'` for HEIC, `img.jxlsave_buffer(Q=quality)` for JXL
- **Metadata**: pyvips `strip` parameter
- **Eliminated**: pillow-avif/pillow-heif/jxlpy Python library imports, separate metadata-strip method

### Code Impact

| Optimizer | Current LoC | Estimated LoC | Reduction |
|-----------|-------------|---------------|-----------|
| JPEG | 242 | 35 | 86% |
| PNG | 141 | 45 | 68% |
| WebP | 140 | 25 | 82% |
| GIF | 37 | 30 | 19% |
| SVG | 77 | 50 | 35% |
| TIFF | 91 | 28 | 69% |
| BMP | 253 | 25 | 90% |
| AVIF | 94 | 25 | 73% |
| HEIC | 90 | 25 | 72% |
| JXL | 94 | 25 | 73% |
| **Total** | **1,369** | **~340** | **~75%** |

## Dockerfile

### Current (3 build stages)

1. jpegli-builder: Build jpegli + cjxl/djxl from libjxl source
2. mozjpeg-builder: Build cjpeg, jpegtran
3. Production: Copy binaries + install pngquant, gifsicle, webp, libheif-dev, etc.

### New (2 build stages)

1. **libvips-builder**: Build libvips from source linked against jpegli, libimagequant, libwebp, libheif, libjxl, cgif
2. **Production**: Copy libvips + codec libraries, install gifsicle, pip install pyvips scour oxipng

What's eliminated:
- MozJPEG build stage (cjpeg, jpegtran no longer needed)
- System packages: pngquant, webp CLI tools
- Python packages: Pillow, pillow-avif-plugin, pillow-heif, jxlpy

## Estimation System Impact

The sample-based estimator (estimation/estimator.py) uses Pillow for decode, resize, and encode in BPP helpers. All calls change from Pillow API to pyvips API. Same logic, different function signatures:

```python
# Before (Pillow):
img = Image.open(io.BytesIO(data))
sample = img.resize((w, h), Image.LANCZOS)
sample.save(buf, format="JPEG", quality=q)

# After (pyvips):
img = pyvips.Image.new_from_buffer(data, '')
sample = img.resize(w / img.width)
result = sample.jpegsave_buffer(Q=q, optimize_coding=True)
```

The estimation architecture stays identical: decode -> resize -> compress sample -> measure BPP -> extrapolate.

## Migration Strategy

### Benchmark-Driven: No Code Ships Without Validation

**Step 0: Capture baseline**
Run `python -m benchmarks.run` and save per-format, per-preset (HIGH/MEDIUM/LOW) data:
- Reduction % (compression quality)
- Latency (speed)
- Method selected
- Estimation accuracy (Avg Err)

**Step 1: Migrate one format at a time** (lowest risk first)

1. BMP — simplest optimizer, biggest code reduction (253 -> ~25)
2. TIFF — simple, pure Pillow currently
3. WebP — eliminates cwebp temp files
4. AVIF/HEIC/JXL — near-identical, migrate together
5. PNG — verify libimagequant = pngquant quality, keep oxipng
6. JPEG — most critical, verify jpegli quality preserved via pyvips
7. GIF — keep gifsicle, minimal change
8. SVG — keep scour, minimal change

**After each step**: Run `python -m benchmarks.run --compare` and verify:
- Reduction % didn't drop by more than 1-2% per format per preset
- Latency improved or stayed neutral
- Preset differentiation maintained (HIGH > MEDIUM > LOW)
- Estimation accuracy stayed within target

**Step 2: Migrate estimation BPP helpers** to pyvips

**Step 3: Simplify Dockerfile** (remove unused build stages and dependencies)

## Expected Outcomes

| Metric | Current | After Migration |
|--------|---------|----------------|
| Optimizer code | 1,369 lines | ~340 lines |
| CLI tools | 7 | 1 (gifsicle) |
| Python deps | 7+ | 3 (pyvips, scour, oxipng) |
| Docker stages | 3 | 2 |
| JPEG quality | jpegli (best) | jpegli (same, via pyvips) |
| Encode/decode speed | Pillow baseline | 3-10x faster (pyvips) |
| Memory usage | Pillow (whole image in RAM) | pyvips (streaming, lower) |
| Flexibility | Ad-hoc per optimizer | Pluggable enhancement system |

## References

- [libvips documentation](https://www.libvips.org/)
- [pyvips documentation](https://libvips.github.io/pyvips/)
- [jpegli vs mozjpeg benchmark](https://arxiv.org/html/2403.18589v1)
- [jpegli blog post](https://giannirosato.com/blog/post/jpegli/)
- [libvips PNG quantization](https://www.libvips.org/API/current/method.Image.pngsave.html)
