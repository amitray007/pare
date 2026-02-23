# pyvips Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate all optimizers from Pillow + CLI tools to pyvips (libvips), reducing 1,369 lines to ~340 lines while maintaining or improving compression quality.

**Architecture:** Each optimizer is rewritten to use pyvips API instead of Pillow + subprocess calls. The migration is done one format at a time (lowest risk first), with benchmark validation after each step. The estimation BPP helpers are migrated last.

**Tech Stack:** pyvips (Python bindings for libvips), libvips compiled with jpegli + libimagequant + libwebp + libheif + libjxl + cgif

**Baseline Benchmark (2026-02-23, commit 2079eb3):**

| Format | HIGH (q=40) | MEDIUM (q=60) | LOW (q=80) | Est Err (avg) |
|--------|-------------|---------------|------------|---------------|
| JPEG | 53.5% | 34.3% | 20.7% | 10.8% |
| PNG | 60.6% | 58.7% | 42.1% | 6.6% |
| WebP | 48.5% | 28.2% | 8.9% | 7.7% |
| AVIF | 47.4% | 25.2% | 7.7% | 0.0% |
| HEIC | 37.8% | 17.1% | 1.8% | 0.0% |
| JXL | 55.0% | 35.8% | 8.8% | 0.0% |
| GIF | 27.8% | 14.9% | 7.7% | 8.2% |
| BMP | 87.8% | 87.8% | 65.7% | 0.0% |
| TIFF | 80.9% | 78.5% | 58.8% | 0.0% |
| SVG | 24.5% | 24.5% | 12.1% | 0.0% |

**Validation criteria per format:**
- Reduction % must not drop by more than 2% per preset
- Preset ordering maintained: HIGH > MEDIUM > LOW
- Estimation error stays under ~15%
- All existing tests pass

---

### Task 1: Dockerfile — Build libvips with jpegli and all codecs

**Files:**
- Modify: `Dockerfile`

**Context:** The current Dockerfile has 3 stages: jpegli-builder, mozjpeg-builder, production. The new Dockerfile has 2 stages: libvips-builder (compiles libvips linked against jpegli + all codecs), production.

**Step 1: Write the new Dockerfile**

Replace the 3-stage Dockerfile with a 2-stage build. Stage 0 builds libvips from source with:
- jpegli (from libjxl, as libjpeg.so.62)
- libimagequant (for PNG palette quantization)
- libwebp (WebP encode/decode)
- libheif + libaom (AVIF) + x265 (HEIC)
- libjxl (JPEG XL)
- cgif (GIF write)
- libpng + zlib (PNG lossless)

Stage 1 is the production image with pyvips installed.

```dockerfile
# ---- Stage 0: Build libvips with jpegli and all codecs ----
FROM debian:bookworm-slim AS libvips-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential git ca-certificates pkg-config nasm \
    meson ninja-build gobject-introspection \
    # Core libvips deps
    libglib2.0-dev libexpat1-dev \
    # PNG
    libpng-dev zlib1g-dev \
    # WebP
    libwebp-dev \
    # HEIF (AVIF + HEIC)
    libheif-dev libaom-dev libde265-dev libx265-dev \
    # TIFF
    libtiff-dev \
    # GIF
    libcgif-dev \
    # libimagequant (for PNG palette quantization)
    libimagequant-dev \
    # Highway (required by libjxl)
    libhwy-dev \
    # Brotli (required by libjxl)
    libbrotli-dev \
    && rm -rf /var/lib/apt/lists/*

# Build libjxl with jpegli
RUN git clone --depth 1 --branch v0.11.1 https://github.com/libjxl/libjxl.git /libjxl \
    && cd /libjxl \
    && git submodule update --init --depth 1 third_party/skcms third_party/libjpeg-turbo \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
             -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_TESTING=OFF \
             -DJPEGXL_ENABLE_TOOLS=OFF \
             -DJPEGXL_ENABLE_DOXYGEN=OFF \
             -DJPEGXL_ENABLE_MANPAGES=OFF \
             -DJPEGXL_ENABLE_BENCHMARK=OFF \
             -DJPEGXL_ENABLE_EXAMPLES=OFF \
             -DJPEGXL_ENABLE_FUZZERS=OFF \
             -DJPEGXL_ENABLE_JPEGLI=ON \
             -DJPEGXL_ENABLE_JPEGLI_LIBJPEG=ON \
             -DJPEGXL_ENABLE_SKCMS=ON \
             -DJPEGXL_ENABLE_SJPEG=OFF \
             -DJPEGXL_ENABLE_OPENEXR=OFF \
             .. \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# Build libvips from source (linked against jpegli + all codecs above)
ARG VIPS_VERSION=8.16.0
RUN curl -L https://github.com/libvips/libvips/releases/download/v${VIPS_VERSION}/vips-${VIPS_VERSION}.tar.xz \
    | tar xJ \
    && cd vips-${VIPS_VERSION} \
    && meson setup build --prefix=/usr/local --buildtype=release \
         -Dintrospection=disabled \
    && cd build \
    && ninja \
    && ninja install \
    && ldconfig

# ---- Stage 1: Production image ----
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/amitray007/pare"
LABEL org.opencontainers.image.description="Serverless image compression API"
LABEL org.opencontainers.image.licenses="MIT"

# Copy libvips and all codec libraries
COPY --from=libvips-builder /usr/local/lib/ /usr/local/lib/
COPY --from=libvips-builder /usr/local/include/ /usr/local/include/
RUN ldconfig

# gifsicle is kept for animated GIF inter-frame optimization
RUN apt-get update && apt-get install -y --no-install-recommends \
    gifsicle \
    libglib2.0-0 \
    libexpat1 \
    libpng16-16 \
    libwebp7 \
    libheif1 \
    libaom3 \
    libde265-0 \
    libx265-199 \
    libtiff6 \
    libcgif0 \
    libimagequant0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
```

**Important notes for implementation:**
- The exact library package names may vary. Check `apt-cache search` in debian:bookworm-slim.
- libvips auto-detects codecs at build time via pkg-config. After building, verify with `vips --vips-config` or `python3 -c "import pyvips; print(pyvips.version(3))"`.
- The critical thing: jpegli's libjpeg.so.62 must be installed BEFORE building libvips so that libvips links against jpegli (not libjpeg-turbo).
- After building, verify jpegli linkage: `ldd /usr/local/lib/libvips.so | grep jpeg` should show the jpegli-built libjpeg.so.62.

**Step 2: Update requirements.txt**

Remove Pillow ecosystem deps. Add pyvips.

```
# Remove these lines:
Pillow>=10.2.0,<11.0.0
pillow-heif>=0.14.0,<1.0.0
jxlpy>=0.9.0,<1.0.0

# Add:
pyvips>=2.2.0,<3.0.0
```

Keep: `pyoxipng`, `scour` (still used as enhancements).

**Note:** Pillow is still used in `estimation/estimator.py` and `routers/estimate.py` (`_fetch_dimensions`). Keep Pillow in requirements.txt until Task 9 (estimation migration). Mark it with a comment:

```
Pillow>=10.2.0,<11.0.0  # TODO: remove after estimation migrated to pyvips (Task 9)
```

**Step 3: Build and test Docker image**

Run: `docker build -t pare-pyvips .`

Verify pyvips works:
```bash
docker run --rm pare-pyvips python -c "
import pyvips
print('pyvips version:', pyvips.version(3))
# Verify jpegli is the JPEG backend
img = pyvips.Image.new_from_buffer(open('/app/tests/sample_images/sample.jpg','rb').read(), '')
buf = img.jpegsave_buffer(Q=80, optimize_coding=True)
print('JPEG encode via pyvips: OK, size:', len(buf))
# Verify PNG quantization (libimagequant)
img2 = pyvips.Image.new_from_buffer(open('/app/tests/sample_images/sample.png','rb').read(), '')
buf2 = img2.pngsave_buffer(palette=True, Q=80)
print('PNG palette encode: OK, size:', len(buf2))
"
```

**Step 4: Commit**

```bash
git add Dockerfile requirements.txt
git commit -m "build: add pyvips + libvips with jpegli to Dockerfile"
```

---

### Task 2: Migrate BMP optimizer to pyvips

**Files:**
- Modify: `optimizers/bmp.py`
- Test: `tests/test_formats.py` (existing BMP tests)

**Context:** BMP is the simplest migration target with the biggest code reduction (253 → ~25 lines). Current BMP optimizer has hand-coded RLE8, multi-tier palette quantization. pyvips can decode BMP and we re-encode to BMP with simpler palette logic.

**Step 1: Write the failing test**

No new tests needed — existing tests in `tests/test_formats.py` cover BMP optimization. The test we verify against:

Run: `pytest tests/test_formats.py -k "bmp" -v`
Expected: Tests pass with current implementation.

**Step 2: Rewrite BMP optimizer**

Replace `optimizers/bmp.py` entirely:

```python
import asyncio
import io

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BmpOptimizer(BaseOptimizer):
    """BMP optimization — quality-aware compression tiers.

    LOW  (quality >= 70): Lossless 32->24 bit downconversion only.
    MEDIUM (quality 50-69): Palette quantization to 256 colors.
    HIGH (quality < 50): Palette quantization to 256 colors (+ future RLE8).

    Each tier tries its methods plus all gentler methods, picks the smallest.
    """

    format = ImageFormat.BMP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, best_method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, best_method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        # Drop alpha if fully opaque
        if img.bands == 4 and img.interpretation == "srgb":
            alpha = img[3]
            if alpha.min() == 255:
                img = img[:3]

        best = data
        best_method = "none"

        # Tier 1 (all presets): lossless 24-bit re-encode
        candidate = img.write_to_buffer(".bmp")
        if len(candidate) < len(best):
            best = candidate
            best_method = "pyvips-bmp"

        # Tier 2 (quality < 70): palette quantization to 256 colors
        if config.quality < 70:
            # Quantize to 8-bit palette using pyvips
            # pyvips doesn't have direct BMP palette save, so we:
            # 1. Save as palette PNG (uses libimagequant)
            # 2. Re-load and save as BMP
            png_buf = img.pngsave_buffer(palette=True, Q=config.quality, effort=1)
            palette_img = pyvips.Image.new_from_buffer(png_buf, "")
            candidate = palette_img.write_to_buffer(".bmp")
            if len(candidate) < len(best):
                best = candidate
                best_method = "pyvips-bmp-palette"

        return best, best_method
```

**Note:** The hand-coded RLE8 encoder is eliminated. BMP RLE8 is a niche optimization that Phase 2 (format conversion) makes irrelevant — BMP images will simply convert to PNG/WebP. The palette quantization path (which handles the HIGH and MEDIUM presets) is kept because it provides the bulk of the savings (66-87% reduction).

**Step 3: Run tests**

Run: `pytest tests/test_formats.py -k "bmp" -v`
Expected: PASS

**Step 4: Run BMP benchmark**

Run: `python -m benchmarks.run --fmt bmp`

Expected: Reduction% within 2% of baseline per preset:
- HIGH: ~87.8%
- MEDIUM: ~87.8%
- LOW: ~65.7%

**Important:** If pyvips BMP palette path produces different results than Pillow's median-cut + RLE8, the numbers may differ. The key check: reduction% didn't regress by more than 2% and preset ordering is maintained.

**Step 5: Commit**

```bash
git add optimizers/bmp.py
git commit -m "refactor(bmp): migrate BMP optimizer from Pillow to pyvips"
```

---

### Task 3: Migrate TIFF optimizer to pyvips

**Files:**
- Modify: `optimizers/tiff.py`
- Test: `tests/test_formats.py` (existing TIFF tests)

**Context:** Current TIFF optimizer (91 lines) runs deflate + LZW + JPEG-in-TIFF concurrently via Pillow. pyvips handles TIFF compression natively with better performance.

**Step 1: Rewrite TIFF optimizer**

Replace `optimizers/tiff.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class TiffOptimizer(BaseOptimizer):
    """TIFF optimization — try multiple compression methods, pick smallest.

    Lossless methods (all presets): deflate, lzw
    Lossy method (quality < 70): jpeg at config.quality

    All methods run concurrently via asyncio.gather.
    """

    format = ImageFormat.TIFF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        img = pyvips.Image.new_from_buffer(data, "")

        strip = config.strip_metadata

        methods = [
            ("deflate", {"compression": "deflate", "strip": strip}),
            ("lzw", {"compression": "lzw", "strip": strip}),
        ]

        # Lossy JPEG-in-TIFF for quality < 70 and compatible band count
        if config.quality < 70 and img.bands in (1, 3):
            methods.append(
                ("tiff_jpeg", {"compression": "jpeg", "Q": config.quality, "strip": strip})
            )

        results = await asyncio.gather(
            *[
                asyncio.to_thread(self._try_compression, img, method_name, save_kwargs)
                for method_name, save_kwargs in methods
            ]
        )

        best, best_method = data, "none"
        for candidate, method in results:
            if candidate is not None and len(candidate) < len(best):
                best, best_method = candidate, method

        return self._build_result(data, best, best_method)

    @staticmethod
    def _try_compression(
        img: pyvips.Image, method_name: str, save_kwargs: dict
    ) -> tuple[bytes | None, str]:
        try:
            return img.tiffsave_buffer(**save_kwargs), method_name
        except Exception:
            return None, method_name
```

**Step 2: Run tests**

Run: `pytest tests/test_formats.py -k "tiff" -v`
Expected: PASS

**Step 3: Run TIFF benchmark**

Run: `python -m benchmarks.run --fmt tiff`

Expected within 2% of baseline:
- HIGH: ~80.9%
- MEDIUM: ~78.5%
- LOW: ~58.8%

**Step 4: Commit**

```bash
git add optimizers/tiff.py
git commit -m "refactor(tiff): migrate TIFF optimizer from Pillow to pyvips"
```

---

### Task 4: Migrate WebP optimizer to pyvips

**Files:**
- Modify: `optimizers/webp.py`
- Test: `tests/test_formats.py` (existing WebP tests)

**Context:** Current WebP optimizer (140 lines) runs Pillow + cwebp CLI (with temp files) concurrently. pyvips uses libwebp directly — no temp files, no subprocess.

**Step 1: Rewrite WebP optimizer**

Replace `optimizers/webp.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class WebpOptimizer(BaseOptimizer):
    """WebP optimization via pyvips (libwebp).

    Pipeline:
    1. Encode at target quality with effort=4
    2. If max_reduction set and exceeded, binary search quality
    3. Enforce output-never-larger guarantee
    """

    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        # Check if animated (multi-page)
        n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
        is_animated = n_pages > 1

        best = self._encode(img, config.quality, is_animated)
        method = "pyvips-webp"

        # Cap reduction if max_reduction is set
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = self._find_capped_quality(img, data, config, is_animated)
                if capped is not None:
                    best = capped

        return best, method

    @staticmethod
    def _encode(img: pyvips.Image, quality: int, animated: bool) -> bytes:
        save_kwargs = {"Q": quality, "effort": 4}
        if animated:
            # For animated WebP, load all pages first
            save_kwargs["page_height"] = img.get("page-height") if img.get_typeof("page-height") else img.height
        return img.webpsave_buffer(**save_kwargs)

    def _find_capped_quality(
        self,
        img: pyvips.Image,
        data: bytes,
        config: OptimizationConfig,
        animated: bool,
    ) -> bytes | None:
        target = config.max_reduction
        orig_size = len(data)

        out_100 = self._encode(img, 100, animated)
        if (1 - len(out_100) / orig_size) * 100 > target:
            return None

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = self._encode(img, mid, animated)
            if (1 - len(out_mid) / orig_size) * 100 > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out
```

**Step 2: Run tests**

Run: `pytest tests/test_formats.py -k "webp" -v`
Expected: PASS

**Step 3: Run WebP benchmark**

Run: `python -m benchmarks.run --fmt webp`

Expected within 2% of baseline:
- HIGH: ~48.5%
- MEDIUM: ~28.2%
- LOW: ~8.9%

**Step 4: Commit**

```bash
git add optimizers/webp.py
git commit -m "refactor(webp): migrate WebP optimizer from Pillow+cwebp to pyvips"
```

---

### Task 5: Migrate AVIF, HEIC, JXL optimizers to pyvips

**Files:**
- Modify: `optimizers/avif.py`
- Modify: `optimizers/heic.py`
- Modify: `optimizers/jxl.py`
- Test: `tests/test_formats.py` (existing tests for all three)

**Context:** These three optimizers share the same pattern (metadata strip + lossy re-encode). All three use Pillow plugins (pillow-avif, pillow-heif, jxlpy). pyvips handles all three natively via libheif and libjxl.

**Step 1: Rewrite AVIF optimizer**

Replace `optimizers/avif.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class AvifOptimizer(BaseOptimizer):
    """AVIF optimization — lossy re-encoding via pyvips (libheif + libaom).

    Quality mapping: avif_quality = max(30, min(90, quality + 10))
    """

    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")
        avif_quality = max(30, min(90, config.quality + 10))

        save_kwargs = {
            "Q": avif_quality,
            "compression": "av1",
            "effort": 4,  # 0=fastest, 9=slowest
            "strip": config.strip_metadata,
        }

        result = img.heifsave_buffer(**save_kwargs)
        return result, "avif-reencode"
```

**Step 2: Rewrite HEIC optimizer**

Replace `optimizers/heic.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class HeicOptimizer(BaseOptimizer):
    """HEIC optimization — lossy re-encoding via pyvips (libheif + x265).

    Quality mapping: heic_quality = max(30, min(90, quality + 10))
    """

    format = ImageFormat.HEIC

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")
        heic_quality = max(30, min(90, config.quality + 10))

        save_kwargs = {
            "Q": heic_quality,
            "compression": "hevc",
            "effort": 4,
            "strip": config.strip_metadata,
        }

        result = img.heifsave_buffer(**save_kwargs)
        return result, "heic-reencode"
```

**Step 3: Rewrite JXL optimizer**

Replace `optimizers/jxl.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class JxlOptimizer(BaseOptimizer):
    """JPEG XL optimization — lossy re-encoding via pyvips (libjxl).

    Quality mapping: jxl_quality = max(30, min(95, quality + 10))
    """

    format = ImageFormat.JXL

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")
        jxl_quality = max(30, min(95, config.quality + 10))

        save_kwargs = {
            "Q": jxl_quality,
            "effort": 7,  # 1=fastest, 9=slowest
            "strip": config.strip_metadata,
        }

        # pyvips detects JXL support via libjxl
        result = img.jxlsave_buffer(**save_kwargs)
        return result, "jxl-reencode"
```

**Step 4: Run tests**

Run: `pytest tests/test_formats.py -k "avif or heic or jxl" -v`
Expected: PASS (JXL tests may skip locally if jxlpy not installed — Docker required)

**Step 5: Run benchmarks**

Run: `python -m benchmarks.run --fmt avif && python -m benchmarks.run --fmt heic && python -m benchmarks.run --fmt jxl`

Expected within 2% of baseline:
- AVIF: HIGH ~47.4%, MEDIUM ~25.2%, LOW ~7.7%
- HEIC: HIGH ~37.8%, MEDIUM ~17.1%, LOW ~1.8%
- JXL: HIGH ~55.0%, MEDIUM ~35.8%, LOW ~8.8%

**Step 6: Commit**

```bash
git add optimizers/avif.py optimizers/heic.py optimizers/jxl.py
git commit -m "refactor(avif,heic,jxl): migrate to pyvips from Pillow plugins"
```

---

### Task 6: Migrate PNG optimizer to pyvips

**Files:**
- Modify: `optimizers/png.py`
- Test: `tests/test_formats.py` (existing PNG tests)

**Context:** Current PNG optimizer (141 lines) runs pngquant (lossy) + oxipng (lossless) in parallel. pyvips has libimagequant built-in for palette quantization. oxipng is kept as an enhancement (+2-5% extra squeeze).

**Step 1: Rewrite PNG optimizer**

Replace `optimizers/png.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat, is_apng


class PngOptimizer(BaseOptimizer):
    """PNG optimization: pyvips (libimagequant lossy) + oxipng (lossless enhancement).

    Pipeline:
    1. If APNG or lossless-only: pyvips lossless encode + oxipng enhancement
    2. Otherwise: pyvips lossy palette + oxipng enhancement, concurrently with lossless
    3. Pick smallest result

    Quality controls:
    - quality < 50: 64 max colors, effort=10 (aggressive)
    - quality < 70: 256 max colors, effort=7 (moderate)
    - quality >= 70: lossless only, oxipng level=2 (gentle)
    """

    format = ImageFormat.PNG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        animated = is_apng(data)
        if animated:
            self.format = ImageFormat.APNG

        strip = config.strip_metadata

        # APNG or lossless-only: skip lossy path
        if animated or not config.png_lossy:
            optimized = await asyncio.to_thread(self._lossless_encode, data, strip)
            # Enhancement: oxipng post-processing
            enhanced = await asyncio.to_thread(self._run_oxipng, optimized, config.quality)
            best = min([optimized, enhanced], key=len)
            return self._build_result(data, best, "oxipng")

        # Lossy + lossless paths run concurrently
        lossy_task = asyncio.to_thread(self._lossy_encode, data, config, strip)
        lossless_task = asyncio.to_thread(self._lossless_with_oxipng, data, config.quality, strip)

        lossy_result, lossless_result = await asyncio.gather(lossy_task, lossless_task)

        # Pick smallest
        candidates = []
        if lossy_result is not None:
            candidates.append((lossy_result, "pngquant + oxipng"))
        candidates.append((lossless_result, "oxipng"))

        best_data, best_method = min(candidates, key=lambda x: len(x[0]))
        return self._build_result(data, best_data, best_method)

    @staticmethod
    def _lossy_encode(data: bytes, config: OptimizationConfig, strip: bool) -> bytes | None:
        """Lossy PNG: pyvips palette quantization (libimagequant) + oxipng."""
        import oxipng

        img = pyvips.Image.new_from_buffer(data, "")

        if config.quality < 50:
            colours = 64
            effort = 10
        else:
            colours = 256
            effort = 7

        try:
            lossy_buf = img.pngsave_buffer(
                palette=True,
                Q=config.quality,
                colours=colours,
                effort=effort,
                dither=1.0,
                strip=strip,
            )
        except Exception:
            return None

        # Post-process with oxipng
        oxipng_level = 4
        return oxipng.optimize_from_memory(lossy_buf, level=oxipng_level)

    @staticmethod
    def _lossless_encode(data: bytes, strip: bool) -> bytes:
        """Lossless PNG encode via pyvips."""
        img = pyvips.Image.new_from_buffer(data, "")
        return img.pngsave_buffer(compression=9, effort=10, strip=strip)

    @staticmethod
    def _lossless_with_oxipng(data: bytes, quality: int, strip: bool) -> bytes:
        """Lossless pyvips encode + oxipng enhancement."""
        import oxipng

        img = pyvips.Image.new_from_buffer(data, "")
        lossless_buf = img.pngsave_buffer(compression=9, effort=10, strip=strip)

        oxipng_level = 4 if quality < 70 else 2
        return oxipng.optimize_from_memory(lossless_buf, level=oxipng_level)

    @staticmethod
    def _run_oxipng(data: bytes, quality: int) -> bytes:
        """Run oxipng for lossless post-processing enhancement."""
        import oxipng

        oxipng_level = 4 if quality < 70 else 2
        return oxipng.optimize_from_memory(data, level=oxipng_level)
```

**Step 2: Run tests**

Run: `pytest tests/test_formats.py -k "png" -v`
Expected: PASS

**Step 3: Run PNG benchmark**

Run: `python -m benchmarks.run --fmt png`

Expected within 2% of baseline:
- HIGH: ~60.6%
- MEDIUM: ~58.7%
- LOW: ~42.1%

**Critical check:** Verify libimagequant palette quality matches pngquant. Both use the same engine, but pyvips may expose different default parameters. Compare sample outputs visually if reduction% differs by more than 2%.

**Step 4: Commit**

```bash
git add optimizers/png.py
git commit -m "refactor(png): migrate PNG optimizer from pngquant+Pillow to pyvips"
```

---

### Task 7: Migrate JPEG optimizer to pyvips

**Files:**
- Modify: `optimizers/jpeg.py`
- Test: `tests/test_formats.py` (existing JPEG tests)

**Context:** This is the most critical optimizer. Current JPEG (242 lines) uses Pillow/jpegli + jpegtran subprocess + legacy cjpeg fallback. pyvips uses jpegli natively (via libjpeg.so.62). The `optimize_coding=True` flag in pyvips replaces jpegtran's lossless Huffman optimization.

**Step 1: Rewrite JPEG optimizer**

Replace `optimizers/jpeg.py`:

```python
import asyncio

import pyvips

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class JpegOptimizer(BaseOptimizer):
    """JPEG optimization via pyvips (jpegli).

    Pipeline:
    1. Encode at target quality with optimize_coding=True (Huffman optimization)
    2. If max_reduction set and exceeded, binary search quality
    3. Enforce output-never-larger guarantee

    jpegli provides 35% better compression than mozjpeg at equivalent quality.
    optimize_coding=True replaces jpegtran's lossless Huffman optimization.
    """

    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        best, method = await asyncio.to_thread(self._optimize_sync, data, config)
        return self._build_result(data, best, method)

    def _optimize_sync(self, data: bytes, config: OptimizationConfig) -> tuple[bytes, str]:
        img = pyvips.Image.new_from_buffer(data, "")

        save_kwargs = {
            "Q": config.quality,
            "optimize_coding": True,
            "strip": config.strip_metadata,
        }

        if not config.strip_metadata:
            # Preserve ICC profile if present
            pass  # pyvips preserves ICC by default when strip=False

        best = img.jpegsave_buffer(**save_kwargs)
        method = "jpegli"

        # Cap reduction if max_reduction is set
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = self._find_capped_quality(img, data, config)
                if capped is not None:
                    best = capped

        return best, method

    def _find_capped_quality(
        self,
        img: pyvips.Image,
        data: bytes,
        config: OptimizationConfig,
    ) -> bytes | None:
        target = config.max_reduction
        orig_size = len(data)

        out_100 = img.jpegsave_buffer(Q=100, optimize_coding=True, strip=config.strip_metadata)
        if (1 - len(out_100) / orig_size) * 100 > target:
            return None

        lo, hi = config.quality, 100
        best_out = out_100

        for _ in range(5):
            if hi - lo <= 1:
                break
            mid = (lo + hi) // 2
            out_mid = img.jpegsave_buffer(
                Q=mid, optimize_coding=True, strip=config.strip_metadata
            )
            if (1 - len(out_mid) / orig_size) * 100 > target:
                lo = mid
            else:
                hi = mid
                best_out = out_mid

        return best_out
```

**Step 2: Run tests**

Run: `pytest tests/test_formats.py -k "jpeg" -v`
Expected: PASS

**Step 3: Run JPEG benchmark**

Run: `python -m benchmarks.run --fmt jpeg`

Expected within 2% of baseline:
- HIGH: ~53.5%
- MEDIUM: ~34.3%
- LOW: ~20.7%

**This is the most critical benchmark.** jpegli via pyvips should produce identical or better results than jpegli via Pillow (same encoder, same libjpeg.so.62). If results differ, check:
1. Is libvips actually using jpegli? Run: `python -c "import pyvips; img = pyvips.Image.black(100,100); buf = img.jpegsave_buffer(Q=80); print(len(buf))"`
2. Compare with Pillow output at same quality to verify jpegli linkage.

**Step 4: Commit**

```bash
git add optimizers/jpeg.py
git commit -m "refactor(jpeg): migrate JPEG optimizer from Pillow+jpegtran to pyvips/jpegli"
```

---

### Task 8: Update GIF and SVG optimizers (minimal changes)

**Files:**
- Modify: `optimizers/gif.py` (no change needed — already uses gifsicle subprocess)
- Modify: `optimizers/svg.py` (no change needed — already uses scour Python library)

**Context:** GIF and SVG optimizers don't use Pillow, so they don't need migration. This task verifies they still work and removes any stale Pillow imports if present.

**Step 1: Verify GIF optimizer has no Pillow imports**

Read `optimizers/gif.py`. It imports only from `optimizers.base`, `schemas`, `utils.format_detect`, `utils.subprocess_runner`. No changes needed.

**Step 2: Verify SVG optimizer has no Pillow imports**

Read `optimizers/svg.py`. It imports from `optimizers.base`, `schemas`, `security.svg_sanitizer`, `utils.format_detect`, `scour.scour`, `gzip`. No changes needed.

**Step 3: Run tests**

Run: `pytest tests/test_formats.py -k "gif or svg" -v`
Expected: PASS

**Step 4: Run benchmarks**

Run: `python -m benchmarks.run --fmt gif && python -m benchmarks.run --fmt svg`

Expected: Identical to baseline (no code changed).

**Step 5: Commit (only if any changes were made)**

No commit expected for this task unless stale imports were found.

---

### Task 9: Migrate estimation BPP helpers to pyvips

**Files:**
- Modify: `estimation/estimator.py`
- Modify: `routers/estimate.py` (lines 116-133: `_fetch_dimensions` uses Pillow)
- Test: `tests/test_estimate.py`

**Context:** The estimator uses Pillow for decode, resize, and encode in BPP helpers. All Pillow calls change to pyvips API. Same logic, different function signatures. The `_fetch_dimensions` helper in `routers/estimate.py` also uses Pillow to parse dimensions from partial HTTP response.

**Step 1: Rewrite `estimation/estimator.py`**

Key changes:
- Replace `from PIL import Image` with `import pyvips`
- Replace `Image.open(io.BytesIO(data))` with `pyvips.Image.new_from_buffer(data, "")`
- Replace `img.resize((w, h), Image.LANCZOS)` with `img.resize(w / img.width)` (pyvips uses scale factor)
- Replace `img.save(buf, format=...)` with format-specific `img.*save_buffer(...)` calls
- Replace `img.size` with `(img.width, img.height)`
- Replace `getattr(img, "n_frames", 1)` with `img.get("n-pages")` check

Replace the entire file:

```python
"""Sample-based estimation engine.

Instead of heuristic prediction, this module compresses a downsized sample
of the image using the actual optimizers and extrapolates BPP (bits per pixel)
to the full image size.

For small images (<150K pixels), SVG, and animated formats, it compresses the
full file for an exact result.
"""

import asyncio

import pyvips

from optimizers.router import optimize_image
from schemas import EstimateResponse, OptimizationConfig
from utils.format_detect import ImageFormat, detect_format

SAMPLE_MAX_WIDTH = 300
JPEG_SAMPLE_MAX_WIDTH = 1200
LOSSY_SAMPLE_MAX_WIDTH = 800
EXACT_PIXEL_THRESHOLD = 150_000


async def estimate(
    data: bytes,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate compression savings by compressing a sample."""
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(data)
    file_size = len(data)

    # SVG/SVGZ: no pixel data — compress the whole file
    if fmt in (ImageFormat.SVG, ImageFormat.SVGZ):
        return await _estimate_exact(data, fmt, config, file_size)

    # Decode image for dimensions and animation detection
    img = await asyncio.to_thread(_open_image, data)
    width = img.width
    height = img.height
    original_pixels = width * height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    # Animated images: compress full file (inter-frame redundancy matters)
    n_pages = img.get("n-pages") if img.get_typeof("n-pages") else 1
    if n_pages > 1:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Small images: compress fully for exact result
    if original_pixels <= EXACT_PIXEL_THRESHOLD:
        return await _estimate_exact(
            data, fmt, config, file_size, width, height, color_type, bit_depth
        )

    # Large raster images: downsample + compress sample + extrapolate BPP
    return await _estimate_by_sample(
        data, img, fmt, config, file_size, width, height, color_type, bit_depth
    )


def _open_image(data: bytes) -> pyvips.Image:
    """Open image with pyvips."""
    return pyvips.Image.new_from_buffer(data, "")


async def _estimate_exact(
    data: bytes,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int = 0,
    height: int = 0,
    color_type: str | None = None,
    bit_depth: int | None = None,
) -> EstimateResponse:
    """Compress the full image with the actual optimizer. Returns exact result."""
    result = await optimize_image(data, config)
    already_optimized = result.method == "none"
    reduction = result.reduction_percent if not already_optimized else 0.0

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=result.optimized_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=already_optimized,
        confidence="high",
    )


async def _estimate_by_sample(
    data: bytes,
    img: pyvips.Image,
    fmt: ImageFormat,
    config: OptimizationConfig,
    file_size: int,
    width: int,
    height: int,
    color_type: str | None,
    bit_depth: int | None,
) -> EstimateResponse:
    """Downsample, compress sample, extrapolate BPP to full image."""
    original_pixels = width * height

    if fmt == ImageFormat.JPEG:
        max_width = JPEG_SAMPLE_MAX_WIDTH
    elif fmt in (
        ImageFormat.HEIC, ImageFormat.AVIF, ImageFormat.JXL,
        ImageFormat.WEBP, ImageFormat.PNG, ImageFormat.APNG,
    ):
        max_width = LOSSY_SAMPLE_MAX_WIDTH
    else:
        max_width = SAMPLE_MAX_WIDTH

    max_width = min(max_width, width)
    ratio = max_width / width
    sample_width = max_width
    sample_height = max(1, int(height * ratio))
    sample_pixels = sample_width * sample_height

    _DIRECT_ENCODE_BPP_FNS = {
        ImageFormat.JPEG: _jpeg_sample_bpp,
        ImageFormat.HEIC: _heic_sample_bpp,
        ImageFormat.AVIF: _avif_sample_bpp,
        ImageFormat.JXL: _jxl_sample_bpp,
        ImageFormat.WEBP: _webp_sample_bpp,
        ImageFormat.PNG: _png_sample_bpp,
        ImageFormat.APNG: _png_sample_bpp,
    }

    bpp_fn = _DIRECT_ENCODE_BPP_FNS.get(fmt)
    if bpp_fn is not None:
        return await _bpp_to_estimate(
            bpp_fn, img, sample_width, sample_height, config,
            original_pixels, file_size, fmt, width, height, color_type, bit_depth,
        )

    # Generic fallback: create sample, run actual optimizer
    sample_data = await asyncio.to_thread(_create_sample, img, sample_width, sample_height, fmt)
    result = await optimize_image(sample_data, config)

    if result.method == "none":
        return EstimateResponse(
            original_size=file_size,
            original_format=fmt.value,
            dimensions={"width": width, "height": height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="high",
        )

    sample_output_bpp = result.optimized_size * 8 / sample_pixels
    estimated_size = min(int(sample_output_bpp * original_pixels / 8), file_size)
    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=reduction == 0,
        confidence="high",
    )


async def _bpp_to_estimate(
    bpp_fn, img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig, original_pixels: int, file_size: int,
    fmt: ImageFormat, width: int, height: int,
    color_type: str | None, bit_depth: int | None,
) -> EstimateResponse:
    output_bpp, method = await asyncio.to_thread(bpp_fn, img, sample_width, sample_height, config)
    estimated_size = min(int(output_bpp * original_pixels / 8), file_size)
    reduction = max(0.0, round((file_size - estimated_size) / file_size * 100, 1))

    return EstimateResponse(
        original_size=file_size,
        original_format=fmt.value,
        dimensions={"width": width, "height": height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=method,
        already_optimized=reduction == 0,
        confidence="high",
    )


def _jpeg_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JPEG sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    save_kwargs = {
        "Q": config.quality,
        "optimize_coding": True,
        "strip": True,
    }

    buf = sample.jpegsave_buffer(**save_kwargs)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "jpegli")


def _heic_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a HEIC sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    heic_quality = max(30, min(90, config.quality + 10))

    buf = sample.heifsave_buffer(Q=heic_quality, compression="hevc", strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "heic-reencode")


def _avif_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode an AVIF sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    avif_quality = max(30, min(90, config.quality + 10))

    buf = sample.heifsave_buffer(Q=avif_quality, compression="av1", effort=4, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "avif-reencode")


def _jxl_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a JXL sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    jxl_quality = max(30, min(95, config.quality + 10))

    buf = sample.jxlsave_buffer(Q=jxl_quality, effort=7, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "jxl-reencode")


def _webp_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a WebP sample at target quality and return output BPP."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    buf = sample.webpsave_buffer(Q=config.quality, effort=4, strip=True)
    sample_pixels = sample_width * sample_height
    return (len(buf) * 8 / sample_pixels, "pyvips-webp")


def _png_sample_bpp(
    img: pyvips.Image, sample_width: int, sample_height: int,
    config: OptimizationConfig,
) -> tuple[float, str]:
    """Encode a PNG sample and return output BPP."""
    import oxipng

    scale = sample_width / img.width
    sample = img.resize(scale)

    # Lossy path: quantize to palette (simulates pngquant via libimagequant)
    if config.png_lossy and config.quality < 70:
        max_colors = 64 if config.quality < 50 else 256
        png_data = sample.pngsave_buffer(
            palette=True, Q=config.quality, colours=max_colors, dither=1.0, strip=True
        )
        method = "pngquant + oxipng"
    else:
        png_data = sample.pngsave_buffer(compression=9, effort=10, strip=True)
        method = "oxipng"

    # oxipng post-processing
    oxipng_level = 4 if config.quality < 70 else 2
    optimized = oxipng.optimize_from_memory(png_data, level=oxipng_level)
    sample_pixels = sample_width * sample_height
    return (len(optimized) * 8 / sample_pixels, method)


def _create_sample(
    img: pyvips.Image, sample_width: int, sample_height: int, fmt: ImageFormat,
) -> bytes:
    """Resize image and encode with minimal compression."""
    scale = sample_width / img.width
    sample = img.resize(scale)

    if fmt == ImageFormat.GIF:
        # Save as GIF via pyvips (cgif)
        return sample.gifsave_buffer()
    elif fmt == ImageFormat.TIFF:
        return sample.tiffsave_buffer(compression="none")
    elif fmt == ImageFormat.BMP:
        return sample.write_to_buffer(".bmp")
    else:
        return sample.pngsave_buffer(compression=0)


async def estimate_from_thumbnail(
    thumbnail_data: bytes,
    original_file_size: int,
    original_width: int,
    original_height: int,
    config: OptimizationConfig | None = None,
) -> EstimateResponse:
    """Estimate using a pre-downsized thumbnail (for large images)."""
    if config is None:
        config = OptimizationConfig()

    fmt = detect_format(thumbnail_data)
    original_pixels = original_width * original_height

    img = await asyncio.to_thread(_open_image, thumbnail_data)
    thumb_width = img.width
    thumb_height = img.height
    thumb_pixels = thumb_width * thumb_height
    color_type = _get_color_type(img)
    bit_depth = _get_bit_depth(img)

    result = await optimize_image(thumbnail_data, config)

    if result.method == "none":
        return EstimateResponse(
            original_size=original_file_size,
            original_format=fmt.value,
            dimensions={"width": original_width, "height": original_height},
            color_type=color_type,
            bit_depth=bit_depth,
            estimated_optimized_size=original_file_size,
            estimated_reduction_percent=0.0,
            optimization_potential="low",
            method="none",
            already_optimized=True,
            confidence="medium",
        )

    thumb_output_bpp = result.optimized_size * 8 / thumb_pixels
    estimated_size = min(int(thumb_output_bpp * original_pixels / 8), original_file_size)
    reduction = max(0.0, round((original_file_size - estimated_size) / original_file_size * 100, 1))

    return EstimateResponse(
        original_size=original_file_size,
        original_format=fmt.value,
        dimensions={"width": original_width, "height": original_height},
        color_type=color_type,
        bit_depth=bit_depth,
        estimated_optimized_size=estimated_size,
        estimated_reduction_percent=reduction,
        optimization_potential=_classify_potential(reduction),
        method=result.method,
        already_optimized=reduction == 0,
        confidence="medium",
    )


def _classify_potential(reduction: float) -> str:
    if reduction >= 30:
        return "high"
    elif reduction >= 10:
        return "medium"
    return "low"


def _get_color_type(img: pyvips.Image) -> str | None:
    """Map pyvips interpretation to color type string."""
    interp = img.interpretation
    bands = img.bands
    mapping = {
        "srgb": "rgba" if bands == 4 else "rgb",
        "rgb": "rgba" if bands == 4 else "rgb",
        "b-w": "grayscale",
        "grey16": "grayscale",
    }
    return mapping.get(interp)


def _get_bit_depth(img: pyvips.Image) -> int | None:
    """Extract bit depth (per channel) from pyvips image."""
    fmt = img.format
    depth_map = {
        "uchar": 8,
        "char": 8,
        "ushort": 16,
        "short": 16,
        "uint": 32,
        "int": 32,
        "float": 32,
        "double": 64,
    }
    return depth_map.get(fmt, 8)
```

**Step 2: Update `routers/estimate.py` `_fetch_dimensions`**

Replace the `_fetch_dimensions` function (lines 109-133) to use pyvips instead of Pillow:

```python
async def _fetch_dimensions(url: str, is_authenticated: bool) -> tuple[int, int]:
    """Fetch just enough of the image to parse dimensions."""
    import httpx
    import pyvips

    from security.ssrf import validate_url

    validate_url(url)

    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=False) as client:
            resp = await client.get(url, headers={"Range": "bytes=0-8191"})
            partial = resp.content
            img = pyvips.Image.new_from_buffer(partial, "")
            return (img.width, img.height)
    except Exception:
        data = await fetch_image(url, is_authenticated=is_authenticated)
        img = pyvips.Image.new_from_buffer(data, "")
        return (img.width, img.height)
```

Also remove `from PIL import Image` import from `routers/estimate.py` and the `import io` if no longer needed.

**Step 3: Run tests**

Run: `pytest tests/test_estimate.py -v`
Expected: PASS

**Step 4: Run full estimation benchmark**

Run: `python -m benchmarks.run`

Check estimation accuracy columns — all formats should have Avg Err within 2% of baseline values.

**Step 5: Commit**

```bash
git add estimation/estimator.py routers/estimate.py
git commit -m "refactor(estimation): migrate estimation BPP helpers from Pillow to pyvips"
```

---

### Task 10: Update health check and remove dead dependencies

**Files:**
- Modify: `routers/health.py`
- Modify: `requirements.txt`
- Modify: `optimizers/router.py` (verify imports still work)

**Context:** The health check currently checks for CLI tools (pngquant, jpegtran, cwebp) and Python libraries (Pillow, pillow_heif, jxl_plugin). After migration, most of these are replaced by pyvips.

**Step 1: Update health check**

Replace `routers/health.py`:

```python
import shutil

from fastapi import APIRouter

from schemas import HealthResponse

router = APIRouter()


def check_tools() -> dict[str, bool]:
    """Check availability of all required tools and libraries."""
    results = {}

    # Core: pyvips (libvips)
    try:
        import pyvips
        results["pyvips"] = True
        # Verify key codecs are available
        # jpegsave requires libjpeg (jpegli)
        results["jpegli"] = pyvips.type_find("VipsForeignSave", "jpegsave_buffer") != 0
        # heifsave requires libheif
        results["libheif"] = pyvips.type_find("VipsForeignSave", "heifsave_buffer") != 0
        # jxlsave requires libjxl
        results["libjxl"] = pyvips.type_find("VipsForeignSave", "jxlsave_buffer") != 0
        # webpsave requires libwebp
        results["libwebp"] = pyvips.type_find("VipsForeignSave", "webpsave_buffer") != 0
    except ImportError:
        results["pyvips"] = False

    # CLI tools (only gifsicle remains)
    results["gifsicle"] = bool(shutil.which("gifsicle"))

    # Python libraries
    try:
        import oxipng  # noqa: F401
        results["oxipng"] = True
    except ImportError:
        results["oxipng"] = False
    try:
        import scour  # noqa: F401
        results["scour"] = True
    except ImportError:
        results["scour"] = False

    return results


@router.get("/health", response_model=HealthResponse)
async def health():
    tools = check_tools()
    all_available = all(tools.values())
    return HealthResponse(
        status="ok" if all_available else "degraded",
        tools=tools,
        version="0.1.0",
    )
```

**Step 2: Clean up requirements.txt**

Final requirements.txt:

```
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
pyvips>=2.2.0,<3.0.0
pyoxipng>=9.0.0,<10.0.0
scour>=0.38.0,<1.0.0
python-multipart>=0.0.6,<1.0.0
httpx>=0.27.0,<1.0.0
google-cloud-storage>=2.14.0,<3.0.0
defusedxml>=0.7.0,<1.0.0
redis[hiredis]>=5.0.0,<6.0.0
pydantic-settings>=2.1.0,<3.0.0
```

Removed: `Pillow`, `pillow-heif`, `jxlpy`.

**Step 3: Remove stale Pillow imports from any remaining files**

Search entire codebase for `from PIL` or `import PIL` or `import pillow` and remove any remaining references. Key files to check:
- `estimation/estimator.py` (should already be migrated in Task 9)
- `routers/estimate.py` (should already be migrated in Task 9)
- `tests/conftest.py` — the `sample_jxl` fixture uses Pillow + jxlpy to generate JXL test images. This needs updating to use pyvips.

Update `tests/conftest.py` `sample_jxl` fixture:

```python
@pytest.fixture
def sample_jxl():
    """Generate a JXL sample in-memory via pyvips."""
    try:
        import pyvips
        img = pyvips.Image.black(64, 64) + [100, 150, 200]
        return img.jxlsave_buffer(Q=90)
    except Exception:
        pytest.skip("pyvips JXL support not available")
```

**Step 4: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add routers/health.py requirements.txt tests/conftest.py
git commit -m "chore: update health check and remove Pillow dependencies"
```

---

### Task 11: Update CLAUDE.md documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `optimizers/CLAUDE.md`
- Modify: `estimation/CLAUDE.md`

**Context:** The CLAUDE.md files reference Pillow, CLI tools, and the old optimizer patterns. Update to reflect pyvips.

**Step 1: Update root `CLAUDE.md`**

Key changes:
- Replace "Pillow" with "pyvips" in the description
- Remove references to pngquant, jpegtran, cjpeg, cwebp, cjxl, djxl CLI tools
- Update "Docker" section to mention 2-stage build
- Update "CLI tools via stdin/stdout" convention (only gifsicle remains)
- Update dependency list

**Step 2: Update `optimizers/CLAUDE.md`**

Key changes:
- Replace "Pillow + CLI tools" with "pyvips"
- Update CLI tools section (only gifsicle remains)
- Update Python libraries section (pyvips replaces Pillow, pillow-heif, jxlpy, pillow-avif-plugin)
- Remove `img.copy()` thread safety note (pyvips handles this)

**Step 3: Update `estimation/CLAUDE.md`**

Key changes:
- Replace "Pillow" with "pyvips" in all BPP helper descriptions
- Update function signatures in table

**Step 4: Commit**

```bash
git add CLAUDE.md optimizers/CLAUDE.md estimation/CLAUDE.md
git commit -m "docs: update CLAUDE.md files for pyvips migration"
```

---

### Task 12: Final benchmark validation

**Files:** None (read-only verification)

**Context:** Run the complete benchmark suite and compare against the baseline captured at the top of this plan. This is the final gate before merging.

**Step 1: Capture baseline for comparison**

The baseline is already saved as `reports/benchmark-20260223-074342.json`.

**Step 2: Run full benchmark**

Run: `python -m benchmarks.run`

**Step 3: Compare against baseline**

Run: `python -m benchmarks.run --compare`

**Validation criteria (per format, per preset):**
- Reduction % must not drop by more than 2%
- Preset ordering maintained: HIGH > MEDIUM > LOW
- Estimation error (Avg Err) stays under ~15%
- No test failures (cases_failed = 0)

**Step 4: If any format regresses**

1. Identify which format and preset regressed
2. Compare method names (e.g., did pyvips select a different method?)
3. Check pyvips save parameters match the design
4. Adjust parameters if needed, re-run that format's benchmark

**Step 5: Commit benchmark report**

No code changes expected. If all passes, the migration is complete.

---

## Execution Notes

### Docker Requirement

Tasks 2-10 require the Docker image from Task 1 to be built and working. Run `docker build -t pare-pyvips .` first, then execute all optimizer tasks inside the container or with pyvips installed locally.

### Testing Locally vs Docker

- **Locally (no Docker):** pyvips can be installed via pip, but libvips must be installed via system package manager. The jpegli codec will NOT be available locally — libjpeg-turbo will be used instead. Tests will pass but benchmark numbers for JPEG may differ.
- **In Docker:** Full codec stack including jpegli. This is where benchmark validation matters.

### Rollback Strategy

Each task is committed separately. If a format regresses:
1. `git revert <commit>` for that format's migration
2. The old Pillow-based optimizer is restored
3. Other migrated formats are unaffected

### Migration Order Rationale

1. **BMP first** — simplest, no codec concerns, biggest code reduction
2. **TIFF** — simple, pure Pillow → pyvips swap
3. **WebP** — eliminates temp files and cwebp subprocess
4. **AVIF/HEIC/JXL** — near-identical pattern, migrate together
5. **PNG** — verify libimagequant = pngquant quality
6. **JPEG** — most critical, verify jpegli preservation
7. **GIF/SVG** — no change, just verification
8. **Estimation** — depends on all optimizers being migrated
9. **Health/cleanup** — final dependency removal
