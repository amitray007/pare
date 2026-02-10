# Phase 2 — Optimization Engine

## Objectives

- Implement format detection via magic bytes (including APNG detection)
- Build all 11 format-specific optimizer implementations
- Create the optimizer router that dispatches to the correct optimizer
- Implement async subprocess runner for CLI tools (stdin/stdout piping, no temp files)
- Implement selective metadata handling (preserve orientation + ICC, strip GPS/camera)
- Handle animated images (GIF, animated WebP, APNG) without frame destruction

## Deliverables

- `utils/format_detect.py` — magic byte detection for all 11 formats
- `utils/metadata.py` — selective EXIF/ICC stripping
- `utils/subprocess_runner.py` — async subprocess with piping and timeout
- `optimizers/base.py` — abstract base class
- `optimizers/router.py` — format → optimizer dispatch
- `optimizers/png.py`, `jpeg.py`, `webp.py`, `gif.py`, `svg.py`, `avif.py`, `heic.py`, `passthrough.py`

## Dependencies

- Phase 1 (project skeleton, config, schemas, exceptions)

---

## Files to Create

### 1. `utils/format_detect.py`

**Purpose:** Detect image format from magic bytes (file header). Never trust file extensions or Content-Type headers.

**Magic byte table:**

| Format | Magic Bytes | Offset | Length |
|--------|------------|--------|--------|
| PNG | `\x89PNG\r\n\x1a\n` | 0 | 8 |
| JPEG | `\xFF\xD8\xFF` | 0 | 3 |
| GIF87a | `GIF87a` | 0 | 6 |
| GIF89a | `GIF89a` | 0 | 6 |
| WebP | `RIFF` + `WEBP` at offset 8 | 0, 8 | 4, 4 |
| BMP | `BM` | 0 | 2 |
| TIFF (LE) | `II\x2a\x00` | 0 | 4 |
| TIFF (BE) | `MM\x00\x2a` | 0 | 4 |
| PSD | `8BPS` | 0 | 4 |
| AVIF | `ftyp` at offset 4, then brand `avif`/`avis` | 4, 8 | 4, 4 |
| HEIC | `ftyp` at offset 4, then brand `heic`/`heix`/`mif1` | 4, 8 | 4, 4 |
| SVG | `<?xml` or `<svg` (after stripping BOM/whitespace) | 0 | 5/4 |
| SVGZ | gzip header `\x1f\x8b` + decompress → SVG check | 0 | 2 |

**Key function signatures:**

```python
class ImageFormat(str, Enum):
    PNG = "png"
    APNG = "apng"
    JPEG = "jpeg"
    WEBP = "webp"
    GIF = "gif"
    SVG = "svg"
    SVGZ = "svgz"
    AVIF = "avif"
    HEIC = "heic"
    TIFF = "tiff"
    BMP = "bmp"
    PSD = "psd"


def detect_format(data: bytes) -> ImageFormat:
    """Detect image format from magic bytes.

    Args:
        data: Raw image bytes (at least first 32 bytes needed).

    Returns:
        ImageFormat enum value.

    Raises:
        UnsupportedFormatError: If no known format matches.
    """


def is_apng(data: bytes) -> bool:
    """Check if PNG data contains an acTL (animation control) chunk.

    Scans PNG chunks after the IHDR for an acTL chunk (bytes: b'acTL').
    Must be called only after confirming format is PNG.

    Returns:
        True if acTL chunk found (animated PNG), False otherwise.
    """
```

**APNG detection algorithm:**

```
1. Verify PNG signature (first 8 bytes)
2. Skip IHDR chunk (8 bytes length + 4 bytes type + data + 4 bytes CRC)
3. Iterate subsequent chunks:
   a. Read 4-byte length (big-endian uint32)
   b. Read 4-byte chunk type
   c. If chunk type == b'acTL' → return True (is APNG)
   d. If chunk type == b'IDAT' → return False (data started, no acTL found)
   e. Skip chunk data + 4-byte CRC, continue
4. Return False (reached end without acTL)
```

**AVIF vs HEIC detection (ISO BMFF ftyp box):**

```
1. Read bytes 4-7: must be b'ftyp'
2. Read bytes 8-11: major brand
   - b'avif' or b'avis' → AVIF
   - b'heic' or b'heix' or b'mif1' → HEIC
3. If neither → check compatible brands list (bytes 12+, 4 bytes each)
```

---

### 2. `utils/subprocess_runner.py`

**Purpose:** Run CLI compression tools via async subprocess with stdin/stdout piping. No temp files — everything stays in memory.

```python
import asyncio
from config import settings
from exceptions import ToolTimeoutError


async def run_tool(
    cmd: list[str],
    input_data: bytes,
    timeout: int | None = None,
) -> bytes:
    """Run a CLI tool with stdin/stdout piping.

    Args:
        cmd: Command and arguments (e.g., ["pngquant", "--quality", "65-80", "-"]).
        input_data: Raw bytes to pipe to stdin.
        timeout: Seconds before killing the process. Defaults to settings.tool_timeout_seconds.

    Returns:
        Raw bytes from stdout.

    Raises:
        ToolTimeoutError: If the process exceeds the timeout.
        OptimizationError: If the process exits with a non-zero code
                          (except known codes like pngquant exit 99).
    """
    if timeout is None:
        timeout = settings.tool_timeout_seconds

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_data),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolTimeoutError(
            f"Tool {cmd[0]} timed out after {timeout}s",
            tool=cmd[0],
        )

    return stdout, stderr, proc.returncode
```

**Piping patterns for each tool:**

| Tool | stdin | stdout | CLI flags for piping |
|------|-------|--------|---------------------|
| pngquant | PNG bytes | PNG bytes | `--quality {range} - --output -` |
| cjpeg (MozJPEG) | BMP/PPM bytes | JPEG bytes | `-quality {q}` (reads stdin by default) |
| jpegtran | JPEG bytes | JPEG bytes | `-optimize -copy none` (reads stdin) |
| gifsicle | GIF bytes | GIF bytes | `--optimize=3 -` → stdout |
| cwebp | — | — | Does NOT support stdin/stdout; use Pillow instead |

**Note on cwebp:** The `cwebp` tool does not support stdin/stdout piping. The WebP optimizer uses Pillow as the primary path and only falls back to cwebp via temp files if Pillow's result is poor (>=90% of input size).

---

### 3. `utils/metadata.py`

**Purpose:** Selective EXIF/ICC metadata handling per PRD Section 3.9.

**Algorithm:**

```
IF strip_metadata is False:
    Return image unchanged

FOR the detected format:
    CASE JPEG:
        1. Read EXIF via Pillow
        2. Preserve: Orientation tag (0x0112), ICC profile
        3. Strip: GPS (IFD 0x8825), Camera info, XMP, IPTC, thumbnails, comments
        4. Write back preserved EXIF + ICC

    CASE PNG:
        1. Iterate PNG chunks
        2. Preserve: iCCP (ICC profile), pHYs (physical dimensions)
        3. Strip: tEXt, iTXt, zTXt chunks (metadata text)
        4. Reassemble PNG

    CASE TIFF:
        1. Similar to JPEG EXIF handling via Pillow

    CASE AVIF/HEIC:
        1. Use pillow-heif to read
        2. Strip XMP/EXIF metadata, preserve ICC

    DEFAULT (WebP, GIF, SVG, BMP, PSD):
        Tool-specific handling (metadata stripped during optimization)
```

**Key function:**

```python
def strip_metadata_selective(
    data: bytes,
    fmt: ImageFormat,
    preserve_orientation: bool = True,
    preserve_icc: bool = True,
) -> bytes:
    """Strip non-essential metadata while preserving critical fields.

    Preserves:
        - EXIF Orientation tag (prevents rotated images)
        - ICC Color Profile (prevents color degradation)

    Strips:
        - GPS / Location data (privacy)
        - Camera/Device info
        - XMP / IPTC editorial metadata
        - Embedded thumbnails
        - Comments
    """
```

---

### 4. `optimizers/base.py`

**Purpose:** Abstract base class for all format-specific optimizers.

```python
from abc import ABC, abstractmethod
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class BaseOptimizer(ABC):
    """Abstract base for format-specific optimizers."""

    format: ImageFormat

    @abstractmethod
    async def optimize(
        self,
        data: bytes,
        config: OptimizationConfig,
    ) -> OptimizeResult:
        """Optimize image bytes.

        Args:
            data: Raw input image bytes.
            config: Optimization parameters from the request.

        Returns:
            OptimizeResult with optimized bytes and stats.
        """

    def _build_result(
        self,
        original: bytes,
        optimized: bytes,
        method: str,
    ) -> OptimizeResult:
        """Build a result, enforcing the optimization guarantee.

        If optimized >= original, returns original with 0% reduction.
        """
        original_size = len(original)
        optimized_size = len(optimized)

        if optimized_size >= original_size:
            return OptimizeResult(
                success=True,
                original_size=original_size,
                optimized_size=original_size,
                reduction_percent=0.0,
                format=self.format.value,
                method="none",
                optimized_bytes=original,
                message="Image is already optimized",
            )

        reduction = round(
            (1 - optimized_size / original_size) * 100, 1
        )
        return OptimizeResult(
            success=True,
            original_size=original_size,
            optimized_size=optimized_size,
            reduction_percent=reduction,
            format=self.format.value,
            method=method,
            optimized_bytes=optimized,
        )
```

---

### 5. `optimizers/router.py`

**Purpose:** Detect format, dispatch to the correct optimizer. Central routing logic.

```python
from utils.format_detect import detect_format, ImageFormat
from optimizers.png import PngOptimizer
from optimizers.jpeg import JpegOptimizer
from optimizers.webp import WebpOptimizer
from optimizers.gif import GifOptimizer
from optimizers.svg import SvgOptimizer
from optimizers.avif import AvifOptimizer
from optimizers.heic import HeicOptimizer
from optimizers.passthrough import PassthroughOptimizer
from schemas import OptimizationConfig, OptimizeResult


# Optimizer registry — initialized once at import time
OPTIMIZERS = {
    ImageFormat.PNG: PngOptimizer(),
    ImageFormat.APNG: PngOptimizer(),  # APNG uses PNG optimizer with safety flag
    ImageFormat.JPEG: JpegOptimizer(),
    ImageFormat.WEBP: WebpOptimizer(),
    ImageFormat.GIF: GifOptimizer(),
    ImageFormat.SVG: SvgOptimizer(),
    ImageFormat.SVGZ: SvgOptimizer(),  # SVGZ handled inside SvgOptimizer
    ImageFormat.AVIF: AvifOptimizer(),
    ImageFormat.HEIC: HeicOptimizer(),
    ImageFormat.TIFF: PassthroughOptimizer(ImageFormat.TIFF),
    ImageFormat.BMP: PassthroughOptimizer(ImageFormat.BMP),
    ImageFormat.PSD: PassthroughOptimizer(ImageFormat.PSD),
}


async def optimize_image(
    data: bytes,
    config: OptimizationConfig,
) -> OptimizeResult:
    """Detect format and dispatch to the correct optimizer.

    Args:
        data: Raw image bytes.
        config: Optimization parameters.

    Returns:
        OptimizeResult with optimized bytes and stats.

    Raises:
        UnsupportedFormatError: If format is not recognized.
    """
    fmt = detect_format(data)
    optimizer = OPTIMIZERS[fmt]
    return await optimizer.optimize(data, config)
```

---

### 6. `optimizers/png.py` — PNG Pipeline

**Pipeline flowchart (PRD Section 9.1):**

```
Input PNG bytes
    │
    ├── Is APNG? (acTL chunk detected)
    │   ├── YES → oxipng only (lossless, preserves animation frames)
    │   └── NO → continue
    │
    ├── png_lossy == False?
    │   ├── YES → oxipng only (user requested lossless)
    │   └── NO → continue
    │
    └── pngquant --quality {q-15}-{q} - --output -
        │
        ├── Exit code 0 → oxipng on pngquant output (squeeze extra bytes)
        ├── Exit code 99 → quality threshold not met → oxipng on original (lossless)
        └── Other exit code → error
```

**Key class:**

```python
class PngOptimizer(BaseOptimizer):
    format = ImageFormat.PNG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """PNG optimization pipeline.

        Strategy:
        1. Check for APNG — if animated, skip pngquant (destroys frames)
        2. Check png_lossy flag — if False, skip pngquant
        3. Run pngquant for lossy quantization
        4. Run oxipng for lossless optimization on the result
        5. Enforce optimization guarantee (never return larger output)
        """

    async def _run_pngquant(self, data: bytes, quality: int) -> tuple[bytes | None, bool]:
        """Run pngquant with quality range.

        Returns:
            (output_bytes, success) — success=False means exit code 99 (quality too low).
        """
        q_floor = max(1, quality - 15)
        q_ceil = quality
        # pngquant --quality {floor}-{ceil} - --output -
        stdout, stderr, returncode = await run_tool(
            ["pngquant", "--quality", f"{q_floor}-{q_ceil}", "-", "--output", "-"],
            data,
        )
        if returncode == 99:
            return None, False  # Quality threshold not met
        if returncode != 0:
            raise OptimizationError(f"pngquant failed: {stderr.decode()}")
        return stdout, True

    def _run_oxipng(self, data: bytes) -> bytes:
        """Run oxipng (in-process via pyoxipng library)."""
        return pyoxipng.optimize_from_memory(data)
```

---

### 7. `optimizers/jpeg.py` — JPEG Pipeline

**Pipeline flowchart (PRD Section 9.2):**

```
Input JPEG bytes
    │
    ├── Analyze quantization tables → estimate input quality
    │
    ├── Input quality <= target quality?
    │   ├── YES → jpegtran only (lossless Huffman optimization)
    │   │         Flags: -optimize -copy none
    │   │         Optional: -progressive if progressive_jpeg=True
    │   └── NO → MozJPEG cjpeg lossy re-encode
    │             Requires decode to BMP/PPM first (Pillow), then pipe to cjpeg
    │
    └── Compare sizes, enforce optimization guarantee
```

**Key implementation detail — cjpeg input format:**

MozJPEG's `cjpeg` does not accept JPEG input — it accepts BMP, PPM, or Targa. The pipeline must:
1. Decode JPEG to raw pixels via Pillow
2. Save as BMP to a BytesIO buffer
3. Pipe BMP bytes to `cjpeg -quality {q}`
4. Receive optimized JPEG from stdout

For lossless path, `jpegtran` accepts JPEG directly.

```python
class JpegOptimizer(BaseOptimizer):
    format = ImageFormat.JPEG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """JPEG optimization pipeline."""

    def _estimate_jpeg_quality(self, data: bytes) -> int:
        """Estimate input JPEG quality from quantization tables.

        Uses Pillow to read quantization tables, then compares
        against standard JPEG quantization matrices to estimate
        the quality level (1-100).

        Returns:
            Estimated quality (1-100). Returns 100 if cannot determine.
        """

    async def _run_cjpeg(self, bmp_data: bytes, quality: int, progressive: bool) -> bytes:
        """Run MozJPEG cjpeg on BMP input.

        Args:
            bmp_data: BMP-format bytes (decoded from JPEG via Pillow).
            quality: Target quality (1-100).
            progressive: Whether to encode as progressive JPEG.
        """
        cmd = ["cjpeg", f"-quality", str(quality)]
        if progressive:
            cmd.append("-progressive")
        stdout, stderr, rc = await run_tool(cmd, bmp_data)
        return stdout

    async def _run_jpegtran(self, data: bytes, progressive: bool) -> bytes:
        """Run jpegtran for lossless optimization.

        Optimizes Huffman tables without re-encoding.
        """
        cmd = ["jpegtran", "-optimize", "-copy", "none"]
        if progressive:
            cmd.append("-progressive")
        stdout, stderr, rc = await run_tool(cmd, data)
        return stdout
```

---

### 8. `optimizers/webp.py` — WebP Pipeline

**Pipeline flowchart (PRD Section 9.3):**

```
Input WebP bytes
    │
    ├── Detect animated WebP (Pillow: im.n_frames > 1)
    │   ├── YES → Pillow re-encode preserving all frames
    │   └── NO → continue
    │
    ├── Pillow decode + re-encode at target quality
    │
    ├── Result >= 90% of input size?
    │   ├── YES → try cwebp CLI as fallback (temp file required)
    │   │         Use whichever output is smaller
    │   └── NO → use Pillow result
    │
    └── Enforce optimization guarantee
```

```python
class WebpOptimizer(BaseOptimizer):
    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """WebP optimization: Pillow first, cwebp fallback if poor result."""

    def _pillow_optimize(self, data: bytes, quality: int) -> bytes:
        """In-process WebP optimization via Pillow.

        Handles both static and animated WebP.
        For animated: preserves all frames via save_all=True.
        """

    async def _cwebp_fallback(self, data: bytes, quality: int) -> bytes | None:
        """Fallback to cwebp CLI. Requires temp files since cwebp
        doesn't support stdin/stdout. Returns None if cwebp unavailable."""
```

---

### 9. `optimizers/gif.py` — GIF Pipeline

**Pipeline (PRD Section 9.4):**

```
Input GIF bytes → gifsicle --optimize=3 → Output GIF bytes
```

Quality parameter is ignored — gifsicle performs lossless frame optimization only.

```python
class GifOptimizer(BaseOptimizer):
    format = ImageFormat.GIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """GIF optimization via gifsicle.

        gifsicle --optimize=3 performs:
        - Shrinks frame bounding boxes
        - Optimizes frame disposal methods
        - Re-compresses LZW per frame
        """
        stdout, stderr, rc = await run_tool(
            ["gifsicle", "--optimize=3"],
            data,
        )
        return self._build_result(data, stdout, "gifsicle")
```

---

### 10. `optimizers/svg.py` — SVG/SVGZ Pipeline

**Pipeline (PRD Sections 9.5, 9.6):**

```
SVG:  Input SVG → sanitize → scour → Output SVG
SVGZ: Input SVGZ → gunzip → sanitize → scour → gzip → Output SVGZ
```

SVG sanitization is applied before optimization (Phase 4 details the sanitizer, but the optimizer calls it here).

```python
class SvgOptimizer(BaseOptimizer):
    format = ImageFormat.SVG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """SVG/SVGZ optimization via scour.

        For SVGZ: decompress → optimize → recompress.
        Sanitization (script stripping, XXE prevention) is applied
        before optimization — see security/svg_sanitizer.py.
        """

    def _run_scour(self, svg_text: str) -> str:
        """Run scour in-process (Python library, no subprocess).

        Scour options:
        - remove-metadata=True
        - enable-viewboxing=True
        - strip-xml-prolog=True
        - remove-descriptive-elements=True
        - enable-comment-stripping=True
        - shorten-ids=True
        - indent=none
        """
```

---

### 11. `optimizers/avif.py` — AVIF Pipeline

**Pipeline (PRD Section 9.7):**

```
Input AVIF → strip metadata only (no decode/re-encode) → Output AVIF
```

No re-encoding — AVIF is lossy, and decode→re-encode causes generation loss.

```python
class AvifOptimizer(BaseOptimizer):
    format = ImageFormat.AVIF

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """AVIF lossless optimization — metadata stripping only.

        Does NOT decode + re-encode (would cause generation loss).
        If no metadata to strip, returns original with 0% reduction.
        """
```

---

### 12. `optimizers/heic.py` — HEIC Pipeline

Identical approach to AVIF — metadata strip only, no re-encode.

```python
class HeicOptimizer(BaseOptimizer):
    format = ImageFormat.HEIC

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """HEIC lossless optimization — metadata stripping only."""
```

---

### 13. `optimizers/passthrough.py` — TIFF/BMP/PSD

**Pipeline (PRD Section 9.8):**

```
Input TIFF/BMP/PSD → Pillow decode → re-encode with optimization → compare sizes
    → If smaller: return optimized
    → If same/larger: return original with 0% reduction
```

```python
class PassthroughOptimizer(BaseOptimizer):
    """Best-effort optimization for TIFF, BMP, PSD via Pillow."""

    def __init__(self, fmt: ImageFormat):
        self.format = fmt

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        """Attempt Pillow optimization. Return original if no improvement."""
```

---

## Quality Parameter Mapping (Reference)

Implemented inside each optimizer, mapping the unified `quality` param (1-100):

| Format | Internal Mapping | Implementation |
|--------|-----------------|----------------|
| PNG | pngquant `--quality {q-15}-{q}` | `PngOptimizer._run_pngquant()` |
| JPEG | cjpeg `-quality {q}` | `JpegOptimizer._run_cjpeg()` |
| WebP | Pillow `quality={q}` / cwebp `-q {q}` | `WebpOptimizer._pillow_optimize()` |
| GIF | Ignored (gifsicle is lossless) | `GifOptimizer.optimize()` |
| SVG | Ignored (scour optimizes structure) | `SvgOptimizer.optimize()` |
| AVIF | `quality={q}` via pillow-heif (metadata only) | `AvifOptimizer.optimize()` |
| HEIC | `quality={q}` via pillow-heif (metadata only) | `HeicOptimizer.optimize()` |

---

## Environment Variables Introduced

| Variable | Default | Description |
|----------|---------|-------------|
| `TOOL_TIMEOUT_SECONDS` | `60` | Per-tool subprocess timeout |

---

## Verification Steps

### Manual verification

```bash
# Test format detection with sample images from test framework
docker run --rm -v /path/to/test-images:/images pare:dev python -c "
from utils.format_detect import detect_format
import os
for f in os.listdir('/images'):
    data = open(f'/images/{f}', 'rb').read()
    fmt = detect_format(data)
    print(f'{f}: {fmt}')
"

# Test PNG optimization pipeline
curl -X POST http://localhost:8080/optimize \
  -F "file=@sample.png" \
  -o optimized.png
# Compare file sizes: ls -la sample.png optimized.png

# Test JPEG quality estimation
docker run --rm pare:dev python -c "
from optimizers.jpeg import JpegOptimizer
opt = JpegOptimizer()
# Test with known quality JPEG
"

# Verify APNG detection doesn't destroy frames
# Use an animated PNG and verify frame count is preserved after optimization
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_detect_all_formats` | All 13 sample images detected correctly |
| `test_detect_apng` | APNG detected by acTL chunk, static PNG is not APNG |
| `test_detect_avif_vs_heic` | ftyp box correctly distinguishes AVIF from HEIC |
| `test_detect_svgz` | Gzipped SVG detected as SVGZ |
| `test_detect_unknown_rejects` | Random bytes raise UnsupportedFormatError |
| `test_png_lossy_pipeline` | pngquant + oxipng produces smaller output |
| `test_png_lossless_only` | `png_lossy=False` skips pngquant, uses oxipng only |
| `test_png_apng_safety` | APNG input → pngquant skipped, frames preserved |
| `test_png_quality_99_fallback` | pngquant exit 99 → falls back to oxipng |
| `test_jpeg_lossy_reencode` | High-quality JPEG re-encoded with MozJPEG |
| `test_jpeg_lossless_only` | Low-quality JPEG → jpegtran only (no generation loss) |
| `test_jpeg_progressive` | `progressive_jpeg=True` produces progressive output |
| `test_webp_pillow_primary` | WebP optimized via Pillow in-process |
| `test_webp_cwebp_fallback` | Poor Pillow result triggers cwebp fallback |
| `test_gif_optimization` | gifsicle reduces GIF size |
| `test_svg_scour` | SVG metadata/comments removed, paths simplified |
| `test_svgz_roundtrip` | SVGZ → gunzip → scour → gzip → valid SVGZ |
| `test_avif_no_reencode` | AVIF output has same pixel data (no generation loss) |
| `test_heic_no_reencode` | HEIC output has same pixel data |
| `test_passthrough_tiff` | TIFF optimized or returned unchanged |
| `test_optimization_guarantee` | No optimizer returns output larger than input |
| `test_metadata_strip_jpeg` | GPS stripped, orientation + ICC preserved |
| `test_metadata_strip_png` | tEXt stripped, iCCP + pHYs preserved |
| `test_subprocess_timeout` | Hanging process killed after timeout |
| `test_subprocess_piping` | stdin/stdout piping works for all CLI tools |
