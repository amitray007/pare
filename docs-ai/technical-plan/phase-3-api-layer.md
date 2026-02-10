# Phase 3 — API Layer

## Objectives

- Implement the three FastAPI endpoints: `POST /optimize`, `POST /estimate`, `GET /health`
- Handle both input modes: multipart file upload and JSON body with URL
- Format response based on presence of `storage` key (raw bytes vs JSON)
- Build the estimation engine (header analysis, heuristics, thumbnail compression)
- Implement concurrency control (semaphore + backpressure queue)

## Deliverables

- `routers/optimize.py` — POST /optimize endpoint
- `routers/estimate.py` — POST /estimate endpoint
- `routers/health.py` — GET /health endpoint
- `utils/concurrency.py` — semaphore and backpressure management
- `estimation/estimator.py` — main estimation dispatch
- `estimation/header_analysis.py` — fast header-only reading
- `estimation/heuristics.py` — format-specific prediction rules

## Dependencies

- Phase 2 (optimization engine, format detection, subprocess runner)

---

## Files to Create

### 1. `routers/optimize.py`

**Purpose:** POST /optimize endpoint. Handles both file upload and URL-based input, dispatches to the optimizer, and formats the response.

**Request parsing flowchart:**

```
Incoming POST /optimize
    │
    ├── Content-Type: multipart/form-data?
    │   ├── YES → Extract file from "file" field
    │   │         Parse "options" form field as JSON → OptimizationConfig + StorageConfig
    │   │         Read file bytes into memory
    │   └── NO → continue
    │
    ├── Content-Type: application/json?
    │   ├── YES → Parse JSON body → OptimizeRequest model
    │   │         Fetch image from request.url via url_fetch (Phase 5)
    │   └── NO → 400 Bad Request
    │
    ├── Validate file size (≤ 32 MB)
    │   └── FAIL → 413 FileTooLargeError
    │
    ├── Validate file format (magic bytes)
    │   └── FAIL → 415 UnsupportedFormatError
    │
    ├── Acquire semaphore slot (with backpressure)
    │   └── FAIL → 503 BackpressureError with Retry-After header
    │
    ├── Dispatch to optimizers/router.optimize_image(data, config)
    │
    ├── Has storage config?
    │   ├── YES → Upload to GCS (Phase 5) → return JSON OptimizeResponse
    │   └── NO → return raw bytes with X-* headers
    │
    └── Release semaphore slot
```

**Key function:**

```python
from fastapi import APIRouter, File, Form, UploadFile, Request
from fastapi.responses import Response, JSONResponse

router = APIRouter()


@router.post("/optimize")
async def optimize(
    request: Request,
    file: UploadFile | None = File(None),
    options: str | None = Form(None),
):
    """Optimize an image.

    Two input modes:
    1. Multipart: file field + optional options JSON string
    2. JSON body: url field + inline optimization/storage config

    Two response modes:
    1. No storage → raw bytes with X-* headers
    2. Storage present → JSON with storage URL + stats
    """
```

**Binary response construction:**

```python
def _build_binary_response(result: OptimizeResult, request_id: str) -> Response:
    """Build raw bytes response with X-* headers."""
    mime_types = {
        "png": "image/png",
        "apng": "image/apng",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
        "svg": "image/svg+xml",
        "svgz": "image/svg+xml",
        "avif": "image/avif",
        "heic": "image/heic",
        "tiff": "image/tiff",
        "bmp": "image/bmp",
        "psd": "image/vnd.adobe.photoshop",
    }
    return Response(
        content=result.optimized_bytes,
        media_type=mime_types.get(result.format, "application/octet-stream"),
        headers={
            "X-Original-Size": str(result.original_size),
            "X-Optimized-Size": str(result.optimized_size),
            "X-Reduction-Percent": str(result.reduction_percent),
            "X-Original-Format": result.format,
            "X-Optimization-Method": result.method,
            "X-Request-ID": request_id,
        },
    )
```

**Multipart options parsing:**

The `options` form field is a JSON string. It must be parsed and validated against the Pydantic models:

```python
import json
from schemas import OptimizationConfig, StorageConfig

def _parse_form_options(options_str: str | None) -> tuple[OptimizationConfig, StorageConfig | None]:
    """Parse the 'options' form field JSON string.

    Returns:
        (optimization_config, storage_config_or_none)

    Raises:
        400 if JSON is malformed or validation fails.
    """
    if not options_str:
        return OptimizationConfig(), None

    try:
        data = json.loads(options_str)
    except json.JSONDecodeError:
        raise PareError(...)  # 400

    opt_config = OptimizationConfig(**(data.get("optimization", {})))
    storage_config = None
    if "storage" in data:
        storage_config = StorageConfig(**data["storage"])

    return opt_config, storage_config
```

---

### 2. `routers/estimate.py`

**Purpose:** POST /estimate endpoint. Predicts compression savings without performing full compression.

**Request flow:**

```
Incoming POST /estimate
    │
    ├── Parse input (same two modes as /optimize)
    │
    ├── Validate file size + format (same as /optimize)
    │
    ├── NO semaphore needed (estimation is lightweight)
    │
    ├── Dispatch to estimation/estimator.estimate(data)
    │
    └── Return JSON EstimateResponse
```

```python
router = APIRouter()


@router.post("/estimate")
async def estimate(
    request: Request,
    file: UploadFile | None = File(None),
    options: str | None = Form(None),
):
    """Estimate compression savings without compressing.

    Returns predicted size, reduction percent, confidence level,
    and the method that would be used.

    Estimation is lightweight (~20-50ms) and does not require
    a semaphore slot.
    """
```

---

### 3. `routers/health.py`

**Purpose:** GET /health endpoint. Returns service status and tool availability.

```python
import shutil
import asyncio

router = APIRouter()


@router.get("/health")
async def health():
    """Check service health and tool availability.

    Verifies each CLI tool is present on PATH and each
    Python library imports successfully.

    Response example:
    {
        "status": "ok",
        "tools": {
            "pngquant": true,
            "cjpeg": true,
            "jpegtran": true,
            "gifsicle": true,
            "cwebp": true,
            "pillow": true,
            "pyoxipng": true,
            "scour": true,
            "pillow_heif": true
        },
        "version": "0.1.0"
    }
    """
    tools = {}

    # CLI tools — check if binary exists on PATH
    for tool in ["pngquant", "cjpeg", "jpegtran", "gifsicle", "cwebp"]:
        tools[tool] = shutil.which(tool) is not None

    # Python libraries — check if importable
    for lib in ["PIL", "pyoxipng", "scour", "pillow_heif"]:
        try:
            __import__(lib)
            tools[lib.lower().replace("pil", "pillow")] = True
        except ImportError:
            tools[lib.lower().replace("pil", "pillow")] = False

    status = "ok" if all(tools.values()) else "degraded"

    return {
        "status": status,
        "tools": tools,
        "version": "0.1.0",
    }
```

---

### 4. `utils/concurrency.py`

**Purpose:** Semaphore for concurrent compression jobs + backpressure queue to prevent OOM.

**Concurrency model (PRD Section 5.2):**

```
HTTP Server (async, handles 1000s of connections)
        │
   Semaphore(N)  ← N = CPU core count (configurable)
        │
   Queue depth limit = 2 * N
        │
   ┌────┴────┐
   ▼         ▼
Library    Async Subprocess
```

```python
import asyncio
from config import settings
from exceptions import BackpressureError


class CompressionGate:
    """Controls concurrent compression jobs with backpressure.

    - Semaphore limits active jobs to CPU count
    - Queue depth limit prevents OOM from queued payloads (up to 32MB each)
    - When queue is full, returns 503 immediately
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(settings.compression_semaphore_size)
        self._queue_depth = 0
        self._max_queue = settings.max_queue_depth
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a compression slot.

        Raises BackpressureError (503) if queue is full.
        """
        async with self._lock:
            if self._queue_depth >= self._max_queue:
                raise BackpressureError(
                    "Compression queue full. Try again shortly.",
                    retry_after=5,
                )
            self._queue_depth += 1

        await self._semaphore.acquire()

    def release(self):
        """Release a compression slot."""
        self._semaphore.release()
        # Decrement queue depth (no lock needed — atomic on CPython)
        self._queue_depth -= 1

    @property
    def active_jobs(self) -> int:
        """Number of currently active compression jobs."""
        return settings.compression_semaphore_size - self._semaphore._value

    @property
    def queued_jobs(self) -> int:
        """Number of jobs waiting for a semaphore slot."""
        return self._queue_depth - self.active_jobs


# Module-level singleton
compression_gate = CompressionGate()
```

**Usage in router:**

```python
from utils.concurrency import compression_gate

async def optimize(...):
    await compression_gate.acquire()
    try:
        result = await optimize_image(data, config)
    finally:
        compression_gate.release()
```

---

### 5. `estimation/estimator.py`

**Purpose:** Main estimation dispatch. Coordinates the three estimation layers.

**Three-layer estimation (PRD Section 4.1):**

```
Layer 1: Header Analysis (~1ms)
    → format, dimensions, bit depth, color type, optimization markers
    │
Layer 2: Format-Specific Heuristics (~1ms)
    → predicted reduction based on format signals
    │
Layer 3: Thumbnail Compression (~15-30ms) — JPEG/WebP only
    → resize to 64x64, compress with actual tool, extrapolate
    │
Combine layers → EstimateResponse
```

```python
from utils.format_detect import detect_format, ImageFormat
from estimation.header_analysis import analyze_header
from estimation.heuristics import predict_reduction
from schemas import EstimateResponse


async def estimate(data: bytes) -> EstimateResponse:
    """Estimate compression savings without full compression.

    Strategy:
    1. Header analysis (all formats)
    2. Format-specific heuristics (all formats)
    3. Thumbnail compression (JPEG/WebP only — other formats skip)
    4. Combine signals into a prediction with confidence level

    Target latency: ~20-50ms.
    """
    fmt = detect_format(data)
    header_info = analyze_header(data, fmt)
    prediction = predict_reduction(header_info, fmt)

    # Layer 3: thumbnail compression for JPEG/WebP only
    if fmt in (ImageFormat.JPEG, ImageFormat.WEBP):
        thumbnail_ratio = await _thumbnail_compress(data, fmt)
        prediction = _combine_with_thumbnail(prediction, thumbnail_ratio)

    return EstimateResponse(
        original_size=len(data),
        original_format=fmt.value,
        dimensions=header_info.dimensions,
        color_type=header_info.color_type,
        bit_depth=header_info.bit_depth,
        estimated_optimized_size=prediction.estimated_size,
        estimated_reduction_percent=prediction.reduction_percent,
        optimization_potential=prediction.potential,  # "high" / "medium" / "low"
        method=prediction.method,
        already_optimized=prediction.already_optimized,
        confidence=prediction.confidence,  # "high" / "medium" / "low"
    )


async def _thumbnail_compress(data: bytes, fmt: ImageFormat) -> float:
    """Resize to 64x64, compress with actual tool, return ratio.

    Only used for JPEG and WebP where thumbnail compression
    ratio scales roughly linearly with resolution.
    """


def _combine_with_thumbnail(prediction, thumbnail_ratio: float):
    """Adjust prediction using thumbnail compression ratio.

    If thumbnail and heuristics agree → high confidence.
    If they diverge → use thumbnail ratio, medium confidence.
    """
```

---

### 6. `estimation/header_analysis.py`

**Purpose:** Fast header-only image reading. Extracts metadata without decoding full pixels.

```python
from dataclasses import dataclass
from typing import Optional
from utils.format_detect import ImageFormat


@dataclass
class HeaderInfo:
    """Parsed image header information."""
    format: ImageFormat
    dimensions: dict  # {"width": int, "height": int}
    color_type: Optional[str]  # "rgb", "rgba", "palette", "grayscale"
    bit_depth: Optional[int]
    has_icc_profile: bool
    has_exif: bool
    estimated_quality: Optional[int]  # JPEG only
    is_progressive: bool  # JPEG only
    is_palette_mode: bool  # PNG only
    color_count: Optional[int]  # PNG palette mode only
    has_metadata_chunks: bool  # PNG text chunks, SVG comments
    frame_count: int  # 1 for static, >1 for animated
    file_size: int


def analyze_header(data: bytes, fmt: ImageFormat) -> HeaderInfo:
    """Extract header information without full image decode.

    For most formats, reads only the first few hundred bytes.
    Uses Pillow in lazy mode (does not load pixel data).

    Format-specific parsing:
    - PNG: Read IHDR chunk for dimensions/color type, check palette, count chunks
    - JPEG: Read SOF marker for dimensions, parse quantization tables for quality
    - WebP: Read VP8/VP8L header for dimensions
    - GIF: Read logical screen descriptor + count frames
    - SVG: Parse XML root attributes for viewBox/width/height
    """
```

---

### 7. `estimation/heuristics.py`

**Purpose:** Format-specific prediction rules based on header analysis signals.

**Heuristic rules (PRD Section 4.1, Layer 2):**

```python
from dataclasses import dataclass
from estimation.header_analysis import HeaderInfo
from utils.format_detect import ImageFormat


@dataclass
class Prediction:
    """Estimation prediction result."""
    estimated_size: int
    reduction_percent: float
    potential: str  # "high", "medium", "low"
    method: str
    already_optimized: bool
    confidence: str  # "high", "medium", "low"


def predict_reduction(info: HeaderInfo, fmt: ImageFormat) -> Prediction:
    """Predict compression reduction based on format-specific heuristics.

    Rules by format:

    PNG:
        - Color count < 256 → pngquant very effective (~70% reduction) → "high" potential
        - Already palette-mode → limited room (~5-10%) → "low" potential
        - Has large tEXt/iCCP chunks → metadata removal helps → bump estimate
        - Confidence: "medium" (heuristics only, no thumbnail)

    JPEG:
        - Estimated quality > 90 → big savings with MozJPEG (~50-65%) → "high" potential
        - Estimated quality < 70 → minimal savings (~5-15%) → "low" potential
        - Already progressive → slightly less room
        - Confidence: "medium" (thumbnail will upgrade to "high")

    WebP:
        - Large file + high quality → moderate savings (~40%) → "medium" potential
        - Confidence: "medium" (thumbnail will upgrade to "high")

    SVG:
        - Check for: comments, metadata, editor attributes, whitespace
        - Has comments/metadata/editors → good scour savings (~40-60%) → "high"
        - Already minified → minimal (~5%) → "low"
        - Confidence: "medium"

    GIF:
        - Many frames + unoptimized → gifsicle effective (~20-30%) → "medium"
        - Already optimized → minimal → "low"
        - Confidence: "medium"

    AVIF/HEIC:
        - Metadata-only optimization → estimate based on metadata/file size ratio
        - Confidence: "low"

    TIFF/BMP/PSD:
        - Limited optimization potential
        - Confidence: "low"
    """
```

---

## Request/Response Examples

### File upload → binary response

```
POST /optimize
Content-Type: multipart/form-data; boundary=---
file=@product.png

→ 200 OK
Content-Type: image/png
Content-Length: 62154
X-Original-Size: 235169
X-Optimized-Size: 62154
X-Reduction-Percent: 73.6
X-Original-Format: png
X-Optimization-Method: pngquant + oxipng
X-Request-ID: 550e8400-e29b-41d4-a716-446655440000
<binary bytes>
```

### File upload with storage → JSON response

```
POST /optimize
Content-Type: multipart/form-data
file=@product.png
options={"storage": {"provider": "gcs", "bucket": "my-bucket", "path": "opt/product.png"}}

→ 200 OK
Content-Type: application/json
{
    "success": true,
    "original_size": 235169,
    "optimized_size": 62154,
    "reduction_percent": 73.6,
    "format": "png",
    "method": "pngquant + oxipng",
    "storage": {
        "provider": "gcs",
        "url": "gs://my-bucket/opt/product.png",
        "public_url": null
    }
}
```

### No reduction (already optimized)

```
POST /optimize
file=@already-optimized.png

→ 200 OK
X-Original-Size: 235169
X-Optimized-Size: 235169
X-Reduction-Percent: 0.0
X-Optimization-Method: none
<original bytes>
```

### Backpressure (queue full)

```
POST /optimize
file=@huge.png

→ 503 Service Unavailable
Retry-After: 5
{
    "success": false,
    "error": "service_overloaded",
    "message": "Compression queue full. Try again shortly."
}
```

---

## Environment Variables Introduced

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPRESSION_SEMAPHORE_SIZE` | `CPU_COUNT` | Max concurrent compression jobs |
| `MAX_QUEUE_DEPTH` | `2 * CPU_COUNT` | Backpressure queue limit |

---

## Verification Steps

### Manual verification

```bash
# Test file upload (binary response)
curl -X POST http://localhost:8080/optimize \
  -F "file=@sample.png" \
  -o optimized.png -D -
# Check X-* headers in output, compare file sizes

# Test file upload with custom quality
curl -X POST http://localhost:8080/optimize \
  -F "file=@sample.jpg" \
  -F 'options={"optimization": {"quality": 60, "progressive_jpeg": true}}' \
  -o optimized.jpg -D -

# Test estimation
curl -X POST http://localhost:8080/estimate \
  -F "file=@sample.png" | jq .

# Test health
curl http://localhost:8080/health | jq .

# Test backpressure (send many concurrent requests)
for i in $(seq 1 50); do
  curl -X POST http://localhost:8080/optimize \
    -F "file=@large-image.png" -o /dev/null -s -w "%{http_code}\n" &
done
wait
# Some should return 503 when queue fills up
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_optimize_file_upload` | File upload returns optimized bytes with correct headers |
| `test_optimize_json_url` | JSON body with URL fetches and optimizes (Phase 5 mock) |
| `test_optimize_with_options` | Custom quality/progressive flags applied correctly |
| `test_optimize_with_storage` | Storage config triggers JSON response (Phase 5 mock) |
| `test_optimize_file_too_large` | >32MB file returns 413 |
| `test_optimize_unsupported_format` | Unknown magic bytes return 415 |
| `test_optimize_malformed_options` | Invalid JSON in options field returns 400 |
| `test_optimize_no_reduction` | Already-optimized image returns 200 with 0% reduction |
| `test_estimate_basic` | Estimate returns predicted savings with confidence |
| `test_estimate_jpeg_high_quality` | High-quality JPEG estimates high savings |
| `test_estimate_png_palette` | Palette PNG estimates moderate savings |
| `test_health_all_tools` | Health endpoint reports all tools available |
| `test_health_missing_tool` | Missing tool reports "degraded" status |
| `test_backpressure_503` | Queue-full scenario returns 503 with Retry-After |
| `test_semaphore_limits_concurrency` | Only N concurrent compressions run |
| `test_request_id_in_response` | X-Request-ID present in all responses |
