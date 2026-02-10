# Image Optimizer Service — Implementation Plan

## 1. Background & Problem Statement

We are building an **Image Optimizer** that will serve two products:

1. **Shopify App** — Optimizes product images, collection images, and blog/article images for Shopify stores
2. **Marketing Website Free Tool** — A public image optimization tool to drive traffic and demonstrate value

Shopify accepts: **PNG (recommended), JPEG, PSD, TIFF, BMP, GIF, SVG, HEIC, and WebP.**

We need a single, unified API — one source of truth — that every consumer calls without knowing or caring about format-specific internals.

---

## 2. Research & Testing Summary

### 2.1 APIs Tested

We evaluated 4 existing open-source image processing APIs:

| API | Type | URL |
|-----|------|-----|
| Imagor (MozJPEG) | URL-based GET | `https://cshum-imagor-mozjpeg.sliplane.app` |
| Imagor (Standard) | URL-based GET | `https://shumc-imagor.sliplane.app` |
| Imaginary | Multipart POST | `https://h2non-imaginary.sliplane.app` |
| imgproxy | URL-based GET | `https://imgproxy-imgproxy.sliplane.app` |

### 2.2 API Compression Results (Synthetic Test Image)

| Format | Imagor MozJPEG | Imagor Standard | Imaginary | imgproxy |
|--------|------:|------:|------:|------:|
| JPEG | **34.8%** | 42.6% | 44.3% | 44.3% |
| PNG | 18.8% | 18.8% | 8.7% | **8.5%** |
| GIF | 75.1% | 75.1% | -- | 75.1% |
| WebP | **56.3%** | 56.3% | 55.0% | 56.3% |
| AVIF | **81.3%** | 81.3% | -- | Broken (130%) |
| HEIC | FAIL | FAIL | -- | FAIL |
| SVG | -- | -- | -- | 99.0% |

### 2.3 Real-World PNG Test (2.3MB Wallpaper Image)

This test exposed a critical finding — APIs are **image processing servers**, not optimizers:

| Tool | Output | Ratio | Verdict |
|------|------:|------:|---------|
| **pngquant (dedicated)** | **621,540** | **26.4%** | Excellent |
| **oxipng (dedicated)** | 2,186,980 | 93.0% | Good (lossless) |
| Imaginary | 2,318,776 | 98.6% | Barely compressed |
| imgproxy | 2,391,816 | 101.7% | Made it bigger |
| Imagor MozJPEG | 3,744,670 | 159.2% | Made it much bigger |

### 2.4 Key Findings

1. **No single API optimizes all formats well.** Imagor MozJPEG is great for JPEG but terrible for PNG. imgproxy is decent for PNG synthetic images but fails on real-world PNGs. Imaginary is abandoned.
2. **Dedicated compression tools massively outperform APIs.** pngquant achieves 73% reduction on real PNGs vs 0-1.4% by the APIs. MozJPEG achieves 65% reduction on JPEG vs ~55% by standard encoders.
3. **The APIs are built for resizing/cropping/watermarking**, not compression. Compression is a side effect.
4. **Imaginary is abandoned** — no new releases, limited format support (JPEG/PNG/WebP only). Not suitable for production.
5. **imgproxy converts formats by default** — needed `IMGPROXY_PREFERRED_FORMATS` config or explicit `f:{format}` per request. Its AVIF support is broken (increases file size).
6. **HEIC fails on all APIs** — output is valid HEIF bytes but Pillow cannot re-open them (codec compatibility issue).

### 2.5 Decision

**Build a custom Image Optimizer API** — not a compression engine, but an **orchestrator** that routes each format to the best-in-class dedicated tool.

---

## 3. Architecture

### 3.1 High-Level Design

```
Clients (Shopify App, Marketing Website, Future Tools)
                    │
                    ▼
            Load Balancer
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Worker  │ │ Worker  │ │ Worker  │   (auto-scaled containers)
  │ Container│ │ Container│ │ Container│
  │         │ │         │ │         │
  │ API     │ │ API     │ │ API     │
  │ Layer   │ │ Layer   │ │ Layer   │
  │    │    │ │    │    │ │    │    │
  │ ┌──┴──┐ │ │ ┌──┴──┐ │ │ ┌──┴──┐ │
  │ │Tools│ │ │ │Tools│ │ │ │Tools│ │   (all binaries baked into image)
  │ └─────┘ │ │ └─────┘ │ │ └─────┘ │
  └─────────┘ └─────────┘ └─────────┘
```

**Single Docker image. Single endpoint. All compression tools baked in. Scale by adding containers.**

### 3.2 Format-to-Tool Routing

| Format | Tool | Type | Expected Compression |
|--------|------|------|---------------------|
| PNG | pngquant + oxipng | CLI + Library | ~70-75% reduction |
| JPEG | MozJPEG (cjpeg) | CLI / Pillow | ~50-65% reduction |
| WebP | cwebp / Pillow | CLI / Library | ~40-55% reduction |
| GIF | gifsicle | CLI | ~20-25% reduction |
| SVG | scour (Python lib) | Library | ~40-60% reduction |
| SVGZ | scour + gzip | Library | ~40-60% reduction |
| AVIF | pillow-heif / cavif | Library / CLI | ~15-20% reduction |
| HEIC | pillow-heif | Library | ~15-20% reduction |
| TIFF | Pillow (convert → optimize) | Library | Format-dependent |
| BMP | Pillow (convert → optimize) | Library | Format-dependent |
| PSD | Pillow (convert → optimize) | Library | Format-dependent |

**Strategy:** Use in-process library bindings where available (faster, no subprocess overhead). Fall back to async subprocess for CLI-only tools.

### 3.3 API Endpoints

```
POST /optimize
    - Accepts: multipart file upload OR JSON with image URL
    - All configuration via structured request body (no query params)
    - Returns: optimized image raw bytes (default) OR JSON with storage URL (when storage configured)
    - Note: No download URL mode — Cloud Run is stateless with multiple instances,
      so a download URL generated on one instance won't be accessible on another
    - Use case: Marketing website free tool, Shopify single image

POST /estimate
    - Accepts: multipart file upload OR JSON with image URL
    - Returns: JSON with estimated savings (no actual compression)
    - Use case: Preview before optimization, Shopify dashboard stats

GET /health
    - Returns: service health + available tools
```

### 3.4 Request Schema

All configuration is passed in a structured request body — **no query params**. This keeps the API
clean, extensible, and easy to validate with Pydantic models.

**Two input modes, same schema:**

| Mode | How | Options field |
|------|-----|---------------|
| File upload | `multipart/form-data` with `file` field | `options` form field (JSON string) |
| URL-based | `application/json` body | Inline at root level |

**Full options schema (all fields optional except where noted):**

```jsonc
{
    // --- Input (URL mode only, omit for file upload) ---
    "url": "https://cdn.shopify.com/s/files/.../product.png",

    // --- Optimization ---
    "optimization": {
        "quality": 80,                  // 1-100, default 80
        "strip_metadata": true,         // Strip non-essential EXIF/XMP (default true)
        "progressive_jpeg": false,      // Opt-in progressive JPEG (default false)
        "png_lossy": true               // Allow pngquant lossy. false = lossless oxipng only (default true)
    },

    // --- Storage (optional — triggers JSON response) ---
    "storage": {
        "provider": "gcs",             // Storage provider. Currently: "gcs". Future: "s3", "azure"
        "bucket": "my-bucket",         // Required: bucket/container name
        "path": "optimized/product.png", // Required: object path within bucket
        "project": "my-gcp-project",   // Optional: GCP project ID (uses default if omitted)
        "public": false                // Optional: make object publicly accessible (default false)
    }
}
```

**What determines the response format:**
- **No `storage` key** → returns raw optimized bytes with `X-*` headers
- **`storage` key present** → returns JSON with storage URL + optimization stats

### 3.5 Request/Response Examples

**1. Simple file upload (raw bytes response):**
```http
POST /optimize
Content-Type: multipart/form-data

file=@product.png

Response:
Content-Type: image/png
Content-Length: 62154
X-Original-Size: 235169
X-Optimized-Size: 62154
X-Reduction-Percent: 73.6

<optimized image bytes>
```

**2. File upload with custom quality:**
```http
POST /optimize
Content-Type: multipart/form-data

file=@product.png
options={"optimization": {"quality": 60, "progressive_jpeg": true}}

Response:
<optimized image bytes with X-* headers>
```

**3. URL-based input:**
```http
POST /optimize
Content-Type: application/json

{
    "url": "https://cdn.shopify.com/s/files/.../product.png",
    "optimization": {
        "quality": 80
    }
}

Response:
<optimized image bytes with X-* headers>
```

**4. File upload with GCS storage:**
```http
POST /optimize
Content-Type: multipart/form-data

file=@product.png
options={"optimization": {"quality": 80}, "storage": {"provider": "gcs", "bucket": "my-bucket", "path": "optimized/product.png"}}

Response:
{
    "success": true,
    "original_size": 235169,
    "optimized_size": 62154,
    "reduction_percent": 73.6,
    "format": "png",
    "method": "pngquant + oxipng",
    "storage": {
        "provider": "gcs",
        "url": "gs://my-bucket/optimized/product.png",
        "public_url": "https://storage.googleapis.com/my-bucket/optimized/product.png"
    }
}
```

**5. URL-based input with GCS storage:**
```http
POST /optimize
Content-Type: application/json

{
    "url": "https://cdn.shopify.com/s/files/.../product.png",
    "optimization": {"quality": 75},
    "storage": {
        "provider": "gcs",
        "bucket": "shopify-images",
        "path": "stores/123/optimized/product.png",
        "project": "my-gcp-project",
        "public": true
    }
}
```

**6. Estimate:**
```http
POST /estimate
Content-Type: multipart/form-data

file=@product.png

Response:
{
    "original_size": 2351691,
    "original_format": "png",
    "dimensions": {"width": 1920, "height": 1080},
    "color_type": "rgba",
    "bit_depth": 8,
    "estimated_optimized_size": 615000,
    "estimated_reduction_percent": 73.8,
    "optimization_potential": "high",
    "method": "pngquant + oxipng",
    "already_optimized": false
}
```

**7. Error response (standard across all endpoints):**
```json
{
    "success": false,
    "error": "optimization_failed",
    "message": "Could not optimize image: output is larger than input",
    "original_size": 235169,
    "format": "png"
}
```

**8. No-reduction response (image already optimized):**
```http
POST /optimize
Content-Type: multipart/form-data

file=@already-optimized.png

Response (binary mode):
HTTP/1.1 200 OK
Content-Type: image/png
X-Original-Size: 235169
X-Optimized-Size: 235169
X-Reduction-Percent: 0.0
X-Optimization-Method: none

<original image bytes unchanged>
```

### 3.6 Response Headers & Status Codes

**Response headers (on binary responses):**

All binary responses (when no `storage` key is present) include these headers:

| Header | Description |
|--------|-------------|
| `Content-Type` | MIME type of the optimized image |
| `Content-Length` | Size of the optimized image in bytes |
| `X-Original-Size` | Original file size in bytes |
| `X-Optimized-Size` | Optimized file size in bytes |
| `X-Reduction-Percent` | Percentage reduction (e.g., 73.6) |
| `X-Original-Format` | Detected input format |
| `X-Optimization-Method` | Tool used (e.g., "pngquant + oxipng") |
| `X-Request-ID` | Unique request ID for debugging |

**HTTP status codes:**

| Status | When |
|--------|------|
| `200 OK` | Optimization succeeded (including 0% reduction — original returned) |
| `400 Bad Request` | Malformed request body, invalid JSON in `options` field, missing required fields |
| `401 Unauthorized` | Missing or invalid `Authorization` header (when auth is required) |
| `413 Payload Too Large` | File exceeds 32 MB limit |
| `415 Unsupported Media Type` | Unrecognized file format (magic bytes don't match any supported type) |
| `422 Unprocessable Entity` | Optimization produced a larger file, URL fetch failed, corrupt/invalid image |
| `429 Too Many Requests` | Rate limit exceeded (public API) |
| `500 Internal Server Error` | Unexpected server failure (tool crash, OOM, etc.) |
| `503 Service Unavailable` | Compression queue full (backpressure), or service shutting down |

### 3.7 Security & File Validation

**File Size Limit:** 32 MB max. Reject with `413 Payload Too Large` before processing. Aligns with Cloud Run's default request body limit. If larger files are needed in the future, the client uploads to GCS first and passes the URL.

**File Type Validation:**
- Detect format via **magic bytes** (file header), not file extension or Content-Type header
- Supported magic signatures: PNG (`\x89PNG`), JPEG (`\xFF\xD8\xFF`), GIF (`GIF87a`/`GIF89a`), WebP (`RIFF....WEBP`), AVIF/HEIC (ftyp box), SVG (`<?xml`/`<svg`), TIFF (`II*\x00`/`MM\x00*`), BMP (`BM`), PSD (`8BPS`)
- Reject unrecognized formats with `415 Unsupported Media Type`

**SVG Sanitization:**
- Strip all `<script>` tags and event attributes (`onload`, `onclick`, etc.)
- Prevent XXE (XML External Entity) attacks — disable external entity resolution in XML parser via `defusedxml`
- Strip embedded `<foreignObject>` elements
- Strip `data:` URIs in href attributes, block `<use>` with external references

**SSRF Protection (URL-based input):**

When the service fetches a user-supplied URL, an attacker could submit internal network addresses to probe infrastructure (e.g., `http://169.254.169.254/latest/meta-data/` — the Cloud Run metadata server, or `http://localhost:8080/health`). This is called Server-Side Request Forgery (SSRF).

Mitigations:
- **HTTPS only** — reject `http://` URLs
- **Block private/reserved IP ranges** — resolve hostname before fetching, reject if IP is in `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `169.254.0.0/16` (link-local/metadata), `127.0.0.0/8` (localhost), `::1`, `fc00::/7`
- **Block Cloud metadata endpoints** — explicitly reject `metadata.google.internal` and `169.254.169.254`
- **Limit redirect hops to 5** — prevents redirect chains that bounce into internal networks
- **Abort downloads exceeding 32 MB** — stream + count bytes, abort early

**CORS Configuration:**

The marketing website free tool makes browser requests, which trigger CORS preflight. FastAPI CORS middleware configured with:
- Allowed origins: configurable via env var (marketing site domain, localhost for dev)
- Allowed methods: `POST`, `OPTIONS`
- Allowed headers: `Content-Type`, `Authorization`
- Expose headers: `X-Original-Size`, `X-Optimized-Size`, `X-Reduction-Percent`, `X-Original-Format`, `X-Optimization-Method`

### 3.8 Authentication & Rate Limiting

**Rate Limiting — Redis-backed, configurable via environment variables:**

Rate limiting uses **Redis via VPC** for shared state across all Cloud Run instances. This ensures consistent limits regardless of which instance handles the request.

| Env Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | (required) | Redis connection URL (VPC-internal) |
| `RATE_LIMIT_PUBLIC_RPM` | `60` | Requests per minute for unauthenticated requests |
| `RATE_LIMIT_PUBLIC_BURST` | `10` | Max burst for unauthenticated requests |
| `RATE_LIMIT_AUTH_ENABLED` | `false` | Whether to rate limit authenticated requests |
| `RATE_LIMIT_AUTH_RPM` | `0` (unlimited) | Requests per minute for authenticated requests |

**Authentication:**
- Authenticated requests pass an API key via `Authorization: Bearer <key>` header
- Authenticated requests bypass rate limiting by default (unless `RATE_LIMIT_AUTH_ENABLED=true`)
- **API key storage:** Single valid key stored in **GCP Secret Manager**, mounted as an environment variable at deploy time. One key for now, used by the Shopify App backend for internal API calls
- The backend (Shopify App) is responsible for determining whether an image has already been optimized — the optimizer service does not track this

### 3.9 Metadata Handling

**Algorithm for strip vs. preserve:**

| Metadata Type | Action | Reason |
|---|---|---|
| EXIF Orientation | **Preserve** | Critical — removing it causes rotated images in browsers |
| ICC Color Profile | **Preserve** | Removing it degrades colors, especially for product photography |
| XMP / IPTC | Strip | Editorial metadata, not needed for web display |
| GPS / Location | **Strip** | Privacy concern — should never be exposed on web |
| Camera/Device info | Strip | Not needed for web display |
| Thumbnail | Strip | Embedded thumbnails waste space |
| Comments | Strip | Editor metadata, not needed |

**Implementation:** Use Pillow's EXIF handling to selectively strip. For JPEG, pipe through `jpegtran` or MozJPEG with explicit flags. For PNG, strip text chunks but preserve `iCCP` and `pHYs`. Controlled by the `optimization.strip_metadata` flag (default `true`) — see Section 3.4.

### 3.10 Quality Parameter Mapping

The `optimization.quality` param (1-100, default 80) is mapped internally per format:

| Format | Internal Mapping | Notes |
|---|---|---|
| PNG | pngquant `--quality {q-15}-{q}` | Range-based quality floor/ceiling |
| JPEG | MozJPEG `-quality {q}` | Direct mapping |
| WebP | cwebp `-q {q}` or Pillow `quality={q}` | Direct mapping |
| GIF | gifsicle `--optimize=3` | Quality param ignored — gifsicle is lossless optimization |
| SVG | scour settings | Quality param ignored — scour optimizes structure |
| AVIF | pillow-heif `quality={q}` | Direct mapping |
| HEIC | pillow-heif `quality={q}` | Direct mapping |

All per-format options (`progressive_jpeg`, `png_lossy`, `strip_metadata`) are part of the
`optimization` object in the request schema — see Section 3.4.

### 3.11 Animated Image Support

Animated images are supported. The optimizer detects animation automatically and preserves all frames — no special parameter needed.

| Format | Tool | Animated Support |
|---|---|---|
| GIF | gifsicle | **Yes** — optimizes frame disposal, LZW per frame |
| WebP | cwebp / Pillow | **Yes** — Pillow preserves animation frames |
| APNG | Detect + skip lossy | **Yes** — detect via `acTL` chunk; apply lossless oxipng only, skip pngquant (which would destroy frames) |

**APNG safety:** APNG files contain an `acTL` (animation control) chunk. If detected, the PNG pipeline skips pngquant (lossy, would drop frames) and applies only lossless oxipng. If oxipng doesn't reduce size, return original with 0% reduction.

### 3.12 Storage Integration

Storage upload is triggered by including a `storage` object in the request body (see schema in Section 3.4).

**Currently supported providers:**

| Provider | `storage.provider` | Required Fields | Optional Fields |
|---|---|---|---|
| Google Cloud Storage | `"gcs"` | `bucket`, `path` | `project`, `public` |

**Future providers** (same schema pattern — just change `provider` and add provider-specific fields):

| Provider | `storage.provider` | Additional Fields |
|---|---|---|
| AWS S3 | `"s3"` | `region`, `acl` |
| Azure Blob | `"azure"` | `container`, `account` |

**GCS Authentication:** The service uses a GCP service account. Credentials provided via:
- `GOOGLE_APPLICATION_CREDENTIALS` env var (service account JSON key)
- Or workload identity (automatic on Cloud Run)

**Behavior:**
1. Image is optimized normally
2. If `storage` is present, upload optimized bytes to the specified provider
3. Return JSON response with storage URLs + optimization stats
4. If storage upload fails, return error — the client needs the URL, not partial results

**Response format is determined by the `storage` key:**
- **No `storage` key** → raw optimized bytes with `X-*` headers (Section 3.6)
- **`storage` key present** → JSON with storage URL + stats (Section 3.5, example 4)

Cloud Run is stateless with multiple instances, so download URLs are not supported. Clients either receive raw bytes directly or get a storage URL to fetch from.

### 3.13 URL Fetching (httpx)

When the client passes a `url` instead of uploading a file, the service fetches the image using `httpx`:

| Setting | Authenticated Requests | Public Requests |
|---|---|---|
| Timeout | 30 seconds | 60 seconds |
| Max download size | 32 MB (stream + count, abort early) | 32 MB |
| Max redirect hops | 5 | 5 |
| Allowed schemes | HTTPS only | HTTPS only |
| SSRF protection | Yes (see Section 3.7) | Yes |

Fetch happens before the optimization pipeline. Bytes are held in memory (no temp files). If the fetch fails or exceeds limits, return `422` with error details.

### 3.14 Optimization Guarantee

Two distinct cases:

**Case 1 — No reduction possible (0% savings):**
The image is already optimized or the format provides no compression opportunity. Return a **special success response** with the original image and 0% reduction:
```json
{
    "success": true,
    "original_size": 235169,
    "optimized_size": 235169,
    "reduction_percent": 0.0,
    "format": "png",
    "method": "none",
    "message": "Image is already optimized"
}
```
For binary responses: return the original bytes with `X-Reduction-Percent: 0.0`. This is not an error — the client gets a usable image either way.

**Case 2 — Optimized output is LARGER than input:**
This means the optimization tool made it worse. Return an **error response**:
```json
{
    "success": false,
    "error": "optimization_failed",
    "message": "Optimization produced a larger file — original returned unchanged would be wasteful. Aborting.",
    "original_size": 235169,
    "attempted_size": 248000,
    "format": "png"
}
```
HTTP status: `422 Unprocessable Entity`. The API never returns a file larger than the input.

---

## 4. Estimation API Design

The estimation endpoint predicts compression results in ~20-50ms without performing full compression.

### 4.1 Technique: Three-Layer Estimation

**Layer 1 — Header Analysis (~1ms)**
- Read image header only (no full decode)
- Extract: format, dimensions, bit depth, color type
- Detect optimization markers (already optimized?)

**Layer 2 — Format-Specific Heuristics (~1ms)**

| Format | Signal | Prediction |
|--------|--------|------------|
| PNG | Color count < 256 | pngquant very effective (~70% reduction) |
| PNG | Already palette-mode | Limited room (~5-10%) |
| PNG | Has large tEXt/iCCP chunks | Metadata removal helps |
| JPEG | High quality tables (q>90) | Big savings with MozJPEG (~50-65%) |
| JPEG | Low quality (q<70) | Minimal savings (~5-15%) |
| JPEG | Progressive already | Slightly less room |
| WebP | High quality, large file | Moderate savings (~40%) |
| SVG | Has comments/metadata/editors | Good SVGO savings (~40-60%) |
| SVG | Already minified | Minimal (~5%) |
| GIF | Many frames, unoptimized | gifsicle effective (~20-30%) |
| GIF | Already optimized | Minimal |

**Layer 3 — Thumbnail Compression (~15-30ms)**
- Resize to 64x64 thumbnail
- Compress thumbnail with the actual target tool
- Extrapolate ratio to full image

**Accuracy limits by format:**

| Format | Thumbnail Reliability | Reason |
|---|---|---|
| JPEG | Good | Compression ratio scales roughly linearly with resolution |
| WebP | Good | Similar to JPEG |
| PNG | Low | pngquant effectiveness depends on color palette complexity at full resolution — a 64x64 thumbnail always has few colors |
| GIF | Skip thumbnail | Frame count and disposal method matter more than pixel content — use heuristics only |
| SVG/SVGZ | Skip thumbnail | Not pixel-based — analyze whitespace, comments, redundant attributes, editor metadata instead |
| AVIF/HEIC | Skip thumbnail | Lossless-only optimization — estimate based on metadata size vs file size ratio |

**Decision:** Use Layer 3 only for JPEG and WebP. For other formats, rely on Layers 1+2 heuristics and return lower confidence.

### 4.2 Confidence Scoring

Return a confidence level with each estimate:

- **high** — thumbnail method + heuristics agree (JPEG, WebP)
- **medium** — heuristics only, well-known format (PNG, GIF)
- **low** — limited signals available (AVIF, HEIC, TIFF/BMP/PSD), or edge case

---

## 5. Concurrency & Performance

### 5.1 Target Performance

- **Peak throughput:** 500+ images/sec (parallel processing across containers)
- **Estimation API:** 1000+ req/sec per container (lightweight, no compression)
- **Optimize API:** ~16 images/sec per container (4 CPU, ~500ms per image)

### 5.2 Concurrency Model

```
HTTP Server (async, handles 1000s of connections)
        │
   Semaphore(N)  ← limits concurrent compression to CPU count
        │
   ┌────┴────┐
   ▼         ▼
Library    Async Subprocess
(in-proc)  (stdin/stdout pipe)
   │         │
pyoxipng   pngquant
scour      gifsicle
Pillow     cjpeg (mozjpeg)
```

- **Semaphore** caps concurrent compression jobs = CPU core count
- **Libraries** (pyoxipng, Pillow, scour) run in-process — no subprocess overhead
- **CLI tools** (pngquant, gifsicle) run via async subprocess with stdin/stdout piping — no temp files
- **Tool timeout:** 60 seconds per tool invocation. If a CLI tool or library hangs, the operation is killed and returns `500`. This is a rare edge case — normal images process in under 5 seconds
- **Estimation requests** bypass the compression semaphore — they're lightweight (header analysis + heuristics). Layer 3 thumbnail compression is optional and fast enough to not need throttling

**Backpressure:**

When all semaphore slots are busy, excess requests queue in memory. Each queued request holds its full image payload (up to 32 MB). To prevent OOM under sustained overload:
- **Max queue depth** = `2 * CPU_COUNT` (configurable via env var `MAX_QUEUE_DEPTH`)
- When queue is full, return `503 Service Unavailable` immediately with `Retry-After` header
- Cloud Run auto-scaling should absorb most bursts before the queue fills

### 5.3 Scaling Math

```
1 container (4 CPU, 2GB RAM):  ~16 images/sec optimize, ~1000/sec estimate
500 images/sec peak:           ~32 containers
RAM budget per container:      ~1.2GB for compression + ~800MB for Python/connections
Cost optimization:             auto-scale down during low traffic
```

Note: Python's per-connection overhead (~1-2MB) means each container needs slightly more RAM than a Go equivalent would. At 32 containers this adds ~750MB total across the cluster — a manageable cost tradeoff for faster development velocity.

### 5.4 Stdin/Stdout Piping (No Temp Files)

```
# Instead of: write file → tool reads file → tool writes file → read file
# Do: pipe bytes in → tool pipes bytes out

pngquant --quality 60-80 - --output - < stdin > stdout
cjpeg -quality 80 < stdin > stdout
gifsicle --optimize=3 < stdin > stdout
```

This eliminates disk I/O entirely. Everything stays in memory.

### 5.5 Logging & Observability

**Structured JSON logging** for Cloud Logging compatibility. Minimal — only errors and critical operational events. No verbose info/debug logs in production.

**What gets logged (ERROR level):**
- Tool crash / timeout (format, tool, image size, duration, error message)
- SSRF attempt blocked (source IP, URL attempted)
- File validation rejection (reason, file size, detected magic bytes)
- Storage upload failure (provider, bucket, path, error)
- Unexpected exceptions (full traceback)

**What gets logged (WARNING level):**
- Optimization produced larger file (format, input size, output size)
- Rate limit hit (source IP, endpoint)
- Tool returned 0% reduction on a format that normally compresses well

**Not logged:** Successful requests, request bodies, image content, auth tokens. Metrics (latency, compression ratio, throughput) are tracked via Cloud Run built-in metrics, not application logs.

**Request ID:** Each request gets a unique ID (UUID) injected via middleware, included in all log entries and response headers (`X-Request-ID`) for debugging.

---

## 6. Technology Decision

### 6.1 Decision: Python (FastAPI)

**Language:** Python 3.12
**Framework:** FastAPI + Uvicorn

### 6.2 Why Python

The compression bottleneck is in C/Rust binaries (pngquant, mozjpeg, oxipng), not the orchestrator. The web server's job is to receive requests, pipe bytes to the right tool, and return results. Python handles this well:

- **98% of CPU time** is spent in compression binaries (C/Rust) — same speed regardless of orchestrator language
- **2% of CPU time** is the web server overhead — Python is fast enough here
- **Faster development** — ship sooner, iterate faster
- **Rich ecosystem** — Pillow, pyoxipng, scour, pillow-heif are all Python-native libraries that run in-process with zero subprocess overhead
- **asyncio + subprocess** — FastAPI's async model handles concurrent requests well; `asyncio.create_subprocess_exec` runs CLI tools without blocking

### 6.3 Go Was Considered But Not Chosen

| Factor | Python (FastAPI) | Go |
|--------|-----------------|-----|
| Memory per connection | ~1-2 MB | ~8 KB |
| 500 concurrent connections | ~750 MB overhead | ~4 MB overhead |
| Subprocess management | asyncio (good) | goroutines (excellent) |
| Deployment | Python runtime + deps | Single static binary |
| Development speed | **Faster** | Slower |
| Image processing ecosystem | **Pillow, pyoxipng, scour** | go-vips, stdlib image |

Go has lower per-connection memory overhead. At 500 images/sec across ~32 containers, this means ~750MB total overhead with Python vs ~4MB with Go. This is a real but manageable cost difference — solved by allocating slightly more RAM per container rather than rewriting in a different language.

**When to reconsider Go:** Only if scaling past 2000+ images/sec sustained, where the memory overhead across 100+ containers becomes a significant cost factor.

### 6.4 Python Dependencies

**Python packages:**
```
fastapi
uvicorn[standard]
Pillow
pillow-heif
pyoxipng
scour
python-multipart
httpx                       # URL-based image fetching (SSRF-safe, async)
google-cloud-storage        # GCS upload integration
defusedxml                  # Safe XML parsing for SVG sanitization
redis[hiredis]              # Rate limiting shared state via VPC Redis
pydantic-settings           # Env-var based configuration
```

**System binaries (baked into Docker image):**
```
pngquant       — lossy PNG quantization
mozjpeg (cjpeg) — optimized JPEG compression
gifsicle       — GIF optimization
cwebp          — WebP encoding (fallback if Pillow WebP insufficient)
```

### 6.5 Language Roles Across the Project

| Component | Language | Why |
|-----------|----------|-----|
| **Image Optimizer Service** (this project) | Python (FastAPI) | Production API, handles all traffic |
| **Shopify App Backend** | Python or Node.js | Shopify integration, OAuth, webhooks — calls the optimizer service |
| **Compression Testing Suite** (existing) | Python | Benchmarking, validation — already built |
| **CI/CD, scripts** | Python / Shell | Automation |

---

## 7. Docker Image

**Multi-stage build** from the start to minimize final image size and cold start time.

```dockerfile
# ---- Stage 1: Build MozJPEG from source ----
FROM debian:bookworm-slim AS mozjpeg-builder
RUN apt-get update && apt-get install -y cmake nasm build-essential
# Build MozJPEG, install to /opt/mozjpeg
# (exact steps TBD — use cmake + make + make install)

# ---- Stage 2: Final image ----
FROM python:3.12-slim

# Copy MozJPEG binaries (cjpeg, jpegtran) from builder
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/cjpeg /usr/local/bin/
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/jpegtran /usr/local/bin/

# Install compression binaries + codec libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    pngquant \
    gifsicle \
    webp \
    libheif-dev \
    libde265-dev \
    libaom-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app

# Use $PORT for Cloud Run compatibility, dynamic worker count
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown 30"]
```

**Key points:**
- Multi-stage build keeps MozJPEG build dependencies out of the final image
- `webp` package provides `cwebp` binary
- `libde265-dev` + `libaom-dev` provide HEIC/AVIF codec support for pillow-heif
- `--timeout-graceful-shutdown 30` — on SIGTERM (Cloud Run shutdown), Uvicorn waits up to 30 seconds for in-flight requests to complete before exiting
- `$PORT` and `$WORKERS` are configurable via env vars (Cloud Run sets `$PORT` automatically)
- GCS credentials via workload identity (automatic on Cloud Run) or `GOOGLE_APPLICATION_CREDENTIALS` env var

**Cold start mitigation:**
- Cloud Run `min-instances=1` to keep at least one warm instance always running
- Keep Docker image as small as possible via multi-stage build and `--no-install-recommends`
- Python dependencies are installed in a single layer for caching

---

## 8. Project Structure

```
image-optimizer-service/
├── Dockerfile
├── requirements.txt
├── main.py                     # FastAPI app, mount routers, startup/shutdown hooks
├── config.py                   # Settings, quality defaults, limits, env vars (Pydantic BaseSettings)
├── schemas.py                  # Pydantic request/response models (OptimizeRequest, OptimizeResponse, EstimateResponse, ErrorResponse, StorageConfig, OptimizationConfig)
├── exceptions.py               # Custom exception classes (OptimizationError, FileTooLargeError, UnsupportedFormatError, SSRFError, ToolTimeoutError)
├── middleware.py                # CORS, auth, rate limiting, request ID injection
├── routers/
│   ├── __init__.py
│   ├── optimize.py             # POST /optimize endpoint logic
│   ├── estimate.py             # POST /estimate endpoint logic
│   └── health.py               # GET /health endpoint
├── optimizers/
│   ├── __init__.py
│   ├── base.py                 # BaseOptimizer abstract class
│   ├── router.py               # Format detection → optimizer dispatch
│   ├── png.py                  # pngquant + oxipng pipeline (with APNG detection)
│   ├── jpeg.py                 # MozJPEG (cjpeg/jpegtran) optimizer
│   ├── webp.py                 # WebP optimizer (Pillow, cwebp fallback)
│   ├── gif.py                  # gifsicle optimizer
│   ├── svg.py                  # scour optimizer + SVG sanitization
│   ├── avif.py                 # AVIF lossless optimizer (metadata strip only)
│   ├── heic.py                 # HEIC lossless optimizer (metadata strip only)
│   └── passthrough.py          # TIFF/BMP/PSD — best-effort Pillow optimization
├── estimation/
│   ├── __init__.py
│   ├── estimator.py            # Main estimation logic (dispatch per format)
│   ├── header_analysis.py      # Fast header-only image reading
│   └── heuristics.py           # Format-specific prediction rules
├── security/
│   ├── __init__.py
│   ├── file_validation.py      # Magic byte detection, size limits
│   ├── rate_limiter.py         # Redis-backed rate limiting (env-var driven)
│   ├── auth.py                 # API key authentication (GCP Secret Manager)
│   ├── ssrf.py                 # URL validation, private IP blocking, metadata endpoint blocking
│   └── svg_sanitizer.py        # SVG script stripping, XXE prevention via defusedxml
├── storage/
│   ├── __init__.py
│   └── gcs.py                  # Google Cloud Storage upload integration
├── utils/
│   ├── __init__.py
│   ├── format_detect.py        # Detect format from magic bytes (including APNG acTL chunk)
│   ├── metadata.py             # Selective EXIF/ICC stripping (Section 3.9)
│   ├── url_fetch.py            # httpx-based URL fetching with SSRF protection, timeouts, size limits
│   ├── subprocess_runner.py    # Async subprocess with stdin/stdout piping + 60s timeout
│   ├── concurrency.py          # Semaphore, queue depth limit, backpressure
│   └── logging.py              # Structured JSON logging (errors + critical cases only)
└── tests/
    ├── test_optimize.py
    ├── test_estimate.py
    ├── test_formats.py
    ├── test_security.py        # SSRF, SVG XSS, oversized files, malformed images
    ├── test_gcs.py             # GCS upload integration tests
    └── sample_images/          # Test images for each format
```

---

## 9. Optimization Pipeline Per Format

### 9.1 PNG Pipeline
```
Input PNG
  → pngquant --quality {q-15}-{q} (lossy quantization, dynamic from API quality param)
      → If exit code 99 (can't meet quality target):
          → oxipng only (lossless, ~7% reduction, pixel-perfect)
      → Else:
          → oxipng on pngquant output (squeeze extra bytes from lossy result)
  → Output PNG
```
Example: `quality=80` → `pngquant --quality 65-80`. The range-based floor ensures
no visible degradation below the requested quality. Images that can't meet the
threshold (complex photographic PNGs) fall back to lossless oxipng instead —
the image is never degraded below the quality floor.

If `optimization.png_lossy=false`, skip pngquant entirely and use oxipng only (lossless).

### 9.2 JPEG Pipeline
```
Input JPEG
  → Analyze quantization tables to estimate input quality
      → If input quality <= target quality:
          → Lossless optimization only (jpegtran: optimize Huffman tables, optionally make progressive)
      → Else:
          → MozJPEG (cjpeg) lossy re-encode at target quality
  → Output JPEG
```
**Avoiding unnecessary re-encoding:** JPEG quantization tables are analyzed to estimate
the original quality level. If the image is already at or below the target quality,
only lossless optimizations are applied (Huffman table optimization via `jpegtran`)
to avoid generation loss from decode→re-encode cycles. MozJPEG's `jpegtran` can
achieve ~5-15% lossless reduction on unoptimized JPEGs.

### 9.3 WebP Pipeline
```
Input WebP
  → Pillow decode + re-encode at target quality
  → If result >= 90% of input size:
      → Try cwebp CLI as fallback (may compress better)
      → Use whichever output is smaller
  → Output WebP
```
**Single-pass by default:** Pillow is tried first (in-process, fast). Only if the result
is suspiciously poor (>=90% of input), cwebp is tried as a fallback. This avoids
double CPU cost on every WebP image.

### 9.4 GIF Pipeline
```
Input GIF
  → gifsicle --optimize=3 (frame optimization, LZW recompression)
  → Output GIF
```

### 9.5 SVG Pipeline
```
Input SVG
  → scour (remove metadata, simplify paths, merge elements)
  → Output SVG
```

### 9.6 SVGZ Pipeline
```
Input SVGZ
  → gunzip → scour → gzip
  → Output SVGZ
```

### 9.7 AVIF / HEIC Pipeline
```
Input AVIF/HEIC
  → Lossless optimization only (strip metadata, optimize headers)
  → Do NOT decode + re-encode (causes generation loss on lossy formats)
  → Output AVIF/HEIC
```
**No re-encoding:** AVIF and HEIC are lossy formats. Decoding and re-encoding introduces
generation loss — each cycle degrades quality. Instead, only lossless operations
are applied: metadata stripping, header optimization. If no lossless reduction is
possible, return the original with 0% reduction (Section 3.14, Case 1).

### 9.8 TIFF / BMP / PSD (Passthrough)
```
Input TIFF/BMP/PSD
  → Pillow decode → re-encode to same format with optimization flags
  → If optimized size < original: return optimized
  → If optimized size >= original: return original with 0% reduction (Section 3.14, Case 1)
  → Output same format
```
These formats have limited optimization potential. The service attempts what it can
(e.g., TIFF compression, stripping metadata) and returns the original unchanged
if no reduction is achieved — this is not an error.

---

## 10. Watermarking (Future)

Watermarking will be added as a processing step in the pipeline:

```json
{
    "url": "https://cdn.shopify.com/s/files/.../product.png",
    "optimization": {
        "quality": 80
    },
    "watermark": {
        "image_url": "https://example.com/logo.png",
        "position": "bottom-right",
        "opacity": 0.5,
        "scale": 0.15
    }
}
```

Implementation: Pillow-based compositing. Applied after decode, before compression encoding (decode → watermark → compress → output). This avoids double encoding.

---

## 11. Verification & Testing Plan

### 11.1 Unit Tests
- Each optimizer: verify output is smaller, format preserved, valid image
- Estimation: verify predictions are within 20% of actual compression
- Format detection: verify all Shopify formats detected correctly

### 11.2 Integration Tests
- End-to-end `/optimize` with real images in each format
- End-to-end `/estimate` accuracy validation
- Concurrent load test (50+ simultaneous requests)

### 11.3 Real-World Validation
- Test with actual Shopify product images (various sizes, formats)
- Compare results against our API testing data (Imagor, imgproxy benchmarks)
- Verify no format produces larger output than input

### 11.4 Performance Benchmarks
- Single container throughput (images/sec per format)
- Estimation endpoint latency (target: <50ms p99)
- Memory usage under concurrent load
- Scale testing: verify linear scaling with container count

---

## 12. Implementation Phases

### Phase 1 — Core Optimizer (Week 1-2)
- Project setup, multi-stage Docker image with all binaries
- Pydantic schemas for request/response models
- Format detection via magic bytes (including APNG `acTL` detection)
- PNG optimizer (pngquant + oxipng, with APNG safety)
- JPEG optimizer (MozJPEG cjpeg + jpegtran lossless path)
- WebP optimizer (Pillow, cwebp fallback)
- Unified `/optimize` endpoint (file upload + URL-based input)
- File size validation (32 MB limit)
- File type validation (magic byte detection)
- SSRF protection for URL-based input
- Optimization guarantee enforcement (0% reduction response vs error for larger output)
- Selective metadata stripping (preserve orientation + ICC, strip GPS/camera)
- Structured error responses with proper HTTP status codes
- Basic tests

### Phase 2 — All Formats + Security (Week 2-3)
- GIF, SVG, SVGZ optimizers
- AVIF, HEIC lossless optimizers (metadata strip only, no re-encoding)
- TIFF/BMP/PSD passthrough
- SVG sanitization (strip `<script>`, prevent XXE via defusedxml)
- Animated image support (GIF, animated WebP, APNG)
- Authentication (API key via GCP Secret Manager)
- Redis-backed rate limiting (env-var configurable)
- CORS middleware configuration
- Concurrency control (semaphore + backpressure queue)

### Phase 3 — Estimation + GCS (Week 3-4)
- `/estimate` endpoint with header analysis + heuristics (thumbnail for JPEG/WebP only)
- GCS upload integration (request body driven)
- Response format handling (binary default, JSON with storage)
- URL fetching via httpx (SSRF-safe, timeout-aware)

### Phase 4 — Production Readiness (Week 4-5)
- Structured JSON error logging (errors + critical events only)
- Request ID injection (middleware)
- Graceful shutdown handling (30s drain on SIGTERM)
- Load testing
- Cloud Run deployment (min-instances=1, auto-scaling)
- Health endpoint

### Phase 5 — Watermarking & Enhancements (Future)
- Watermark support in `/optimize`
- Resize/crop options
- Format conversion (e.g., PNG → WebP)
- Quality auto-tuning (SSIM-based)

---

## 13. Reference: Test Data from API Evaluation

All test data collected during our evaluation is in the `internal-projects/image-compression-api-testing/` project:

- `run_tests.py` — CLI to test any compression API
- `output/` — Compressed sample files from all APIs
- `images/` — 13 generated test images (all formats)
- `api_clients/` — Clients for Imagor, Imaginary, imgproxy (reusable for benchmarking)

This project remains useful as a **benchmark suite** to validate our custom optimizer against the existing APIs.
