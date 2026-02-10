# Pare — Technical Implementation Overview

## Executive Summary

**Pare** is a custom Image Optimizer Service that routes each image format to a best-in-class dedicated compression tool. Unlike general-purpose image processing APIs (Imagor, imgproxy, Imaginary) that treat compression as a side effect of resizing/cropping, Pare is purpose-built for compression — achieving 50-75% file size reductions where existing APIs achieve 0-5%.

Pare is deployed as a single Docker image on Google Cloud Run. All compression binaries are baked into the container. The API layer (Python/FastAPI) orchestrates format detection, tool dispatch, and response formatting. Scaling is horizontal — add containers.

**Consumers:** Shopify App backend, Marketing Website free tool, future internal tools.

---

## High-Level Architecture

```
Clients (Shopify App, Marketing Website, Future Tools)
                    │
                    ▼
            Cloud Run Load Balancer
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Container│ │ Container│ │ Container│   (auto-scaled)
  │         │ │         │ │         │
  │ FastAPI │ │ FastAPI │ │ FastAPI │
  │ Uvicorn │ │ Uvicorn │ │ Uvicorn │
  │    │    │ │    │    │ │    │    │
  │ ┌──┴──┐ │ │ ┌──┴──┐ │ │ ┌──┴──┐ │
  │ │Tools│ │ │ │Tools│ │ │ │Tools│ │   (all binaries baked in)
  │ │     │ │ │ │     │ │ │ │     │ │
  │ │pngq.│ │ │ │pngq.│ │ │ │pngq.│ │
  │ │moz. │ │ │ │moz. │ │ │ │moz. │ │
  │ │gifsi│ │ │ │gifsi│ │ │ │gifsi│ │
  │ │cwebp│ │ │ │cwebp│ │ │ │cwebp│ │
  │ │scour│ │ │ │scour│ │ │ │scour│ │
  │ └─────┘ │ │ └─────┘ │ │ └─────┘ │
  └─────────┘ └─────────┘ └─────────┘
       │            │            │
       └────────────┼────────────┘
                    │
              ┌─────┴─────┐
              │   Redis   │  (rate limiting — VPC internal)
              └───────────┘
              ┌─────┴─────┐
              │    GCS    │  (optional storage upload)
              └───────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.12 | Orchestrator — 98% CPU time is in C/Rust binaries |
| Framework | FastAPI + Uvicorn | Async HTTP server, Pydantic validation |
| PNG | pngquant (CLI) + pyoxipng (library) | Lossy quantization + lossless optimization |
| JPEG | MozJPEG cjpeg/jpegtran (CLI) | Lossy re-encode + lossless Huffman optimization |
| WebP | Pillow (library) + cwebp (CLI fallback) | In-process encode with CLI fallback |
| GIF | gifsicle (CLI) | Frame optimization, LZW recompression |
| SVG/SVGZ | scour (Python library) | Structure optimization, metadata removal |
| AVIF/HEIC | pillow-heif (library) | Metadata stripping only (no re-encode) |
| TIFF/BMP/PSD | Pillow (library) | Best-effort optimization |
| URL Fetching | httpx | Async, SSRF-safe URL fetching |
| Storage | google-cloud-storage | GCS upload integration |
| Rate Limiting | redis[hiredis] | Shared state across Cloud Run instances |
| SVG Security | defusedxml | XXE prevention in SVG parsing |
| Config | pydantic-settings | Environment variable management |
| Deployment | Cloud Run | Auto-scaling, min-instances=1 |
| Container | Docker (multi-stage) | MozJPEG built from source in stage 1 |

---

## API Surface

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/optimize` | POST | Compress an image (file upload or URL) |
| `/estimate` | POST | Predict compression savings without compressing |
| `/health` | GET | Service health + available tool inventory |

**Input modes:** Multipart file upload (`file` field + optional `options` JSON string) or JSON body with `url` field.

**Response modes:** Raw optimized bytes with `X-*` headers (default) or JSON with storage URL (when `storage` key present in request).

---

## Format-to-Tool Routing

| Format | Tool | Expected Reduction | Pipeline Type |
|--------|------|--------------------|---------------|
| PNG | pngquant + oxipng | ~70-75% | CLI → Library |
| JPEG | MozJPEG (cjpeg/jpegtran) | ~50-65% | CLI |
| WebP | Pillow (+ cwebp fallback) | ~40-55% | Library → CLI |
| GIF | gifsicle | ~20-25% | CLI |
| SVG | scour | ~40-60% | Library |
| SVGZ | scour + gzip | ~40-60% | Library |
| AVIF | pillow-heif | ~15-20% | Library (metadata only) |
| HEIC | pillow-heif | ~15-20% | Library (metadata only) |
| TIFF | Pillow | Format-dependent | Library |
| BMP | Pillow | Format-dependent | Library |
| PSD | Pillow | Format-dependent | Library |

---

## Complete File Inventory (46 files across 6 phases)

### Phase 1 — Foundation & Docker (7 files)

| File | Purpose |
|------|---------|
| `Dockerfile` | Multi-stage build: MozJPEG from source → slim Python image |
| `requirements.txt` | Python dependencies |
| `main.py` | FastAPI app, router mounting, startup/shutdown hooks |
| `config.py` | Pydantic BaseSettings, env vars, quality defaults, limits |
| `schemas.py` | Request/response Pydantic models |
| `exceptions.py` | Custom exception classes with HTTP status mapping |
| `docker-compose.yml` | Local development with Redis |

### Phase 2 — Optimization Engine (13 files)

| File | Purpose |
|------|---------|
| `utils/format_detect.py` | Magic byte detection, APNG acTL chunk detection |
| `utils/metadata.py` | Selective EXIF/ICC stripping |
| `utils/subprocess_runner.py` | Async subprocess with stdin/stdout piping + 60s timeout |
| `optimizers/__init__.py` | Package init |
| `optimizers/base.py` | BaseOptimizer abstract class |
| `optimizers/router.py` | Format detection → optimizer dispatch |
| `optimizers/png.py` | pngquant + oxipng pipeline (APNG safety) |
| `optimizers/jpeg.py` | MozJPEG cjpeg + jpegtran lossless path |
| `optimizers/webp.py` | Pillow + cwebp fallback |
| `optimizers/gif.py` | gifsicle optimizer |
| `optimizers/svg.py` | scour + SVG sanitization |
| `optimizers/avif.py` | AVIF metadata strip (no re-encode) |
| `optimizers/heic.py` | HEIC metadata strip (no re-encode) |
| `optimizers/passthrough.py` | TIFF/BMP/PSD best-effort Pillow |

### Phase 3 — API Layer (5 files)

| File | Purpose |
|------|---------|
| `routers/__init__.py` | Package init |
| `routers/optimize.py` | POST /optimize endpoint |
| `routers/estimate.py` | POST /estimate endpoint |
| `routers/health.py` | GET /health endpoint |
| `utils/concurrency.py` | Semaphore, queue depth, backpressure |

### Phase 4 — Security (5 files)

| File | Purpose |
|------|---------|
| `security/__init__.py` | Package init |
| `security/file_validation.py` | Magic byte detection, size limits |
| `security/ssrf.py` | URL validation, private IP blocking |
| `security/svg_sanitizer.py` | Script stripping, XXE prevention |
| `security/rate_limiter.py` | Redis-backed sliding window rate limiting |
| `security/auth.py` | API key authentication |
| `middleware.py` | CORS, auth, rate limiting, request ID injection |

### Phase 5 — Storage & URL Fetch (3 files)

| File | Purpose |
|------|---------|
| `storage/__init__.py` | Package init |
| `storage/gcs.py` | GCS upload integration |
| `utils/url_fetch.py` | httpx URL fetching with SSRF protection |

### Phase 6 — Production Readiness (8 files)

| File | Purpose |
|------|---------|
| `utils/__init__.py` | Package init |
| `utils/logging.py` | Structured JSON logging |
| `tests/test_optimize.py` | Optimize endpoint tests |
| `tests/test_estimate.py` | Estimate endpoint tests |
| `tests/test_formats.py` | Per-format optimization tests |
| `tests/test_security.py` | SSRF, SVG XSS, oversized files |
| `tests/test_gcs.py` | GCS upload integration tests |
| `tests/sample_images/` | Test images (symlinked from test framework) |

### Phase 7 — Estimation Engine (3 files)

| File | Purpose |
|------|---------|
| `estimation/__init__.py` | Package init |
| `estimation/estimator.py` | Main estimation dispatch |
| `estimation/header_analysis.py` | Fast header-only reading |
| `estimation/heuristics.py` | Format-specific prediction rules |

---

## Phase Dependency Graph

```
Phase 1: Foundation & Docker
    │
    ├──→ Phase 2: Optimization Engine  (depends on: Phase 1)
    │        │
    │        ├──→ Phase 3: API Layer  (depends on: Phase 2)
    │        │        │
    │        │        ├──→ Phase 5: Storage & URL Fetch  (depends on: Phase 3, Phase 4)
    │        │        │
    │        │        └──→ Phase 6: Production Readiness  (depends on: Phase 3, Phase 4, Phase 5)
    │        │
    │        └──→ Phase 4: Security  (depends on: Phase 2)
    │
    └──→ (Phase 4 can begin in parallel with Phase 3 after Phase 2)
```

**Critical path:** Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6

**Parallelizable:** Phase 4 (Security) can be developed in parallel with Phase 3 (API Layer) after Phase 2 is complete.

---

## Environment Variables Summary

| Variable | Default | Phase | Description |
|----------|---------|-------|-------------|
| `PORT` | `8080` | 1 | HTTP port (Cloud Run sets automatically) |
| `WORKERS` | `4` | 1 | Uvicorn worker count |
| `MAX_FILE_SIZE_MB` | `32` | 1 | Maximum upload size |
| `DEFAULT_QUALITY` | `80` | 1 | Default optimization quality (1-100) |
| `TOOL_TIMEOUT_SECONDS` | `60` | 2 | Per-tool invocation timeout |
| `COMPRESSION_SEMAPHORE_SIZE` | `CPU_COUNT` | 3 | Max concurrent compression jobs |
| `MAX_QUEUE_DEPTH` | `2 * CPU_COUNT` | 3 | Backpressure queue limit |
| `REDIS_URL` | (required) | 4 | Redis connection for rate limiting |
| `RATE_LIMIT_PUBLIC_RPM` | `60` | 4 | Requests/min for unauthenticated |
| `RATE_LIMIT_PUBLIC_BURST` | `10` | 4 | Burst limit for unauthenticated |
| `RATE_LIMIT_AUTH_ENABLED` | `false` | 4 | Whether to rate limit authenticated requests |
| `RATE_LIMIT_AUTH_RPM` | `0` | 4 | Requests/min for authenticated (0=unlimited) |
| `API_KEY` | (secret) | 4 | API key (from GCP Secret Manager) |
| `ALLOWED_ORIGINS` | `*` | 4 | CORS allowed origins (comma-separated) |
| `GOOGLE_APPLICATION_CREDENTIALS` | (auto) | 5 | GCS service account key path |
| `URL_FETCH_TIMEOUT` | `30` | 5 | httpx fetch timeout (seconds) |
| `URL_FETCH_MAX_REDIRECTS` | `5` | 5 | Max redirect hops for URL fetching |
| `LOG_LEVEL` | `ERROR` | 6 | Logging level (ERROR in production) |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | 6 | Seconds to drain requests on SIGTERM |

---

## Risk Areas & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| MozJPEG build fails in Docker | Blocks JPEG optimization | Pin MozJPEG version, cache build layer, document exact cmake flags |
| pngquant exit code 99 (quality too low) | No PNG output | Fallback to lossless oxipng — documented in PNG pipeline |
| APNG frames destroyed by pngquant | Broken animated PNGs | Detect `acTL` chunk before routing, skip pngquant for APNG |
| HEIC codec incompatibility | Silently corrupted output | Metadata-strip-only approach — no decode/re-encode |
| SSRF via DNS rebinding | Internal network exposure | Resolve DNS → check IP → fetch (not check URL string) |
| SVG XSS / XXE | Security vulnerability | defusedxml for parsing, strip scripts/event handlers/foreignObject |
| Memory pressure from concurrent 32MB uploads | OOM kill | Semaphore + queue depth + 503 backpressure |
| Redis unavailable | Rate limiting fails | Fail-open for rate limiting (allow requests, log warning) |
| Tool hangs indefinitely | Worker blocked | 60s timeout on all subprocess invocations |

---

## References to Existing Test Framework

The test framework at `/internal-projects/image-compression-api-testing/` provides:

| Resource | Path | Use in Pare |
|----------|------|-------------|
| Sample images (13 formats) | `images/sample.{png,jpg,gif,webp,avif,heic,svg,svgz,...}` | Copy/symlink as test fixtures |
| Image generator | `generate_images.py` | Generate new test images with specific characteristics |
| API client base class | `api_clients/base.py` | Pattern reference for HTTP client structure |
| Compression benchmarks | `output/` | Baseline to validate Pare exceeds these results |
| Test runner | `run_tests.py` | Adapt as benchmark comparison tool |

**Validation target:** Pare must outperform all tested APIs (Imagor, imgproxy, Imaginary) on every format. The test framework's benchmark data provides the floor.

---

## PRD Traceability

| Document | PRD Sections Covered |
|----------|---------------------|
| `phase-1-foundation-docker.md` | 7 (Docker), 8 (Project Structure), 6 (Tech Decision) |
| `phase-2-optimization-engine.md` | 3.2 (Routing), 9 (Pipelines), 3.9 (Metadata), 3.11 (Animation) |
| `phase-3-api-layer.md` | 3.3-3.6 (API), 4 (Estimation), 5 (Concurrency) |
| `phase-4-security.md` | 3.7 (Security), 3.8 (Auth/Rate Limiting) |
| `phase-5-storage-url-fetch.md` | 3.12 (Storage), 3.13 (URL Fetching) |
| `phase-6-production-readiness.md` | 5.5 (Logging), 11 (Testing), 12 (Phases) |
