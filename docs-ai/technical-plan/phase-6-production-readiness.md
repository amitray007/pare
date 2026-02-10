# Phase 6 — Production Readiness

## Objectives

- Implement structured JSON logging (errors + critical events only)
- Implement graceful shutdown handling (drain in-flight requests on SIGTERM)
- Build the health check endpoint with tool verification
- Configure Docker Compose for local development
- Document Cloud Run deployment configuration
- Build the test suite (unit + integration tests)
- Set up CI pipeline basics

## Deliverables

- `utils/logging.py` — structured JSON logging
- `tests/test_optimize.py` — optimize endpoint tests
- `tests/test_estimate.py` — estimate endpoint tests
- `tests/test_formats.py` — per-format optimization tests
- `tests/test_security.py` — SSRF, SVG XSS, file validation tests
- `tests/test_gcs.py` — GCS upload integration tests
- Updated `main.py` with graceful shutdown hooks

## Dependencies

- Phase 3 (API layer)
- Phase 4 (security)
- Phase 5 (storage + URL fetch)

---

## Files to Create

### 1. `utils/logging.py`

**Purpose:** Structured JSON logging compatible with Google Cloud Logging. Minimal — only errors and critical operational events. No verbose info/debug logs in production.

**Log schema (JSON per line):**

```json
{
    "severity": "ERROR",
    "message": "Tool crash: pngquant exited with code 139",
    "timestamp": "2024-01-15T10:30:00.123Z",
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "context": {
        "tool": "pngquant",
        "format": "png",
        "file_size": 2351691,
        "duration_ms": 45123,
        "error": "Segmentation fault"
    }
}
```

**What gets logged:**

| Level | Event | Context Fields |
|-------|-------|----------------|
| ERROR | Tool crash / timeout | tool, format, file_size, duration_ms, error |
| ERROR | SSRF attempt blocked | source_ip, url_attempted |
| ERROR | File validation rejection | reason, file_size, detected_bytes |
| ERROR | Storage upload failure | provider, bucket, path, error |
| ERROR | Unexpected exception | full traceback |
| WARNING | Optimization produced larger file | format, input_size, output_size |
| WARNING | Rate limit hit | source_ip, endpoint |
| WARNING | Tool returned 0% on compressible format | format, tool, file_size |

**What is NOT logged:**
- Successful requests (use Cloud Run built-in metrics instead)
- Request bodies or image content
- Auth tokens
- Debug/info messages (production uses ERROR level)

```python
import json
import logging
import sys
from datetime import datetime, timezone

from config import settings


class StructuredFormatter(logging.Formatter):
    """JSON formatter for Google Cloud Logging compatibility.

    Outputs one JSON object per line with fields:
    - severity: Maps Python levels to Cloud Logging severity
    - message: Human-readable message
    - timestamp: ISO 8601 with timezone
    - request_id: From log record extras (if available)
    - context: Additional structured data
    """

    SEVERITY_MAP = {
        "DEBUG": "DEBUG",
        "INFO": "INFO",
        "WARNING": "WARNING",
        "ERROR": "ERROR",
        "CRITICAL": "CRITICAL",
    }

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "severity": self.SEVERITY_MAP.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add request_id if available
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id

        # Add context dict if available
        if hasattr(record, "context"):
            log_entry["context"] = record.context

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["traceback"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging():
    """Configure structured JSON logging.

    Call once at application startup (in main.py lifespan).
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger("pare")
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Suppress noisy library loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'pare' namespace."""
    return logging.getLogger(f"pare.{name}")
```

**Usage pattern:**

```python
logger = get_logger("optimizers.png")

# Log with context
logger.error(
    "Tool crash: pngquant exited with code 139",
    extra={
        "request_id": request_id,
        "context": {
            "tool": "pngquant",
            "format": "png",
            "file_size": len(data),
            "duration_ms": elapsed_ms,
        },
    },
)
```

---

### 2. Graceful Shutdown (update to `main.py`)

**Purpose:** When Cloud Run sends SIGTERM (during scale-down or deployment), drain in-flight requests before exiting.

Uvicorn handles this via `--timeout-graceful-shutdown`. The application-level lifespan hook provides a place for cleanup:

```python
# Updated lifespan in main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    setup_logging()
    logger.info("Pare starting up")

    # Verify compression tools
    from routers.health import check_tools
    tools = check_tools()
    missing = [t for t, available in tools.items() if not available]
    if missing:
        logger.warning(f"Missing tools: {missing}")

    yield

    # --- Shutdown ---
    # Uvicorn's --timeout-graceful-shutdown handles connection draining.
    # This hook is for application-level cleanup:
    # - Close Redis connection pool
    # - Close httpx client sessions
    # - Flush any pending logs

    from security.rate_limiter import _redis
    if _redis:
        await _redis.close()

    logger.info("Pare shutting down")
```

**Cloud Run shutdown timeline:**

```
SIGTERM received
    │
    ├── Uvicorn stops accepting new connections
    │
    ├── In-flight requests continue processing (up to 30s)
    │
    ├── After 30s (GRACEFUL_SHUTDOWN_TIMEOUT), Uvicorn force-closes
    │
    └── Application lifespan shutdown hook runs
        ├── Close Redis pool
        ├── Flush logs
        └── Process exits
```

---

### 3. Cloud Run Deployment Configuration

**`cloudbuild.yaml` (Cloud Build):**

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/pare:$COMMIT_SHA', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/pare:$COMMIT_SHA']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'pare'
      - '--image=gcr.io/$PROJECT_ID/pare:$COMMIT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--memory=2Gi'
      - '--cpu=4'
      - '--min-instances=1'
      - '--max-instances=32'
      - '--timeout=300'
      - '--concurrency=80'
      - '--set-env-vars=WORKERS=4,LOG_LEVEL=ERROR'
      - '--set-secrets=API_KEY=pare-api-key:latest,REDIS_URL=pare-redis-url:latest'
```

**Cloud Run settings rationale:**

| Setting | Value | Rationale |
|---------|-------|-----------|
| `memory` | 2Gi | 1.2GB for compression + 800MB for Python/connections |
| `cpu` | 4 | Matches WORKERS=4 and COMPRESSION_SEMAPHORE_SIZE default |
| `min-instances` | 1 | Cold start mitigation — always one warm container |
| `max-instances` | 32 | ~500 images/sec peak capacity at 16 img/sec/container |
| `timeout` | 300 | Max request duration (5 min for large files) |
| `concurrency` | 80 | Requests per container instance (async handles queuing) |
| `--set-secrets` | API_KEY, REDIS_URL | Secrets from GCP Secret Manager |

---

### 4. Test Suite

#### `tests/conftest.py` — Shared fixtures

```python
import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_png():
    """Load sample PNG from test framework."""
    with open("tests/sample_images/sample.png", "rb") as f:
        return f.read()


@pytest.fixture
def sample_jpeg():
    with open("tests/sample_images/sample.jpg", "rb") as f:
        return f.read()


# ... fixtures for all formats


@pytest.fixture
def auth_headers():
    """Headers for authenticated requests."""
    return {"Authorization": "Bearer test-api-key"}
```

#### `tests/test_optimize.py`

| Test | Description |
|------|-------------|
| `test_optimize_png_file_upload` | Upload PNG → smaller PNG returned with correct headers |
| `test_optimize_jpeg_file_upload` | Upload JPEG → smaller JPEG returned |
| `test_optimize_with_quality` | Custom quality applied (verify size differs from default) |
| `test_optimize_progressive_jpeg` | `progressive_jpeg=True` → progressive JPEG output |
| `test_optimize_png_lossless` | `png_lossy=False` → oxipng only (larger than lossy but pixel-perfect) |
| `test_optimize_already_optimized` | Re-optimizing output → 200 with 0% reduction |
| `test_optimize_returns_valid_image` | Output bytes are a valid image (Pillow can open) |
| `test_optimize_preserves_format` | Input PNG → output PNG (no format conversion) |
| `test_optimize_binary_response_headers` | All X-* headers present and correct |
| `test_optimize_json_response_with_storage` | Storage config → JSON response (mock GCS) |
| `test_optimize_request_id` | X-Request-ID present and unique |
| `test_optimize_413_file_too_large` | >32 MB → 413 error |
| `test_optimize_415_unsupported_format` | Random bytes → 415 error |
| `test_optimize_400_malformed_options` | Invalid JSON in options → 400 |
| `test_optimize_url_mode` | JSON body with URL → optimized bytes (mock httpx) |

#### `tests/test_estimate.py`

| Test | Description |
|------|-------------|
| `test_estimate_png` | PNG → estimate with method "pngquant + oxipng" |
| `test_estimate_jpeg_high_quality` | High-quality JPEG → "high" optimization_potential |
| `test_estimate_jpeg_low_quality` | Low-quality JPEG → "low" optimization_potential |
| `test_estimate_gif` | GIF → estimate with method "gifsicle" |
| `test_estimate_svg` | SVG → estimate with method "scour" |
| `test_estimate_accuracy` | Estimate within 20% of actual compression (known images) |
| `test_estimate_confidence_jpeg` | JPEG → "high" confidence (thumbnail method used) |
| `test_estimate_confidence_png` | PNG → "medium" confidence (heuristics only) |
| `test_estimate_already_optimized` | Optimized image → `already_optimized: true` |

#### `tests/test_formats.py`

| Test | Description |
|------|-------------|
| `test_format_png_reduction` | PNG reduction > 50% on test image |
| `test_format_jpeg_reduction` | JPEG reduction > 30% on test image |
| `test_format_webp_reduction` | WebP reduction > 20% on test image |
| `test_format_gif_reduction` | GIF reduction > 10% on test image |
| `test_format_svg_reduction` | SVG reduction > 20% on test image |
| `test_format_avif_no_generation_loss` | AVIF → metadata stripped, pixels unchanged |
| `test_format_heic_no_generation_loss` | HEIC → metadata stripped, pixels unchanged |
| `test_format_tiff_passthrough` | TIFF → optimized or returned unchanged |
| `test_format_bmp_passthrough` | BMP → optimized or returned unchanged |
| `test_format_psd_passthrough` | PSD → optimized or returned unchanged |
| `test_format_apng_frames_preserved` | APNG → all animation frames intact |
| `test_format_animated_gif` | Animated GIF → all frames preserved |
| `test_format_animated_webp` | Animated WebP → all frames preserved |
| `test_format_svgz_roundtrip` | SVGZ → decompressed, optimized, recompressed |
| `test_all_formats_never_larger` | No format ever returns output > input |

#### `tests/test_security.py`

| Test | Description |
|------|-------------|
| `test_ssrf_private_ranges` | All RFC 1918 ranges blocked |
| `test_ssrf_link_local` | 169.254.x.x blocked |
| `test_ssrf_localhost` | 127.0.0.1 / ::1 blocked |
| `test_ssrf_metadata_google` | metadata.google.internal blocked |
| `test_ssrf_http_rejected` | HTTP scheme rejected (HTTPS only) |
| `test_ssrf_redirect_to_private` | Redirect chain to private IP blocked |
| `test_svg_script_stripped` | `<script>` removed from SVG |
| `test_svg_event_handlers_stripped` | `onload`, `onclick` etc. removed |
| `test_svg_xxe_blocked` | External entity expansion prevented |
| `test_svg_foreign_object_stripped` | `<foreignObject>` removed |
| `test_svg_data_uri_stripped` | `data:text/html` in href removed |
| `test_svg_valid_content_preserved` | Non-malicious SVG content intact |
| `test_file_size_32mb_limit` | Files > 32 MB rejected |
| `test_file_unknown_format_rejected` | Random bytes → 415 |
| `test_rate_limit_enforced` | 61st request/minute → 429 |
| `test_rate_limit_burst` | 11th request in 10s → 429 |
| `test_rate_limit_auth_bypass` | Authenticated requests not limited |
| `test_auth_valid_key` | Valid Bearer token → authenticated |
| `test_auth_invalid_key` | Invalid key → 401 |
| `test_auth_no_header` | No auth → public (not error) |

#### `tests/test_gcs.py`

| Test | Description |
|------|-------------|
| `test_gcs_upload_mock` | Upload bytes to mocked GCS → correct bucket/path |
| `test_gcs_upload_public` | `public=True` → blob.make_public() called |
| `test_gcs_upload_private` | `public=False` → no public_url in result |
| `test_gcs_upload_custom_project` | `project` field passed to GCS client |
| `test_gcs_upload_failure_handling` | GCS error → 500 with error details |

---

### 5. Test Images

Copy or symlink test images from the existing test framework:

```bash
# From the test framework at /internal-projects/image-compression-api-testing/
cp images/sample.png tests/sample_images/
cp images/sample.jpg tests/sample_images/
cp images/sample.gif tests/sample_images/
cp images/sample.webp tests/sample_images/
cp images/sample.avif tests/sample_images/
cp images/sample.heic tests/sample_images/
cp images/sample.svg tests/sample_images/
cp images/sample.svgz tests/sample_images/
```

Additional test images to generate:

| Image | Purpose |
|-------|---------|
| `tiny.png` (1x1 pixel) | Edge case — minimal image |
| `large.png` (~30 MB) | Near the size limit |
| `already-optimized.png` | Pre-optimized with pngquant (0% reduction expected) |
| `animated.gif` (multi-frame) | Animation preservation test |
| `animated.webp` (multi-frame) | Animation preservation test |
| `animated.png` (APNG) | APNG safety test (must not destroy frames) |
| `malicious.svg` | XSS + XXE test |
| `high-quality.jpg` (q=98) | JPEG quality estimation test |
| `low-quality.jpg` (q=30) | JPEG skip re-encode test |

---

## Performance Benchmarks

Run after all phases are complete to validate against PRD targets:

| Metric | Target | How to Measure |
|--------|--------|---------------|
| Single container optimize throughput | ~16 img/sec | `wrk` or `hey` with concurrent requests |
| Single container estimate throughput | ~1000 req/sec | `wrk` against /estimate |
| Estimate latency (p99) | < 50ms | `hey` with latency histogram |
| Memory under load | < 2 GB | Docker stats during load test |
| PNG reduction (real-world) | > 50% | Compare against test framework benchmarks |
| JPEG reduction | > 30% | Compare against test framework benchmarks |
| Cold start time | < 5s | Cloud Run cold start metric |

**Benchmark against existing APIs:**

Use the test framework at `/internal-projects/image-compression-api-testing/` to run the same test images through Pare and compare results:

```bash
# Run existing benchmark suite against Pare
cd /internal-projects/image-compression-api-testing/
python run_tests.py --api pare --url http://localhost:8080

# Compare results
python compare_results.py --baseline output/imagor/ --challenger output/pare/
```

**Validation target:** Pare must outperform all tested APIs on every supported format. Specifically:
- PNG: > 70% reduction (vs ~1% from APIs)
- JPEG: > 50% reduction (vs ~55% from Imagor MozJPEG — competitive)
- WebP: > 40% reduction
- GIF: > 20% reduction
- SVG: > 30% reduction

---

## Environment Variables Introduced

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `ERROR` | Logging level |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | Seconds to drain on SIGTERM |

---

## Verification Steps

### Manual verification

```bash
# Verify structured logging output
docker compose up -d
curl -X POST http://localhost:8080/optimize \
  -F "file=@random-bytes.bin"  # Will cause an error
docker compose logs pare | python -m json.tool  # Should be valid JSON

# Verify health endpoint
curl http://localhost:8080/health | jq .

# Verify graceful shutdown
docker compose up -d
# Send a request that takes a few seconds
curl -X POST http://localhost:8080/optimize -F "file=@large.png" &
sleep 0.5
docker compose stop  # Should wait for in-flight request
# Request should complete successfully

# Run test suite
docker compose run --rm pare pytest tests/ -v

# Run load test
docker compose up -d
hey -n 1000 -c 50 -m POST -F "file=@sample.png" http://localhost:8080/optimize
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_structured_log_format` | Log output is valid JSON with required fields |
| `test_log_error_includes_context` | Error logs include tool, format, file_size |
| `test_log_no_success_logging` | Successful requests produce no log output |
| `test_health_all_tools_available` | Health reports all tools present |
| `test_health_degraded_on_missing_tool` | Missing tool → "degraded" status |
| `test_request_id_propagation` | Request ID in response matches logs |
