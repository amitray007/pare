# Phase 5 — Storage & URL Fetch

## Objectives

- Implement GCS upload integration triggered by the `storage` key in the request body
- Implement httpx-based URL fetching with SSRF protection at each redirect hop
- Wire URL fetching into the `/optimize` and `/estimate` endpoints
- Handle the dual response mode: raw bytes (no storage) vs JSON (with storage)

## Deliverables

- `storage/gcs.py` — Google Cloud Storage upload integration
- `utils/url_fetch.py` — httpx URL fetching with SSRF protection, streaming size limits

## Dependencies

- Phase 3 (API layer — routers that use storage and URL fetch)
- Phase 4 (SSRF protection — called during URL fetch)

---

## Files to Create

### 1. `storage/gcs.py`

**Purpose:** Upload optimized image bytes to Google Cloud Storage. Triggered when the request includes a `storage` config object.

**Upload flow:**

```
optimize_image(data, config) → OptimizeResult
    │
    ├── Has storage config?
    │   ├── NO → return binary response (handled in router)
    │   └── YES ↓
    │
    ├── Upload optimized_bytes to GCS
    │   ├── Bucket: storage.bucket
    │   ├── Path: storage.path
    │   ├── Content-Type: detected MIME type
    │   └── Public: storage.public
    │
    ├── Build GCS URLs
    │   ├── gs:// URL (always)
    │   └── Public HTTPS URL (if storage.public=True)
    │
    └── Return JSON OptimizeResponse with StorageResult
```

```python
from google.cloud import storage as gcs_lib
from schemas import StorageConfig, StorageResult
from exceptions import PareError


# MIME type mapping
MIME_TYPES = {
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


class GCSUploader:
    """Google Cloud Storage upload handler."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-initialized GCS client.

        Uses GOOGLE_APPLICATION_CREDENTIALS env var or
        workload identity (automatic on Cloud Run).
        """
        if self._client is None:
            self._client = gcs_lib.Client()
        return self._client

    async def upload(
        self,
        data: bytes,
        fmt: str,
        config: StorageConfig,
    ) -> StorageResult:
        """Upload optimized bytes to GCS.

        Args:
            data: Optimized image bytes.
            fmt: Image format string (e.g., "png", "jpeg").
            config: Storage configuration from the request.

        Returns:
            StorageResult with GCS URLs.

        Raises:
            PareError: If upload fails (bucket not found, permission denied, etc.).
        """
        try:
            bucket = self.client.bucket(
                config.bucket,
                user_project=config.project,
            )
            blob = bucket.blob(config.path)

            content_type = MIME_TYPES.get(fmt, "application/octet-stream")
            blob.upload_from_string(data, content_type=content_type)

            if config.public:
                blob.make_public()

            gs_url = f"gs://{config.bucket}/{config.path}"
            public_url = (
                f"https://storage.googleapis.com/{config.bucket}/{config.path}"
                if config.public
                else None
            )

            return StorageResult(
                provider="gcs",
                url=gs_url,
                public_url=public_url,
            )

        except Exception as e:
            raise PareError(
                f"GCS upload failed: {str(e)}",
                status_code=500,
                error_code="storage_upload_failed",
                bucket=config.bucket,
                path=config.path,
            )


# Module-level singleton
gcs_uploader = GCSUploader()
```

**GCS authentication:**

| Environment | Credential Source |
|-------------|-----------------|
| Cloud Run (production) | Workload identity (automatic) |
| Local development | `GOOGLE_APPLICATION_CREDENTIALS` env var pointing to service account JSON |
| CI/CD | Service account key or workload identity federation |

**Error handling:**

If storage upload fails, the entire request fails with a 500 error. The client requested storage — returning only bytes without the URL would be a partial/broken response. The error response includes the bucket and path for debugging.

---

### 2. `utils/url_fetch.py`

**Purpose:** Fetch images from user-supplied URLs using httpx. Integrates SSRF protection, streaming size limits, and timeout configuration.

**Fetch flow:**

```
fetch_image(url: str, is_authenticated: bool) → bytes
    │
    ├── 1. Validate URL via security/ssrf.validate_url(url)
    │   └── Reject private IPs, non-HTTPS, metadata endpoints
    │
    ├── 2. Create httpx.AsyncClient with:
    │   ├── timeout: 30s (auth) or 60s (public)
    │   ├── max_redirects: 0 (manual redirect following)
    │   ├── follow_redirects: False
    │   └── verify: True (TLS verification)
    │
    ├── 3. Follow redirects manually (max 5 hops)
    │   ├── For each redirect:
    │   │   ├── Validate redirect URL via ssrf.validate_url()
    │   │   └── Follow to next hop
    │   └── If 5 hops exceeded → URLFetchError
    │
    ├── 4. Stream response body
    │   ├── Count bytes as they arrive
    │   ├── If total > 32 MB → abort, raise FileTooLargeError
    │   └── Accumulate in memory (no temp files)
    │
    ├── 5. Validate HTTP status
    │   └── Non-2xx → URLFetchError with status code
    │
    └── 6. Return fetched bytes
```

```python
import httpx
from config import settings
from exceptions import URLFetchError, FileTooLargeError
from security.ssrf import validate_url


async def fetch_image(url: str, is_authenticated: bool = False) -> bytes:
    """Fetch image from a URL with SSRF protection and size limits.

    Args:
        url: User-supplied HTTPS URL.
        is_authenticated: Affects timeout (30s auth, 60s public).

    Returns:
        Raw image bytes.

    Raises:
        SSRFError: URL targets private/reserved IP.
        URLFetchError: Fetch failed (timeout, non-2xx, redirect limit).
        FileTooLargeError: Response body exceeds 32 MB.
    """
    # 1. SSRF validation
    validate_url(url)

    timeout = (
        settings.url_fetch_timeout
        if is_authenticated
        else settings.url_fetch_timeout * 2  # Double for public
    )

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        follow_redirects=False,
        verify=True,
    ) as client:
        # 2. Manual redirect following with SSRF check at each hop
        current_url = url
        for hop in range(settings.url_fetch_max_redirects + 1):
            response = await client.get(current_url)

            if response.is_redirect:
                redirect_url = str(response.next_request.url)
                validate_url(redirect_url)  # SSRF check on redirect target
                current_url = redirect_url
                continue

            if not response.is_success:
                raise URLFetchError(
                    f"URL returned HTTP {response.status_code}",
                    url=url,
                    status_code=response.status_code,
                )

            # 3. Check content length before reading body
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.max_file_size_bytes:
                raise FileTooLargeError(
                    f"URL content too large: {content_length} bytes",
                    file_size=int(content_length),
                    limit=settings.max_file_size_bytes,
                )

            return response.content

        raise URLFetchError(
            f"Too many redirects (>{settings.url_fetch_max_redirects})",
            url=url,
        )
```

**Streaming variant for large files:**

For production, the fetch should use streaming to avoid loading the full response into memory before checking size:

```python
async def fetch_image_streaming(url: str, is_authenticated: bool = False) -> bytes:
    """Streaming variant that aborts early if size limit exceeded.

    Reads the response body in chunks, counting bytes. Aborts
    as soon as the cumulative size exceeds 32 MB, before the
    full response is buffered.
    """
    # ... SSRF validation and redirect handling same as above ...

    async with client.stream("GET", current_url) as response:
        chunks = []
        total = 0
        async for chunk in response.aiter_bytes(chunk_size=65536):
            total += len(chunk)
            if total > settings.max_file_size_bytes:
                raise FileTooLargeError(
                    f"URL content exceeds {settings.max_file_size_mb} MB limit",
                    file_size=total,
                    limit=settings.max_file_size_bytes,
                )
            chunks.append(chunk)

        return b"".join(chunks)
```

**Integration with routers:**

The `/optimize` and `/estimate` routers call `fetch_image()` when the request is JSON with a `url` field:

```python
# In routers/optimize.py
if request.content_type == "application/json":
    body = await request.json()
    req = OptimizeRequest(**body)
    data = await fetch_image(req.url, request.state.is_authenticated)
    config = req.optimization
    storage_config = req.storage
else:
    # Multipart handling...
```

---

## Response Mode Logic

The presence or absence of the `storage` key determines the response format. This logic lives in `routers/optimize.py`:

```python
async def optimize(request, ...):
    # ... parse input, validate, optimize ...

    result = await optimize_image(data, config)

    if storage_config:
        # Upload to storage, return JSON
        storage_result = await gcs_uploader.upload(
            result.optimized_bytes,
            result.format,
            storage_config,
        )
        return JSONResponse(content={
            "success": True,
            "original_size": result.original_size,
            "optimized_size": result.optimized_size,
            "reduction_percent": result.reduction_percent,
            "format": result.format,
            "method": result.method,
            "storage": storage_result.model_dump(),
        })
    else:
        # Return raw bytes with X-* headers
        return _build_binary_response(result, request.state.request_id)
```

---

## Environment Variables Introduced

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | (auto) | Path to GCS service account JSON key |
| `URL_FETCH_TIMEOUT` | `30` | Base timeout for URL fetching (seconds) |
| `URL_FETCH_MAX_REDIRECTS` | `5` | Maximum redirect hops |

---

## Verification Steps

### Manual verification

```bash
# Test URL-based optimization
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"}' \
  -o optimized.png -D -
# Check X-* headers and file size

# Test URL with storage (requires GCS credentials)
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "url": "https://example.com/image.png",
    "optimization": {"quality": 80},
    "storage": {
      "provider": "gcs",
      "bucket": "test-bucket",
      "path": "test/optimized.png",
      "public": true
    }
  }'
# Expected: JSON response with storage URLs

# Test file upload with storage
curl -X POST http://localhost:8080/optimize \
  -H "Authorization: Bearer $API_KEY" \
  -F "file=@sample.png" \
  -F 'options={"storage": {"provider": "gcs", "bucket": "test-bucket", "path": "test/opt.png"}}'
# Expected: JSON response

# Test URL fetch size limit
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/huge-image.tiff"}'
# Expected: 413 if image > 32 MB

# Test redirect chain SSRF protection
# (Requires a redirect service that bounces to internal IP)
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://redirect-service.example.com/to-internal"}'
# Expected: 422 ssrf_blocked at the redirect hop

# Test URL estimation
curl -X POST http://localhost:8080/estimate \
  -H "Content-Type: application/json" \
  -d '{"url": "https://upload.wikimedia.org/wikipedia/commons/4/47/PNG_transparency_demonstration_1.png"}'
# Expected: JSON estimate response
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_gcs_upload_success` | Bytes uploaded to correct bucket/path, URLs returned |
| `test_gcs_upload_public` | `storage.public=True` makes object publicly accessible |
| `test_gcs_upload_private` | `storage.public=False` → no public_url in response |
| `test_gcs_upload_custom_project` | `storage.project` passed to GCS client |
| `test_gcs_upload_failure` | GCS error → 500 with storage_upload_failed error code |
| `test_url_fetch_basic` | Fetch public HTTPS URL, return bytes |
| `test_url_fetch_ssrf_blocked` | Private IP URL rejected before fetch |
| `test_url_fetch_redirect_ssrf` | Redirect to private IP rejected at hop |
| `test_url_fetch_too_many_redirects` | >5 redirects → URLFetchError |
| `test_url_fetch_size_limit` | >32 MB response body → FileTooLargeError |
| `test_url_fetch_streaming_abort` | Streaming aborts mid-download when limit exceeded |
| `test_url_fetch_timeout` | Slow server → timeout error |
| `test_url_fetch_non_200` | 404/500 from remote → URLFetchError |
| `test_optimize_url_mode` | End-to-end: JSON body with URL → optimized bytes |
| `test_optimize_url_with_storage` | End-to-end: URL + storage → JSON with GCS URLs |
| `test_estimate_url_mode` | End-to-end: URL estimation works |
| `test_response_mode_binary` | No storage → raw bytes with X-* headers |
| `test_response_mode_json` | Storage present → JSON response |
