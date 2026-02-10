# Phase 4 — Security

## Objectives

- Implement SSRF protection for URL-based input (DNS resolution → IP validation)
- Implement SVG sanitization (script stripping, XXE prevention, foreignObject removal)
- Implement file validation (magic byte gating, size limits)
- Implement Redis-backed rate limiting with sliding window
- Implement API key authentication (single key from env var / GCP Secret Manager)
- Wire security middleware into the FastAPI request lifecycle

## Deliverables

- `security/ssrf.py` — URL validation, private IP blocking, metadata endpoint blocking
- `security/svg_sanitizer.py` — SVG script stripping, XXE prevention
- `security/file_validation.py` — magic byte detection, size limits
- `security/rate_limiter.py` — Redis-backed sliding window rate limiting
- `security/auth.py` — API key authentication
- `middleware.py` — CORS, auth, rate limiting, request ID (update from Phase 1 stub)

## Dependencies

- Phase 2 (format detection used by file validation)
- Can be developed in parallel with Phase 3

---

## Files to Create

### 1. `security/ssrf.py`

**Purpose:** Prevent Server-Side Request Forgery when the service fetches user-supplied URLs. An attacker could submit URLs like `http://169.254.169.254/latest/meta-data/` to probe the Cloud Run metadata server.

**SSRF protection algorithm:**

```
validate_url(url: str) → validated URL string
    │
    ├── 1. Parse URL
    │   └── Reject if scheme != "https"
    │
    ├── 2. Extract hostname
    │   └── Reject if hostname is a known metadata endpoint:
    │       - "metadata.google.internal"
    │       - "169.254.169.254"
    │
    ├── 3. Resolve hostname to IP via DNS
    │   └── Use socket.getaddrinfo() to resolve
    │
    ├── 4. Check resolved IP against blocked ranges
    │   └── Reject if IP is in any of:
    │       - 10.0.0.0/8 (private)
    │       - 172.16.0.0/12 (private)
    │       - 192.168.0.0/16 (private)
    │       - 169.254.0.0/16 (link-local / metadata)
    │       - 127.0.0.0/8 (loopback)
    │       - ::1 (IPv6 loopback)
    │       - fc00::/7 (IPv6 private)
    │       - 0.0.0.0/8 (unspecified)
    │
    ├── 5. Return validated URL
    │
    └── On any rejection → raise SSRFError
```

**Key implementation — DNS resolution before fetch:**

The critical security property is resolving the hostname to an IP address **before** making the HTTP request, then validating the resolved IP. This prevents DNS rebinding attacks where a hostname resolves to a public IP for validation but a private IP for the actual request.

```python
import ipaddress
import socket
from urllib.parse import urlparse
from exceptions import SSRFError


# Blocked IP networks
BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

# Blocked hostnames (case-insensitive)
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.google.internal.",
}


def validate_url(url: str) -> str:
    """Validate a URL is safe to fetch (not targeting internal resources).

    Args:
        url: User-supplied URL string.

    Returns:
        The validated URL (unchanged if safe).

    Raises:
        SSRFError: If URL targets a private/reserved IP, metadata endpoint,
                   or uses a non-HTTPS scheme.
    """
    parsed = urlparse(url)

    # 1. Scheme check
    if parsed.scheme != "https":
        raise SSRFError(
            f"Only HTTPS URLs are allowed, got {parsed.scheme}://",
            url=url,
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname", url=url)

    # 2. Hostname blocklist
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise SSRFError(
            "URL targets a blocked metadata endpoint",
            url=url,
        )

    # 3. DNS resolution
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise SSRFError(f"Could not resolve hostname: {hostname}", url=url)

    # 4. IP validation
    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in BLOCKED_NETWORKS:
            if ip in network:
                raise SSRFError(
                    f"URL resolves to a private/reserved IP address",
                    url=url,
                    resolved_ip=str(ip),
                )

    return url
```

**Redirect handling:**

When httpx follows redirects (max 5 hops), each redirect target must also be validated. This is handled in `utils/url_fetch.py` (Phase 5) by hooking into httpx's event system or manually following redirects with validation at each hop.

---

### 2. `security/svg_sanitizer.py`

**Purpose:** Strip dangerous content from SVG files before optimization. SVGs are XML documents that can contain JavaScript, external entity references, and other attack vectors.

**Sanitization checklist:**

| Threat | Element/Attribute | Action |
|--------|------------------|--------|
| XSS via script | `<script>` tags | Remove entirely |
| XSS via event handlers | `onload`, `onclick`, `onerror`, etc. | Remove attribute |
| XXE (XML External Entity) | `<!DOCTYPE>` with entity definitions | Use defusedxml (blocks by default) |
| Foreign content injection | `<foreignObject>` | Remove entirely |
| External resource loading | `<use href="http://...">` | Remove external references |
| Data URI exploitation | `href="data:text/html,..."` | Remove `data:` URIs in href attributes |
| CSS-based attacks | `<style>` with `@import url(...)` | Strip `@import` rules |

**Implementation:**

```python
import re
from defusedxml import ElementTree as ET


# Event handler attributes to strip (case-insensitive)
EVENT_HANDLERS = {
    "onload", "onerror", "onclick", "onmouseover", "onmouseout",
    "onmousedown", "onmouseup", "onmousemove", "onfocus", "onblur",
    "onchange", "onsubmit", "onreset", "onselect", "onkeydown",
    "onkeypress", "onkeyup", "onabort", "onactivate", "onbegin",
    "onend", "onrepeat", "onunload", "onscroll", "onresize",
    "oninput", "onanimationstart", "onanimationend", "onanimationiteration",
    "ontransitionend",
}

# Elements to remove entirely
DANGEROUS_ELEMENTS = {
    "script",
    "foreignObject",
    "foreignobject",  # Case variant
}


def sanitize_svg(data: bytes) -> bytes:
    """Sanitize SVG content to remove security threats.

    Uses defusedxml to prevent XXE attacks during parsing.
    Strips all script tags, event handlers, external references,
    and data: URIs.

    Args:
        data: Raw SVG bytes (UTF-8 encoded XML).

    Returns:
        Sanitized SVG bytes.

    Raises:
        OptimizationError: If SVG is malformed XML.
    """
    # defusedxml blocks XXE, billion laughs, etc. by default
    tree = ET.fromstring(data)

    _strip_dangerous_elements(tree)
    _strip_event_handlers(tree)
    _strip_dangerous_hrefs(tree)
    _strip_css_imports(tree)

    return ET.tostring(tree, encoding="unicode").encode("utf-8")


def _strip_dangerous_elements(root):
    """Remove <script>, <foreignObject>, and similar elements."""
    for element in root.iter():
        local_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if local_name.lower() in DANGEROUS_ELEMENTS:
            parent = _find_parent(root, element)
            if parent is not None:
                parent.remove(element)


def _strip_event_handlers(root):
    """Remove on* event handler attributes from all elements."""
    for element in root.iter():
        attrs_to_remove = [
            attr for attr in element.attrib
            if attr.lower().split("}")[-1] in EVENT_HANDLERS
        ]
        for attr in attrs_to_remove:
            del element.attrib[attr]


def _strip_dangerous_hrefs(root):
    """Remove data: URIs and external references in href/xlink:href."""
    for element in root.iter():
        for attr_name in list(element.attrib):
            if "href" in attr_name.lower():
                value = element.attrib[attr_name].strip()
                if value.startswith("data:") and "text/html" in value:
                    del element.attrib[attr_name]
                elif _is_external_url(value) and element.tag.endswith("use"):
                    del element.attrib[attr_name]


def _strip_css_imports(root):
    """Strip @import rules from <style> elements."""
    for element in root.iter():
        local_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag
        if local_name.lower() == "style" and element.text:
            element.text = re.sub(
                r'@import\s+url\s*\([^)]*\)\s*;?',
                '',
                element.text,
            )
```

---

### 3. `security/file_validation.py`

**Purpose:** Validate uploaded files before processing. First line of defense.

```python
from config import settings
from exceptions import FileTooLargeError, UnsupportedFormatError
from utils.format_detect import detect_format, ImageFormat


def validate_file(data: bytes) -> ImageFormat:
    """Validate file size and format.

    Args:
        data: Raw file bytes.

    Returns:
        Detected ImageFormat.

    Raises:
        FileTooLargeError: If file exceeds MAX_FILE_SIZE_MB.
        UnsupportedFormatError: If magic bytes don't match any known format.
    """
    # Size check
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds limit of {settings.max_file_size_mb} MB",
            file_size=len(data),
            limit=settings.max_file_size_bytes,
        )

    # Format detection (delegates to magic byte analysis)
    fmt = detect_format(data)

    return fmt
```

---

### 4. `security/rate_limiter.py`

**Purpose:** Redis-backed sliding window rate limiting. Shared across all Cloud Run instances via VPC Redis.

**Algorithm — Sliding Window Counter:**

```
For each request:
    1. Key = "rate:{client_ip}:{current_minute}"
    2. INCR key
    3. EXPIRE key 120 (keep two minutes for sliding window)
    4. Also read previous minute's key
    5. Compute weighted count:
       weighted = prev_count * (1 - elapsed_fraction) + current_count
    6. If weighted > limit → reject with 429
```

The sliding window counter avoids the burst-at-boundary problem of fixed windows while being simpler than a true sliding log.

```python
import time
from config import settings
from exceptions import RateLimitError

# redis client initialized lazily
_redis = None


async def get_redis():
    """Get or create async Redis connection."""
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis


async def check_rate_limit(client_ip: str, is_authenticated: bool) -> None:
    """Check if the request should be rate-limited.

    Args:
        client_ip: Client IP address (from X-Forwarded-For or direct).
        is_authenticated: Whether the request has a valid API key.

    Raises:
        RateLimitError: If rate limit exceeded (429).
    """
    # Authenticated requests bypass rate limiting by default
    if is_authenticated and not settings.rate_limit_auth_enabled:
        return

    limit = (
        settings.rate_limit_auth_rpm
        if is_authenticated
        else settings.rate_limit_public_rpm
    )

    if limit == 0:
        return  # Unlimited

    r = await get_redis()
    now = time.time()
    current_minute = int(now // 60)
    elapsed_fraction = (now % 60) / 60

    current_key = f"rate:{client_ip}:{current_minute}"
    prev_key = f"rate:{client_ip}:{current_minute - 1}"

    pipe = r.pipeline()
    pipe.incr(current_key)
    pipe.expire(current_key, 120)
    pipe.get(prev_key)
    results = await pipe.execute()

    current_count = results[0]
    prev_count = int(results[2] or 0)

    # Sliding window weighted count
    weighted = prev_count * (1 - elapsed_fraction) + current_count

    if weighted > limit:
        raise RateLimitError(
            "Rate limit exceeded",
            retry_after=int(60 - now % 60),
            limit=limit,
        )


async def check_burst_limit(client_ip: str) -> None:
    """Check burst limit (requests in a short window).

    Uses a 10-second window for burst detection.

    Raises:
        RateLimitError: If burst limit exceeded.
    """
    if settings.rate_limit_public_burst == 0:
        return

    r = await get_redis()
    now = time.time()
    window = int(now // 10)
    key = f"burst:{client_ip}:{window}"

    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 20)
    results = await pipe.execute()

    if results[0] > settings.rate_limit_public_burst:
        raise RateLimitError(
            "Burst rate limit exceeded",
            retry_after=int(10 - now % 10),
            limit=settings.rate_limit_public_burst,
        )
```

**Redis failure behavior:**

If Redis is unavailable, rate limiting fails **open** (allows requests through). This prevents Redis outages from taking down the entire service. A warning is logged.

```python
async def safe_check_rate_limit(client_ip: str, is_authenticated: bool) -> None:
    """Rate limit check with fail-open on Redis errors."""
    if not settings.redis_url:
        return  # Rate limiting disabled (no Redis configured)
    try:
        await check_rate_limit(client_ip, is_authenticated)
        await check_burst_limit(client_ip)
    except RateLimitError:
        raise  # Propagate actual rate limit hits
    except Exception:
        # Redis unavailable — fail open, log warning
        logger.warning("Rate limiter unavailable — allowing request")
```

---

### 5. `security/auth.py`

**Purpose:** API key authentication. Single key stored as environment variable (sourced from GCP Secret Manager at deploy time).

```python
from fastapi import Request
from config import settings
from exceptions import AuthenticationError


def authenticate(request: Request) -> bool:
    """Check if request has a valid API key.

    Args:
        request: FastAPI request object.

    Returns:
        True if authenticated, False if no auth header present.

    Raises:
        AuthenticationError: If auth header present but key is invalid.
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        return False  # No auth — treated as public request

    if not auth_header.startswith("Bearer "):
        raise AuthenticationError("Invalid Authorization header format. Expected: Bearer <key>")

    provided_key = auth_header[7:]  # Strip "Bearer "

    if not settings.api_key:
        # No API key configured — accept all authenticated requests
        # (development mode)
        return True

    if provided_key != settings.api_key:
        raise AuthenticationError("Invalid API key")

    return True
```

**Key design decisions:**
- No auth header → public request (rate limited)
- Auth header with invalid key → 401 error (not silently treated as public)
- No API key configured → development mode, all Bearer tokens accepted
- Single key for now (Shopify App backend). Multi-key support is a future enhancement

---

### 6. `middleware.py`

**Purpose:** Wire all security middleware into the FastAPI request lifecycle. Updates the Phase 1 stub with actual auth + rate limiting + request ID logic.

```python
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from security.auth import authenticate
from security.rate_limiter import safe_check_rate_limit
from exceptions import PareError


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined middleware for auth, rate limiting, and request ID.

    Order of operations per request:
    1. Inject request ID (UUID)
    2. Authenticate (check Bearer token)
    3. Rate limit check (skip for authenticated if configured)
    4. Process request
    5. Add X-Request-ID to response
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            # 2. Authentication
            is_authenticated = authenticate(request)
            request.state.is_authenticated = is_authenticated

            # 3. Rate limiting
            client_ip = request.headers.get(
                "X-Forwarded-For", request.client.host
            ).split(",")[0].strip()
            await safe_check_rate_limit(client_ip, is_authenticated)

            # 4. Process request
            response = await call_next(request)

        except PareError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.error_code,
                    "message": exc.message,
                    **exc.details,
                },
            )

        # 5. Request ID header
        response.headers["X-Request-ID"] = request_id
        return response
```

**CORS configuration** remains in `main.py` (Phase 1) using FastAPI's built-in CORSMiddleware. The SecurityMiddleware is registered as additional middleware:

```python
# In main.py
from middleware import SecurityMiddleware
app.add_middleware(SecurityMiddleware)
```

---

## Security Threat Model

| Threat | Vector | Protection | Phase |
|--------|--------|------------|-------|
| SSRF | URL input targeting `169.254.169.254` | DNS resolve → IP validation | 4 |
| SSRF via redirect | Redirect chain bouncing to internal IP | Validate each redirect hop | 4+5 |
| DNS rebinding | Hostname resolves to public then private IP | Resolve once, use resolved IP | 4 |
| SVG XSS | `<script>` tags in SVG | Strip all `<script>` elements | 4 |
| SVG event XSS | `onload="alert(1)"` | Strip all `on*` attributes | 4 |
| XXE | `<!DOCTYPE>` with external entities | defusedxml blocks by default | 4 |
| SVG foreign content | `<foreignObject>` embedding HTML | Strip `<foreignObject>` | 4 |
| Resource exhaustion | Large file uploads | 32 MB limit, validated before processing | 4 |
| Brute force | Repeated requests | Sliding window rate limiting | 4 |
| Unauthorized access | Missing/invalid API key | Bearer token authentication | 4 |
| Format confusion | Wrong Content-Type header | Magic byte detection, ignore headers | 2+4 |

---

## Environment Variables Introduced

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `""` (disabled) | Redis connection URL for rate limiting |
| `RATE_LIMIT_PUBLIC_RPM` | `60` | Requests/minute for unauthenticated |
| `RATE_LIMIT_PUBLIC_BURST` | `10` | Burst limit (per 10-second window) |
| `RATE_LIMIT_AUTH_ENABLED` | `false` | Whether to rate limit authenticated requests |
| `RATE_LIMIT_AUTH_RPM` | `0` | Requests/minute for authenticated (0=unlimited) |
| `API_KEY` | `""` | API key (from GCP Secret Manager) |

---

## Verification Steps

### Manual verification

```bash
# Test SSRF protection — should be rejected
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "http://169.254.169.254/latest/meta-data/"}'
# Expected: 422 with "ssrf_blocked" error

curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "https://metadata.google.internal/"}'
# Expected: 422 with "ssrf_blocked" error

# Test non-HTTPS rejection
curl -X POST http://localhost:8080/optimize \
  -H "Content-Type: application/json" \
  -d '{"url": "http://example.com/image.png"}'
# Expected: 422 with "ssrf_blocked" error (HTTP not allowed)

# Test SVG sanitization — upload malicious SVG
echo '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><rect onload="alert(2)" width="100" height="100"/></svg>' > malicious.svg
curl -X POST http://localhost:8080/optimize \
  -F "file=@malicious.svg" -o sanitized.svg
cat sanitized.svg  # Should have no <script> or onload

# Test rate limiting
for i in $(seq 1 70); do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST http://localhost:8080/optimize -F "file=@sample.png"
done
# After ~60 requests, should start getting 429

# Test auth
curl -X POST http://localhost:8080/optimize \
  -H "Authorization: Bearer wrong-key" \
  -F "file=@sample.png"
# Expected: 401 Unauthorized

curl -X POST http://localhost:8080/optimize \
  -H "Authorization: Bearer correct-key" \
  -F "file=@sample.png" -o optimized.png
# Expected: 200 OK
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_ssrf_private_ip_10` | Reject URLs resolving to 10.x.x.x |
| `test_ssrf_private_ip_172` | Reject URLs resolving to 172.16.x.x |
| `test_ssrf_private_ip_192` | Reject URLs resolving to 192.168.x.x |
| `test_ssrf_link_local` | Reject 169.254.x.x (metadata server) |
| `test_ssrf_localhost` | Reject 127.0.0.1 / ::1 |
| `test_ssrf_metadata_hostname` | Reject metadata.google.internal |
| `test_ssrf_http_scheme` | Reject http:// (HTTPS only) |
| `test_ssrf_valid_https` | Allow valid HTTPS URLs to public IPs |
| `test_svg_strip_script` | `<script>` tags removed |
| `test_svg_strip_event_handlers` | `onload`, `onclick` etc. removed |
| `test_svg_xxe_blocked` | XXE entity expansion blocked by defusedxml |
| `test_svg_strip_foreign_object` | `<foreignObject>` removed |
| `test_svg_strip_data_uri` | `data:text/html` in href removed |
| `test_svg_preserve_content` | Non-malicious SVG elements preserved |
| `test_file_size_limit` | >32MB rejected with 413 |
| `test_file_format_validation` | Unknown magic bytes rejected with 415 |
| `test_rate_limit_public` | 61st request in a minute returns 429 |
| `test_rate_limit_burst` | 11th request in 10 seconds returns 429 |
| `test_rate_limit_auth_bypass` | Authenticated requests not rate limited |
| `test_rate_limit_redis_down` | Redis failure → requests allowed (fail-open) |
| `test_auth_valid_key` | Valid Bearer token accepted |
| `test_auth_invalid_key` | Invalid token returns 401 |
| `test_auth_no_header` | No auth header → public request (not error) |
| `test_auth_malformed_header` | Non-Bearer format returns 401 |
