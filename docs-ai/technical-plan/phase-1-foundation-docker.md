# Phase 1 — Foundation & Docker

## Objectives

- Establish the project skeleton with all directories and package init files
- Build a multi-stage Docker image with all compression binaries baked in
- Define configuration management via environment variables (Pydantic BaseSettings)
- Create Pydantic request/response schemas matching the PRD API contract
- Define custom exception classes with HTTP status code mapping
- Verify all compression tools are accessible inside the container

## Deliverables

- Working Docker image with all CLI tools (pngquant, cjpeg, jpegtran, gifsicle, cwebp) and Python libraries (Pillow, pyoxipng, scour, pillow-heif)
- `main.py` with a minimal FastAPI app that starts and responds to `/health`
- `config.py` with all environment variable bindings
- `schemas.py` with all request/response models
- `exceptions.py` with all custom exception classes

## Dependencies

- None (this is the foundation phase)

---

## Files to Create

### 1. `Dockerfile`

**Purpose:** Multi-stage build — compile MozJPEG from source in stage 1, assemble the slim production image in stage 2.

```dockerfile
# ---- Stage 1: Build MozJPEG from source ----
FROM debian:bookworm-slim AS mozjpeg-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake nasm build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG MOZJPEG_VERSION=4.1.5
RUN curl -L https://github.com/nickterhaar/mozjpeg/archive/refs/tags/v${MOZJPEG_VERSION}.tar.gz \
    | tar xz \
    && cd mozjpeg-${MOZJPEG_VERSION} \
    && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/mozjpeg \
             -DENABLE_SHARED=OFF \
             -DENABLE_STATIC=ON \
             -DPNG_SUPPORTED=OFF \
             .. \
    && make -j$(nproc) \
    && make install

# ---- Stage 2: Production image ----
FROM python:3.12-slim

# Copy MozJPEG binaries
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/cjpeg /usr/local/bin/cjpeg
COPY --from=mozjpeg-builder /opt/mozjpeg/bin/jpegtran /usr/local/bin/jpegtran

# Install system compression tools + codec libraries
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

# Copy application
COPY . /app
WORKDIR /app

# Cloud Run sets $PORT; Uvicorn workers configurable
CMD ["sh", "-c", \
     "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080} --workers ${WORKERS:-4} --timeout-graceful-shutdown ${GRACEFUL_SHUTDOWN_TIMEOUT:-30}"]
```

**Key decisions:**
- MozJPEG is pinned to a specific version tag for reproducibility
- Static linking (`ENABLE_SHARED=OFF`) for MozJPEG to avoid shared library issues
- `--no-install-recommends` minimizes image size
- `libheif-dev` + `libde265-dev` + `libaom-dev` provide HEIC/AVIF codec support for pillow-heif
- `webp` package provides the `cwebp` binary
- Graceful shutdown timeout is configurable via env var

---

### 2. `requirements.txt`

```
fastapi>=0.109.0,<1.0.0
uvicorn[standard]>=0.27.0,<1.0.0
Pillow>=10.2.0,<11.0.0
pillow-heif>=0.14.0,<1.0.0
pyoxipng>=9.0.0,<10.0.0
scour>=0.38.0,<1.0.0
python-multipart>=0.0.6,<1.0.0
httpx>=0.27.0,<1.0.0
google-cloud-storage>=2.14.0,<3.0.0
defusedxml>=0.7.0,<1.0.0
redis[hiredis]>=5.0.0,<6.0.0
pydantic-settings>=2.1.0,<3.0.0
```

**Pinning strategy:** Lower bound for features, upper bound at next major version to prevent breaking changes.

---

### 3. `config.py`

**Purpose:** Single source of truth for all configuration. Uses `pydantic-settings` to load from environment variables with sensible defaults.

```python
# config.py
import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- Server ---
    port: int = 8080
    workers: int = 4
    graceful_shutdown_timeout: int = 30

    # --- File Limits ---
    max_file_size_mb: int = 32
    max_file_size_bytes: int = 0  # Computed in validator

    # --- Optimization Defaults ---
    default_quality: int = 80
    tool_timeout_seconds: int = 60

    # --- Concurrency ---
    compression_semaphore_size: int = 0  # 0 = use CPU count
    max_queue_depth: int = 0  # 0 = 2 * CPU count

    # --- Security ---
    redis_url: str = ""
    rate_limit_public_rpm: int = 60
    rate_limit_public_burst: int = 10
    rate_limit_auth_enabled: bool = False
    rate_limit_auth_rpm: int = 0
    api_key: str = ""
    allowed_origins: str = "*"

    # --- URL Fetching ---
    url_fetch_timeout: int = 30
    url_fetch_max_redirects: int = 5

    # --- Logging ---
    log_level: str = "ERROR"

    class Config:
        env_prefix = ""
        case_sensitive = False

    def model_post_init(self, __context) -> None:
        if self.max_file_size_bytes == 0:
            self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        if self.compression_semaphore_size == 0:
            self.compression_semaphore_size = os.cpu_count() or 4
        if self.max_queue_depth == 0:
            self.max_queue_depth = 2 * self.compression_semaphore_size


settings = Settings()
```

**Key classes/functions:**
- `Settings` — Pydantic BaseSettings subclass, auto-loads from env vars
- `settings` — Module-level singleton, imported throughout the app

---

### 4. `schemas.py`

**Purpose:** Pydantic models for all request/response types. These enforce the API contract from the PRD.

```python
# schemas.py
from typing import Optional
from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    """Optimization parameters (all optional with defaults)."""
    quality: int = Field(default=80, ge=1, le=100)
    strip_metadata: bool = True
    progressive_jpeg: bool = False
    png_lossy: bool = True


class StorageConfig(BaseModel):
    """Storage upload configuration."""
    provider: str = Field(..., pattern="^(gcs)$")  # Extensible: s3, azure later
    bucket: str
    path: str
    project: Optional[str] = None
    public: bool = False


class OptimizeRequest(BaseModel):
    """JSON body for URL-based optimization."""
    url: str
    optimization: OptimizationConfig = OptimizationConfig()
    storage: Optional[StorageConfig] = None


class OptimizeResult(BaseModel):
    """Internal result passed between optimizer and response formatter."""
    success: bool
    original_size: int
    optimized_size: int
    reduction_percent: float
    format: str
    method: str
    optimized_bytes: bytes = b""
    message: Optional[str] = None


class StorageResult(BaseModel):
    """Storage upload result included in JSON responses."""
    provider: str
    url: str
    public_url: Optional[str] = None


class OptimizeResponse(BaseModel):
    """JSON response when storage is configured."""
    success: bool
    original_size: int
    optimized_size: int
    reduction_percent: float
    format: str
    method: str
    storage: Optional[StorageResult] = None
    message: Optional[str] = None


class EstimateResponse(BaseModel):
    """Response from the /estimate endpoint."""
    original_size: int
    original_format: str
    dimensions: dict  # {"width": int, "height": int}
    color_type: Optional[str] = None
    bit_depth: Optional[int] = None
    estimated_optimized_size: int
    estimated_reduction_percent: float
    optimization_potential: str  # "high", "medium", "low"
    method: str
    already_optimized: bool
    confidence: str  # "high", "medium", "low"


class ErrorResponse(BaseModel):
    """Standard error response."""
    success: bool = False
    error: str
    message: str
    original_size: Optional[int] = None
    format: Optional[str] = None


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "ok"
    tools: dict  # {"pngquant": True, "cjpeg": True, ...}
    version: str
```

**Design notes:**
- `OptimizeRequest` handles JSON/URL mode; file upload mode parses `options` form field into `OptimizationConfig` + `StorageConfig` separately in the router
- `OptimizeResult` is internal — never serialized to the client directly
- `StorageConfig.provider` uses regex validation to restrict to supported providers

---

### 5. `exceptions.py`

**Purpose:** Custom exceptions with HTTP status codes. The FastAPI exception handler maps these to structured JSON responses.

```python
# exceptions.py
from fastapi import HTTPException


class PareError(Exception):
    """Base exception for all Pare errors."""
    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, **kwargs):
        self.message = message
        self.details = kwargs
        super().__init__(message)


class FileTooLargeError(PareError):
    """File exceeds maximum allowed size."""
    status_code = 413
    error_code = "file_too_large"


class UnsupportedFormatError(PareError):
    """File format not recognized via magic bytes."""
    status_code = 415
    error_code = "unsupported_format"


class OptimizationError(PareError):
    """Optimization failed (tool crash, larger output, etc.)."""
    status_code = 422
    error_code = "optimization_failed"


class SSRFError(PareError):
    """URL targets a private/reserved IP range."""
    status_code = 422
    error_code = "ssrf_blocked"


class URLFetchError(PareError):
    """Failed to fetch image from URL."""
    status_code = 422
    error_code = "url_fetch_failed"


class ToolTimeoutError(PareError):
    """Compression tool exceeded timeout."""
    status_code = 500
    error_code = "tool_timeout"


class RateLimitError(PareError):
    """Rate limit exceeded."""
    status_code = 429
    error_code = "rate_limit_exceeded"


class AuthenticationError(PareError):
    """Invalid or missing API key."""
    status_code = 401
    error_code = "unauthorized"


class BackpressureError(PareError):
    """Compression queue is full."""
    status_code = 503
    error_code = "service_overloaded"
```

**Exception handler registration (in `main.py`):**

```python
@app.exception_handler(PareError)
async def pare_error_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.error_code,
            "message": exc.message,
            **exc.details,
        },
    )
```

---

### 6. `main.py`

**Purpose:** FastAPI application factory. Mounts routers, registers exception handlers, runs startup/shutdown hooks.

```python
# main.py
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from exceptions import PareError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify tools are available. Shutdown: drain connections."""
    # Startup — verify all compression binaries exist
    # (Phase 2 will populate this with actual tool checks)
    yield
    # Shutdown — Uvicorn handles graceful drain via --timeout-graceful-shutdown


app = FastAPI(
    title="Pare",
    description="Image Optimizer Service",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
origins = [o.strip() for o in settings.allowed_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=[
        "X-Original-Size",
        "X-Optimized-Size",
        "X-Reduction-Percent",
        "X-Original-Format",
        "X-Optimization-Method",
        "X-Request-ID",
    ],
)


@app.exception_handler(PareError)
async def pare_error_handler(request: Request, exc: PareError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.error_code,
            "message": exc.message,
            **exc.details,
        },
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Router mounting (stubs — implemented in Phase 3)
# from routers import optimize, estimate, health
# app.include_router(optimize.router)
# app.include_router(estimate.router)
# app.include_router(health.router)
```

---

### 7. `docker-compose.yml`

**Purpose:** Local development environment with Redis for rate limiting.

```yaml
version: "3.8"

services:
  pare:
    build: .
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - WORKERS=2
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=WARNING
      - ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

---

## Directory Structure After Phase 1

```
pare/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── main.py
├── config.py
├── schemas.py
├── exceptions.py
├── routers/
│   └── __init__.py
├── optimizers/
│   └── __init__.py
├── estimation/
│   └── __init__.py
├── security/
│   └── __init__.py
├── storage/
│   └── __init__.py
├── utils/
│   └── __init__.py
└── tests/
    └── __init__.py
```

All `__init__.py` files are empty placeholders created in this phase.

---

## Environment Variables Introduced

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `PORT` | `8080` | No | HTTP port (Cloud Run sets automatically) |
| `WORKERS` | `4` | No | Uvicorn worker processes |
| `GRACEFUL_SHUTDOWN_TIMEOUT` | `30` | No | Seconds to drain on SIGTERM |
| `MAX_FILE_SIZE_MB` | `32` | No | Upload size limit |
| `DEFAULT_QUALITY` | `80` | No | Default optimization quality |
| `ALLOWED_ORIGINS` | `*` | No | CORS allowed origins |
| `LOG_LEVEL` | `ERROR` | No | Logging verbosity |

---

## Verification Steps

### Manual verification

```bash
# Build the Docker image
docker build -t pare:dev .

# Verify all CLI tools are present
docker run --rm pare:dev which pngquant cjpeg jpegtran gifsicle cwebp

# Verify Python libraries import
docker run --rm pare:dev python -c "
import PIL; print(f'Pillow {PIL.__version__}')
import pillow_heif; print('pillow-heif OK')
import pyoxipng; print('pyoxipng OK')
import scour; print('scour OK')
import fastapi; print(f'FastAPI {fastapi.__version__}')
import httpx; print('httpx OK')
import defusedxml; print('defusedxml OK')
import redis; print('redis OK')
"

# Start the server and verify it responds
docker compose up -d
curl -s http://localhost:8080/docs | head -5  # FastAPI auto-docs should load

# Verify config loads from env vars
docker run --rm -e DEFAULT_QUALITY=90 -e MAX_FILE_SIZE_MB=16 pare:dev python -c "
from config import settings
assert settings.default_quality == 90
assert settings.max_file_size_mb == 16
print('Config OK')
"
```

### Automated test descriptions

| Test | What it verifies |
|------|-----------------|
| `test_config_defaults` | All Settings fields have correct defaults |
| `test_config_env_override` | Environment variables override defaults |
| `test_config_computed_fields` | `max_file_size_bytes` and semaphore size computed correctly |
| `test_schemas_optimize_request` | OptimizeRequest validates URL, quality range, storage provider |
| `test_schemas_reject_invalid` | Quality < 1 or > 100 rejected, unknown provider rejected |
| `test_exceptions_status_codes` | Each exception maps to correct HTTP status |
| `test_exception_handler` | PareError subclasses produce correct JSON response structure |
| `test_request_id_middleware` | Every response includes `X-Request-ID` header |
