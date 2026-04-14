# Memory & CPU Optimization Brainstorm

**Date:** 2026-04-14
**Status:** Approved
**Approach:** Comprehensive

## Problem Statement

Pare's image compression API (8GB memory, 8 cores on Cloud Run) cannot reliably optimize a 10MB file. The container is OOM-killed within ~5 seconds of receiving the request. The service should handle files up to the 32MB limit comfortably.

**Root cause:** Memory amplification. A single 10MB image consumes 86-184MB of RAM during optimization due to multiple simultaneous copies of image data. With 4 Uvicorn workers x 8 concurrent slots each = up to 32 simultaneous optimizations, total memory demand easily exceeds 8GB.

## What We're Building

A comprehensive memory and CPU optimization pass across the entire codebase, targeting three layers:

1. **Per-request memory reduction** (~60-70% less per optimization)
2. **System-level concurrency guards** (prevent OOM under load)
3. **CPU efficiency** (skip unnecessary work)

### Target: A 10MB TIFF should use ~60MB peak (down from ~184MB), and the system should comfortably handle 32MB files with moderate concurrency.

## Audit Findings

### Memory Amplification Per Format (Single Request, 10MB Input)

| Format | Current Peak | Primary Cause | Copies at Peak |
|--------|-------------|---------------|----------------|
| TIFF   | ~184MB | 3x `img.copy()` for parallel compression | 5 |
| PNG    | ~105MB | pngquant + oxipng parallel + second oxipng pass | 4 |
| JPEG   | ~86MB | Pillow decode + encode + jpegtran stdin/stdout | 4 |
| JPEG (cjpeg) | ~120MB+ | BMP intermediate (~36MB) piped 6x in binary search | 6+ |
| WebP   | ~80MB | Double `Image.open()` + temp files | 4 |
| AVIF/HEIC/JXL | ~80MB | `img.copy()` for strip + img for reencode | 3 |
| BMP    | ~60MB | `getdata()` iteration + palette copies | 3 |
| GIF    | ~20MB | Minimal — just stdin/stdout piping | 2 |
| SVG    | ~5MB | Text processing | 1 |

### Concurrency Multiplication

- **4 Uvicorn workers** x **8 semaphore slots each** = 32 possible concurrent optimizations
- Each worker is a separate process with its own memory space
- CompressionGate semaphore is per-process, not global
- **Estimate endpoint has NO semaphore** — unbounded concurrent PIL decodes

### Missing Safeguards

- No `Image.MAX_IMAGE_PIXELS` limit (decompression bomb vulnerability)
- No decompressed size validation
- No pixel count check before full decode
- No memory profiling or monitoring
- Estimate endpoint unbounded concurrency
- BytesIO `.getvalue()` creates a copy every time

## Key Decisions

### 1. Single Uvicorn Worker
Switch from 4 workers to 1. Cloud Run handles horizontal scaling (multiple container instances). A single process gets all 8GB, eliminating the per-process memory fragmentation. This is the single biggest win.

**Why:** 4 workers = 4 separate Python processes = 4x base memory overhead + no memory sharing between them. Cloud Run's auto-scaling already handles concurrency by spinning up more containers.

### 2. Eliminate `img.copy()` — Sequential for Large Images
Currently TIFF creates 3 copies, AVIF/HEIC/JXL creates 2 copies for parallel threads. For images above a pixel threshold (5 megapixels), run compression methods sequentially on the shared PIL Image. For small images, keep parallel execution (memory is negligible).

**Why:** `img.copy()` duplicates the entire decompressed pixel buffer. For a 12MP RGB image, each copy is ~36MB. Eliminating 2 copies saves ~72MB per TIFF request. Sequential execution is slightly slower but dramatically reduces peak memory.

### 3. Pixel Count Limit
Add `Image.MAX_IMAGE_PIXELS = 100_000_000` (100 megapixels). Reject images exceeding this before decompression. A 100MP RGBA image decompresses to ~400MB — this is the maximum we should accept.

**Why:** The 32MB file size limit does NOT protect against decompression bombs. A carefully crafted 32MB PNG could decompress to >1GB of pixel data.

### 4. Memory-Aware Concurrency
Estimate per-request memory before decompression (based on format + compressed file size) and adjust the semaphore. Don't allow total estimated memory to exceed a budget (e.g., 6GB of 8GB, leaving 2GB for runtime).

**Why:** The current semaphore counts slots (8) but doesn't account for the fact that a 32MB TIFF needs 10x more memory than a 100KB JPEG.

### 5. Guard Estimate Endpoint
Add a separate, lighter semaphore for `/estimate`. Currently unlimited concurrent estimates can each do full `img.load()` with no backpressure.

**Why:** Documented in code comments ("does not acquire semaphore slot") but this creates an unbounded memory path. Even sample-based estimation decompresses the full image for small files.

### 6. BytesIO Zero-Copy
Replace `buf.getvalue()` (which copies the buffer) with `bytes(buf.getbuffer())` or return the BytesIO directly where possible. This eliminates one copy per Pillow save operation.

**Why:** Every `buf.getvalue()` call creates a new `bytes` object that is a copy of the BytesIO internal buffer. For a 10MB output, this means 10MB exists in the BytesIO buffer AND 10MB in the returned bytes.

### 7. Decompressed Size Validation
Before full `img.load()`, peek at image dimensions (Pillow's lazy loading gives us `img.size` without decompressing) and reject if `width * height * bytes_per_pixel` exceeds a threshold (e.g., 512MB).

**Why:** Pillow opens images lazily — `Image.open()` reads headers but doesn't decompress. We can check dimensions before committing to the full decompression.

### 8. Skip Redundant Work for Lossless Presets
When `quality >= 70` (lossless-only), some optimizers still run lossy methods that will always be skipped. Early-exit or skip these branches.

**Why:** Pure CPU savings. No point running lossy compression paths when the quality setting guarantees they won't be used.

## Open Questions

*All resolved during brainstorm discussion.*

## Scope Boundaries

### In Scope
- Memory reduction in all optimizers
- Concurrency model changes (workers, semaphore)
- Pixel/decompression limits
- BytesIO copy elimination
- Estimate endpoint guard
- CPU skip optimizations
- Cloud Run config (workers env var)

### Out of Scope
- Streaming HTTP responses (changes API semantics)
- memoryview/buffer protocol changes (Pillow compatibility issues)
- Temp file fallback for very large images
- PIL memory-mapping
- Per-format memory budgets (too granular for first pass)
- New format support
- API changes

## Expected Impact

| Metric | Before | After |
|--------|--------|-------|
| Peak memory per 10MB TIFF | ~184MB | ~60MB |
| Peak memory per 10MB JPEG | ~86MB | ~45MB |
| Peak memory per 10MB PNG | ~105MB | ~55MB |
| Max concurrent (8GB, 10MB files) | 3-4 (OOM) | 8+ (stable) |
| Workers per container | 4 | 1 |
| 32MB file support | Fails | Works |
| Decompression bomb protection | None | 100MP limit |
| Estimate endpoint concurrency | Unbounded | Semaphore-limited |
