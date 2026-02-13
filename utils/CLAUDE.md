# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Shared utilities used across the codebase.

## Modules

- **`format_detect.py`**: Magic-byte format detection. The `ImageFormat` enum and `detect_format()` function are the single source of truth for supported formats (12 formats: PNG, APNG, JPEG, WebP, GIF, SVG, SVGZ, AVIF, HEIC, TIFF, BMP, JXL). Never trust file extensions or Content-Type headers. AVIF/HEIC/JXL are detected via ISO BMFF `ftyp` box brands; JXL also has a bare codestream signature (`\xFF\x0A`).

- **`subprocess_runner.py`**: `run_tool()` pipes bytes through CLI tools (pngquant, jpegtran, gifsicle, cwebp, cjxl/djxl) via stdin/stdout. No temp files. Handles timeouts and non-zero exit codes. Use `allowed_exit_codes` for expected failures (e.g., pngquant exit 99).

- **`concurrency.py`**: `CompressionGate` singleton — semaphore (CPU count) + queue depth cap (2x semaphore). Returns 503 immediately when full to prevent OOM from queued 32MB payloads. Only `/optimize` acquires a slot; `/estimate` does not.

- **`metadata.py`**: `strip_metadata_selective()` preserves EXIF orientation and ICC profiles while stripping GPS, camera info, XMP, thumbnails, and comments. Format-specific implementations for JPEG (Pillow re-save), PNG (chunk filtering), TIFF (Pillow).

- **`url_fetch.py`**: `fetch_image()` fetches images from user-supplied URLs. Follows redirects manually with SSRF validation at each hop. Streams response and checks Content-Length for early size rejection.

- **`logging.py`**: Structured JSON logging for Google Cloud Logging. `setup_logging()` called once at app startup. Use `get_logger(name)` to get child loggers under the `"pare"` namespace.
