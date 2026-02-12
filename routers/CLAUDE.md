# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

FastAPI endpoint handlers. Three routes:

- **`POST /optimize`** (`optimize.py`): Two input modes (multipart file upload, JSON with URL) and two response modes (raw bytes with X-* headers, JSON with storage URL). Acquires a `CompressionGate` semaphore slot before optimization.
- **`POST /estimate`** (`estimate.py`): Same input modes as /optimize but lightweight (~20-50ms). Does **not** acquire a semaphore slot.
- **`GET /health`** (`health.py`): Checks availability of all CLI tools and Python libraries. Returns `"ok"` or `"degraded"`.

## Request Flow

Middleware (`middleware.py` in root) runs before these handlers: request ID injection -> authentication -> rate limiting. The routers handle input parsing, size/format validation, and response formatting.

## Conventions

- Optimization config comes from the `options` form field (JSON string) in multipart mode, or from `body.optimization` in JSON mode
- Binary responses include stats in `X-*` headers (Original-Size, Optimized-Size, Reduction-Percent, Original-Format, Optimization-Method, Request-ID)
- URL fetch mode requires HTTPS and runs SSRF validation at each redirect hop
