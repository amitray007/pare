# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Purpose

Security layers applied via `SecurityMiddleware` (in root `middleware.py`). All modules are invoked per-request before the route handler runs.

## Modules

- **`auth.py`**: Bearer token validation. Empty `API_KEY` config = dev mode (accepts all tokens). No token header = public request (not an error).
- **`rate_limiter.py`**: Redis-backed sliding window (per-minute) + burst limiter (per-10s). Fail-open design: if Redis is unavailable, requests pass through. Lazy Redis init.
- **`ssrf.py`**: Validates URLs before fetch. Blocks private/reserved IPs, cloud metadata endpoints, non-HTTPS schemes. Resolves DNS before checking to prevent rebinding attacks.
- **`svg_sanitizer.py`**: Strips `<script>`, `<foreignObject>`, event handlers (`on*` attributes), `data:text/html` URIs, CSS `@import url()`. Uses `defusedxml` for XXE-safe parsing.
- **`file_validation.py`**: Size limit check + format detection (delegates to `utils/format_detect.py`).

## Design Decisions

- Rate limiter uses **fail-open** — Redis outage should not take down the API
- SSRF protection resolves DNS and checks **every** resolved IP address against blocked ranges
- SVG sanitization happens during optimization (in `optimizers/svg.py`), not in the security middleware
