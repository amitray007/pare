# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this directory.

## Running Tests

```bash
pytest tests/                              # All tests
pytest tests/test_optimize.py              # One file
pytest tests/ -k "bmp"                     # Filter by keyword
pytest tests/test_optimize.py::test_optimize_png_file_upload -v  # Single test
```

## Fixtures

Defined in `conftest.py`:
- `client` / `strict_client`: FastAPI TestClient instances (strict raises server exceptions)
- `sample_png`, `sample_jpeg`, `sample_webp`, `sample_gif`, `sample_svg`, `sample_bmp`, `sample_tiff`, `tiny_png`: Raw bytes from `tests/sample_images/`
- `malicious_svg`: SVG with XSS payloads for security testing
- `auth_headers`: Bearer token headers for auth tests

## Test Organization

- **`test_optimize.py`**: `/optimize` endpoint — multipart/JSON modes, quality options, error codes (413, 415, 400, 422, 429), output-never-larger guarantee
- **`test_estimate.py`**: `/estimate` endpoint — accuracy, confidence levels, format detection
- **`test_formats.py`**: All 11 formats, APNG detection, format-specific behavior
- **`test_security.py`**: Auth (valid/invalid/missing tokens), rate limiting, SSRF blocking, SVG sanitization
- **`test_gcs.py`**: GCS upload (mocked)
- **`test_logging.py`**: Structured JSON output validation

## Notes

- Some tests (JPEG, WebP, GIF) may skip or degrade gracefully on systems without CLI tools (cjpeg, cwebp, gifsicle)
- Tests use `raise_server_exceptions=False` by default to test error response formatting; use `strict_client` when you want exceptions to propagate
