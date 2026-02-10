"""Tests for security modules: SSRF, SVG sanitization, file validation, auth."""

import pytest

from exceptions import (
    AuthenticationError,
    FileTooLargeError,
    SSRFError,
    UnsupportedFormatError,
)
from security.ssrf import validate_url
from security.svg_sanitizer import sanitize_svg
from security.file_validation import validate_file
from security.auth import authenticate
from utils.format_detect import ImageFormat


# --- SSRF Tests ---


def test_ssrf_http_rejected():
    """HTTP scheme rejected (HTTPS only)."""
    with pytest.raises(SSRFError):
        validate_url("http://example.com/image.png")


def test_ssrf_metadata_google():
    """metadata.google.internal blocked."""
    with pytest.raises(SSRFError):
        validate_url("https://metadata.google.internal/")


def test_ssrf_metadata_google_trailing_dot():
    """metadata.google.internal. (trailing dot) blocked."""
    with pytest.raises(SSRFError):
        validate_url("https://metadata.google.internal./")


def test_ssrf_localhost():
    """127.0.0.1 / localhost blocked."""
    with pytest.raises(SSRFError):
        validate_url("https://localhost/")


def test_ssrf_link_local():
    """169.254.x.x (metadata IP) blocked."""
    with pytest.raises(SSRFError):
        validate_url("https://169.254.169.254/latest/meta-data/")


def test_ssrf_no_hostname():
    """Empty hostname rejected."""
    with pytest.raises(SSRFError):
        validate_url("https://")


def test_ssrf_valid_https():
    """Valid HTTPS URL to public IP accepted."""
    result = validate_url("https://www.google.com/image.png")
    assert result == "https://www.google.com/image.png"


# --- SVG Sanitizer Tests ---


def test_svg_script_stripped():
    """<script> tags removed from SVG."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script><rect width="10" height="10"/></svg>'
    result = sanitize_svg(svg)
    assert b"<script>" not in result
    assert b"alert" not in result
    assert b"rect" in result


def test_svg_event_handlers_stripped():
    """on* event handler attributes removed."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect onload="x()" onclick="y()" width="10" height="10"/></svg>'
    result = sanitize_svg(svg)
    assert b"onload" not in result
    assert b"onclick" not in result
    assert b"rect" in result


def test_svg_foreign_object_stripped():
    """<foreignObject> removed."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><foreignObject><body xmlns="http://www.w3.org/1999/xhtml">evil</body></foreignObject><rect width="10" height="10"/></svg>'
    result = sanitize_svg(svg)
    assert b"foreignObject" not in result
    assert b"foreignobject" not in result.lower()


def test_svg_data_uri_stripped():
    """data:text/html in href removed."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><a href="data:text/html,&lt;script&gt;alert(1)&lt;/script&gt;"><text>click</text></a></svg>'
    result = sanitize_svg(svg)
    assert b"data:text/html" not in result


def test_svg_css_import_stripped():
    """@import url() rules stripped from <style> elements."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><style>@import url(https://evil.com/steal.css); rect { fill: red; }</style><rect width="10" height="10"/></svg>'
    result = sanitize_svg(svg)
    assert b"@import" not in result
    assert b"rect" in result


def test_svg_valid_content_preserved():
    """Non-malicious SVG content intact after sanitization."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100" fill="red"/><circle cx="50" cy="50" r="25" fill="blue"/></svg>'
    result = sanitize_svg(svg)
    assert b"rect" in result
    assert b"circle" in result
    assert b'fill="red"' in result or b"fill='red'" in result or b"fill=&quot;red&quot;" in result or b'fill="red"' in result


def test_svg_xxe_blocked():
    """XXE entity expansion blocked by defusedxml."""
    xxe_svg = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<svg xmlns="http://www.w3.org/2000/svg"><text>&xxe;</text></svg>"""
    # defusedxml should raise an error on DTD with entities
    from exceptions import OptimizationError
    with pytest.raises((OptimizationError, Exception)):
        sanitize_svg(xxe_svg)


# --- File Validation Tests ---


def test_file_validation_png(sample_png):
    """Valid PNG detected."""
    fmt = validate_file(sample_png)
    assert fmt == ImageFormat.PNG


def test_file_validation_jpeg(sample_jpeg):
    """Valid JPEG detected."""
    fmt = validate_file(sample_jpeg)
    assert fmt == ImageFormat.JPEG


def test_file_validation_svg(sample_svg):
    """Valid SVG detected."""
    fmt = validate_file(sample_svg)
    assert fmt == ImageFormat.SVG


def test_file_validation_unknown_format():
    """Unknown magic bytes rejected."""
    with pytest.raises(UnsupportedFormatError):
        validate_file(b"not an image format at all")


# --- Auth Tests ---


class MockHeaders:
    def __init__(self, headers):
        self._headers = headers

    def get(self, key, default=""):
        return self._headers.get(key, default)


class MockRequest:
    def __init__(self, headers):
        self.headers = MockHeaders(headers)


def test_auth_no_header():
    """No auth header -> public (False, not error)."""
    req = MockRequest({})
    result = authenticate(req)
    assert result is False


def test_auth_valid_bearer():
    """Valid Bearer token -> True (dev mode, no API_KEY set)."""
    req = MockRequest({"Authorization": "Bearer any-key"})
    result = authenticate(req)
    assert result is True


def test_auth_malformed_header():
    """Non-Bearer format -> 401."""
    req = MockRequest({"Authorization": "Basic abc123"})
    with pytest.raises(AuthenticationError):
        authenticate(req)


# --- Endpoint Security Tests ---


def test_endpoint_auth_malformed(client, sample_svg):
    """Malformed auth header on optimize -> 401."""
    resp = client.post(
        "/optimize",
        files={"file": ("test.svg", sample_svg, "image/svg+xml")},
        headers={"Authorization": "Basic xyz"},
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"


def test_endpoint_unsupported_format(client):
    """Random bytes on optimize -> 415."""
    resp = client.post(
        "/optimize",
        files={"file": ("test.bin", b"random bytes", "application/octet-stream")},
    )
    assert resp.status_code == 415


def test_endpoint_ssrf_blocked(client):
    """SSRF attempt via JSON URL mode -> 422."""
    resp = client.post(
        "/optimize",
        json={"url": "http://169.254.169.254/latest/meta-data/"},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "ssrf_blocked"
