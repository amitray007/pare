"""Extra security tests — file_validation, svg_sanitizer edge cases, ssrf."""

from unittest.mock import patch

import pytest

from exceptions import FileTooLargeError, OptimizationError, SSRFError
from security.file_validation import validate_file
from security.ssrf import validate_url
from security.svg_sanitizer import _is_external_url, sanitize_svg

# --- file_validation ---


def test_validate_file_too_large():
    """File over max_file_size_bytes raises FileTooLargeError."""
    data = b"x" * (33 * 1024 * 1024 + 1)  # > 32MB
    with pytest.raises(FileTooLargeError):
        validate_file(data)


# --- svg_sanitizer edge cases ---


def test_sanitize_svg_malformed_xml():
    """Malformed SVG -> OptimizationError."""
    with pytest.raises(OptimizationError, match="Malformed SVG"):
        sanitize_svg(b"<svg><not closed")


def test_sanitize_svg_use_external_href():
    """<use> with external href gets stripped."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><use href="https://evil.com/icon.svg#x"/></svg>'
    result = sanitize_svg(svg)
    assert b"evil.com" not in result


def test_sanitize_svg_use_internal_href():
    """<use> with internal href (fragment) is preserved."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><use href="#myshape"/></svg>'
    result = sanitize_svg(svg)
    assert b"#myshape" in result


def test_is_external_url():
    assert _is_external_url("https://example.com") is True
    assert _is_external_url("http://example.com") is True
    assert _is_external_url("#local") is False
    assert _is_external_url("data:image/png;base64,abc") is False


# --- ssrf edge cases ---


def test_ssrf_dns_resolution_failure():
    """DNS resolution failure -> SSRFError."""
    import socket

    with patch("socket.getaddrinfo", side_effect=socket.gaierror("no such host")):
        with pytest.raises(SSRFError, match="resolve"):
            validate_url("https://nonexistent.invalid/image.png")


def test_ssrf_validate_url_happy_path():
    """Cover validate_url returning the URL for a valid public IP."""
    with patch("security.ssrf.socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [
            (2, 1, 6, "", ("93.184.216.34", 443)),
        ]
        result = validate_url("https://example.com/image.png")
        assert result == "https://example.com/image.png"


# --- svg_sanitizer _find_parent ---


def test_svg_sanitizer_find_parent_root():
    """Cover _find_parent returning None for root element."""
    from security.svg_sanitizer import _find_parent
    from xml.etree.ElementTree import Element

    root = Element("svg")
    result = _find_parent(root, root)
    assert result is None


def test_svg_sanitizer_find_parent_not_found():
    """Cover _find_parent returning None when target not in tree."""
    from security.svg_sanitizer import _find_parent
    from xml.etree.ElementTree import Element

    root = Element("svg")
    child = Element("rect")
    root.append(child)
    orphan = Element("circle")

    result = _find_parent(root, orphan)
    assert result is None
