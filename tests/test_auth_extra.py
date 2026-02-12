"""Tests for security/auth.py — authentication logic."""

import pytest
from unittest.mock import patch, MagicMock

from exceptions import AuthenticationError
from security.auth import authenticate


def _make_request(auth_header=None):
    """Create a mock FastAPI Request with optional Authorization header."""
    req = MagicMock()
    if auth_header:
        req.headers = {"Authorization": auth_header}
    else:
        req.headers = {}
    return req


def test_auth_no_header():
    """No auth header -> returns False (public request)."""
    req = _make_request()
    assert authenticate(req) is False


def test_auth_malformed_header():
    """Non-Bearer auth header -> AuthenticationError."""
    req = _make_request("Basic dXNlcjpwYXNz")
    with pytest.raises(AuthenticationError):
        authenticate(req)


def test_auth_no_api_key_configured():
    """No API_KEY in config (dev mode) -> accept all bearers."""
    req = _make_request("Bearer any-key-works")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = ""
        assert authenticate(req) is True


def test_auth_valid_key():
    """Correct API key -> True."""
    req = _make_request("Bearer secret-key-123")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = "secret-key-123"
        assert authenticate(req) is True


def test_auth_invalid_key():
    """Wrong API key -> AuthenticationError."""
    req = _make_request("Bearer wrong-key")
    with patch("security.auth.settings") as mock_settings:
        mock_settings.api_key = "correct-key"
        with pytest.raises(AuthenticationError):
            authenticate(req)
