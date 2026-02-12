"""Tests for /health endpoint and check_tools utility."""

from unittest.mock import patch, MagicMock

import pytest

from routers.health import check_tools, router


def test_check_tools_all_available():
    """All tools available -> all True."""
    with patch("shutil.which", return_value="/usr/bin/tool"):
        results = check_tools()
    for name in ("pngquant", "cjpeg", "jpegtran", "gifsicle", "cwebp"):
        assert results[name] is True


def test_check_tools_none_available():
    """No CLI tools on PATH -> all False."""
    with patch("shutil.which", return_value=None):
        results = check_tools()
    for name in ("pngquant", "cjpeg", "jpegtran", "gifsicle", "cwebp"):
        assert results[name] is False


def test_check_tools_python_libs():
    """Python libraries (oxipng, pillow_heif, scour, pillow) detected."""
    with patch("shutil.which", return_value=None):
        results = check_tools()
    # These libraries are installed in our test env
    assert "oxipng" in results
    assert "pillow" in results
    assert "scour" in results
    assert "pillow_heif" in results


def test_check_tools_oxipng_missing():
    """oxipng import failure -> oxipng=False."""
    with patch("shutil.which", return_value=None):
        with patch.dict("sys.modules", {"oxipng": None}):
            with patch("builtins.__import__", side_effect=_mock_import_no_oxipng):
                results = check_tools()
    assert results["oxipng"] is False


def test_health_endpoint_ok(client):
    """Health endpoint returns status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "tools" in data
    assert data["version"] == "0.1.0"


def test_health_endpoint_degraded(client):
    """When a tool is missing, status=degraded."""
    with patch("routers.health.check_tools", return_value={"pngquant": False, "cjpeg": True}):
        resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "degraded"


def test_health_endpoint_all_ok(client):
    """When all tools present, status=ok."""
    all_true = {k: True for k in ("pngquant", "cjpeg", "jpegtran", "gifsicle", "cwebp", "oxipng", "pillow_heif", "scour", "pillow")}
    with patch("routers.health.check_tools", return_value=all_true):
        resp = client.get("/health")
    data = resp.json()
    assert data["status"] == "ok"


def _mock_import_no_oxipng(name, *args, **kwargs):
    if name == "oxipng":
        raise ImportError("no oxipng")
    return MagicMock()
