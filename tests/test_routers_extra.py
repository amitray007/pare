"""Tests for router endpoints — uncovered paths in estimate.py, optimize.py, health.py, main.py."""

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    from main import app

    return TestClient(app, raise_server_exceptions=False)


def _make_png_bytes(size=(50, 50)):
    img = Image.new("RGB", size, (128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes(size=(50, 50)):
    img = Image.new("RGB", size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# --- /estimate endpoint ---


def test_estimate_file_upload(client):
    """POST /estimate with file upload."""
    data = _make_png_bytes()
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", data, "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "estimated_reduction_percent" in body


def test_estimate_json_url_mode(client):
    """POST /estimate with JSON body containing URL."""
    data = _make_png_bytes()
    with patch("routers.estimate.fetch_image", new=AsyncMock(return_value=data)):
        resp = client.post(
            "/estimate",
            json={"url": "https://example.com/image.png"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "estimated_reduction_percent" in body


def test_estimate_json_missing_url(client):
    """POST /estimate with JSON body but no url field."""
    resp = client.post(
        "/estimate",
        json={"not_url": "something"},
    )
    assert resp.status_code == 400


def test_estimate_bad_content_type(client):
    """POST /estimate with unsupported content type."""
    resp = client.post(
        "/estimate",
        content=b"raw bytes",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 400


def test_estimate_file_too_large(client):
    """POST /estimate with file exceeding size limit."""
    data = _make_png_bytes()
    with patch("routers.estimate.settings") as mock_s:
        mock_s.max_file_size_bytes = 10
        mock_s.max_file_size_mb = 0
        resp = client.post(
            "/estimate",
            files={"file": ("test.png", data, "image/png")},
        )
    assert resp.status_code == 413


def test_estimate_with_options(client):
    """POST /estimate with options JSON string."""
    data = _make_png_bytes()
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", data, "image/png")},
        data={"options": json.dumps({"quality": 40, "png_lossy": True})},
    )
    assert resp.status_code == 200


def test_estimate_with_invalid_options(client):
    """POST /estimate with malformed options JSON -> ignored, uses defaults."""
    data = _make_png_bytes()
    resp = client.post(
        "/estimate",
        files={"file": ("test.png", data, "image/png")},
        data={"options": "not json"},
    )
    assert resp.status_code == 200


# --- /optimize endpoint ---


def test_optimize_json_url_mode(client):
    """POST /optimize with JSON body containing URL."""
    data = _make_png_bytes()
    with patch("routers.optimize.fetch_image", new=AsyncMock(return_value=data)):
        resp = client.post(
            "/optimize",
            json={"url": "https://example.com/image.png"},
        )
    assert resp.status_code == 200
    assert "X-Original-Size" in resp.headers


def test_optimize_json_missing_url(client):
    """POST /optimize with JSON body but no url field."""
    resp = client.post(
        "/optimize",
        json={"not_url": "something"},
    )
    assert resp.status_code == 400


def test_optimize_bad_content_type(client):
    """POST /optimize with unsupported content type."""
    resp = client.post(
        "/optimize",
        content=b"raw bytes",
        headers={"Content-Type": "text/plain"},
    )
    assert resp.status_code == 400


def test_optimize_file_too_large(client):
    """POST /optimize with file exceeding size limit."""
    data = _make_png_bytes()
    with patch("routers.optimize.settings") as mock_s:
        mock_s.max_file_size_bytes = 10
        mock_s.max_file_size_mb = 0
        resp = client.post(
            "/optimize",
            files={"file": ("test.png", data, "image/png")},
        )
    assert resp.status_code == 413


def test_optimize_with_storage_config(client):
    """POST /optimize with storage config returns JSON response."""
    from schemas import StorageResult

    data = _make_png_bytes()
    mock_result = StorageResult(provider="gcs", url="gs://bucket/path", public_url=None)

    with patch("routers.optimize.fetch_image", new=AsyncMock(return_value=data)):
        with patch("routers.optimize.gcs_uploader") as mock_gcs:
            mock_gcs.upload = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/optimize",
                json={
                    "url": "https://example.com/image.png",
                    "storage": {
                        "provider": "gcs",
                        "bucket": "my-bucket",
                        "path": "output/image.png",
                    },
                },
                headers={"Content-Type": "application/json"},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "storage" in body


def test_optimize_json_with_storage(client):
    """POST /optimize with JSON URL + storage config -> JSON response."""
    data = _make_png_bytes()

    from schemas import StorageResult

    mock_result = StorageResult(provider="gcs", url="gs://b/p", public_url=None)

    with patch("routers.optimize.fetch_image", new=AsyncMock(return_value=data)):
        with patch("routers.optimize.gcs_uploader") as mock_gcs:
            mock_gcs.upload = AsyncMock(return_value=mock_result)
            resp = client.post(
                "/optimize",
                json={
                    "url": "https://example.com/image.png",
                    "optimization": {"quality": 60},
                    "storage": {
                        "provider": "gcs",
                        "bucket": "my-bucket",
                        "path": "output/image.png",
                    },
                },
            )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert "storage" in body


def test_optimize_form_with_options(client):
    """POST /optimize multipart with options containing optimization + storage."""
    data = _make_png_bytes()

    from schemas import StorageResult

    mock_result = StorageResult(provider="gcs", url="gs://b/p", public_url="https://cdn/p")

    with patch("routers.optimize.gcs_uploader") as mock_gcs:
        mock_gcs.upload = AsyncMock(return_value=mock_result)
        resp = client.post(
            "/optimize",
            files={"file": ("test.png", data, "image/png")},
            data={
                "options": json.dumps(
                    {
                        "optimization": {"quality": 40},
                        "storage": {
                            "provider": "gcs",
                            "bucket": "my-bucket",
                            "path": "out.png",
                        },
                    }
                )
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "storage" in body


# --- /health endpoint ---


def test_health_degraded(client):
    """Health endpoint with missing tools."""
    with patch("routers.health.shutil.which", return_value=None):
        resp = client.get("/health")
    body = resp.json()
    assert body["status"] == "degraded"


def test_health_missing_python_lib(client):
    """Health endpoint: missing Python library."""
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def selective_import(name, *args, **kwargs):
        if name == "pillow_heif":
            raise ImportError("no pillow_heif")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=selective_import):
        # Just verify the check_tools function handles ImportError
        from routers.health import check_tools

        # Reset to force re-evaluation
        tools = check_tools()
        # Tools should contain all keys
        assert "pillow_heif" in tools


# --- main.py lifespan startup/shutdown ---


@pytest.mark.asyncio
async def test_lifespan_startup_and_shutdown():
    """Cover main.lifespan: startup tool check + shutdown redis close."""
    from main import app, lifespan

    with patch("routers.health.check_tools", return_value={"pngquant": True, "jpegtran": False}):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                async with lifespan(app):
                    pass

                # Verify warning logged for missing tool
                mock_logger.warning.assert_called_once()
                assert "Missing tools" in str(mock_logger.warning.call_args)


@pytest.mark.asyncio
async def test_lifespan_no_missing_tools():
    """Cover the branch where all tools are available (no warning logged)."""
    from main import app, lifespan

    all_tools = {
        "pngquant": True,
        "jpegtran": True,
        "gifsicle": True,
        "cwebp": True,
        "oxipng": True,
        "pillow_heif": True,
        "scour": True,
        "pillow": True,
        "jxl_plugin": True,
    }
    with patch("routers.health.check_tools", return_value=all_tools):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger
                async with lifespan(app):
                    pass
                mock_logger.warning.assert_not_called()


@pytest.mark.asyncio
async def test_lifespan_shutdown_closes_redis():
    """Cover the shutdown path that closes Redis."""
    from main import app, lifespan

    mock_redis = AsyncMock()
    with patch("routers.health.check_tools", return_value={}):
        with patch("main.setup_logging"):
            with patch("main.get_logger") as mock_get_logger:
                mock_get_logger.return_value = MagicMock()
                with patch("security.rate_limiter._redis", mock_redis):
                    async with lifespan(app):
                        pass
                    mock_redis.close.assert_awaited_once()


# --- middleware.py — _get_client_ip ---


def test_get_client_ip_forwarded_for():
    """Cover middleware._get_client_ip parsing X-Forwarded-For header."""
    from middleware import _get_client_ip

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": "1.2.3.4, 5.6.7.8, 9.10.11.12"}
    mock_request.client.host = "127.0.0.1"

    ip = _get_client_ip(mock_request)
    assert ip == "1.2.3.4"


def test_get_client_ip_single_forwarded():
    """Cover single IP in X-Forwarded-For."""
    from middleware import _get_client_ip

    mock_request = MagicMock()
    mock_request.headers = {"X-Forwarded-For": " 10.0.0.1 "}
    mock_request.client.host = "127.0.0.1"

    ip = _get_client_ip(mock_request)
    assert ip == "10.0.0.1"
