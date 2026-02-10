import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app

SAMPLE_DIR = Path(__file__).parent / "sample_images"


@pytest.fixture
def client():
    """FastAPI test client (does not raise server exceptions)."""
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def strict_client():
    """FastAPI test client that raises server exceptions."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def sample_png():
    return (SAMPLE_DIR / "sample.png").read_bytes()


@pytest.fixture
def sample_jpeg():
    return (SAMPLE_DIR / "sample.jpg").read_bytes()


@pytest.fixture
def sample_webp():
    return (SAMPLE_DIR / "sample.webp").read_bytes()


@pytest.fixture
def sample_gif():
    return (SAMPLE_DIR / "sample.gif").read_bytes()


@pytest.fixture
def sample_svg():
    return (SAMPLE_DIR / "sample.svg").read_bytes()


@pytest.fixture
def malicious_svg():
    return (SAMPLE_DIR / "malicious.svg").read_bytes()


@pytest.fixture
def sample_bmp():
    return (SAMPLE_DIR / "sample.bmp").read_bytes()


@pytest.fixture
def sample_tiff():
    return (SAMPLE_DIR / "sample.tiff").read_bytes()


@pytest.fixture
def tiny_png():
    return (SAMPLE_DIR / "tiny.png").read_bytes()


@pytest.fixture
def auth_headers():
    """Headers for authenticated requests (dev mode — no API_KEY set)."""
    return {"Authorization": "Bearer test-api-key"}
