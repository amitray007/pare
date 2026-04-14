"""Tests for memory guards: pixel limits, decompressed size validation, frame limits."""

import io

import pytest
from PIL import Image


class TestPixelCountLimit:
    """Image.MAX_IMAGE_PIXELS enforcement via middleware."""

    def test_small_image_passes(self, client, sample_png):
        """Normal-sized images pass the pixel limit."""
        resp = client.post("/optimize", files={"file": ("img.png", sample_png)})
        assert resp.status_code == 200

    def test_decompression_bomb_returns_413(self, client):
        """Images exceeding MAX_IMAGE_PIXELS return 413 with image_too_large error.

        Uses BMP format because the BMP optimizer calls Image.open() via Pillow,
        which checks MAX_IMAGE_PIXELS. Formats using CLI tools (PNG via pngquant)
        bypass Pillow's check — the Phase 2 decompressed size validation catches those.
        """
        from config import settings

        original_limit = settings.max_image_pixels
        try:
            # Set absurdly low limit so even a 100x100 image triggers it
            Image.MAX_IMAGE_PIXELS = 50
            settings.max_image_pixels = 50

            img = Image.new("RGB", (100, 100), color=(255, 0, 0))
            buf = io.BytesIO()
            img.save(buf, format="BMP")
            data = buf.getvalue()

            resp = client.post("/optimize", files={"file": ("bomb.bmp", data)})
            assert resp.status_code == 413
            body = resp.json()
            assert body["error"] == "image_too_large"
            assert body["success"] is False
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit
            settings.max_image_pixels = original_limit

    def test_error_response_format(self, client):
        """DecompressionBombError response matches PareError JSON structure."""
        from config import settings

        original_limit = settings.max_image_pixels
        try:
            Image.MAX_IMAGE_PIXELS = 50
            settings.max_image_pixels = 50

            img = Image.new("RGB", (100, 100), color=(0, 0, 255))
            buf = io.BytesIO()
            img.save(buf, format="BMP")
            data = buf.getvalue()

            resp = client.post("/optimize", files={"file": ("test.bmp", data)})
            body = resp.json()

            # Must have the same structure as PareError responses
            assert resp.status_code == 413
            assert "success" in body
            assert "error" in body
            assert "message" in body
            assert isinstance(body["message"], str)
            assert len(body["message"]) > 0
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit
            settings.max_image_pixels = original_limit

    def test_estimate_endpoint_also_protected(self, client):
        """The /estimate endpoint also catches DecompressionBombError."""
        from config import settings

        original_limit = settings.max_image_pixels
        try:
            Image.MAX_IMAGE_PIXELS = 50
            settings.max_image_pixels = 50

            # Use BMP — estimator opens image via Pillow for dimension detection
            img = Image.new("RGB", (100, 100), color=(0, 255, 0))
            buf = io.BytesIO()
            img.save(buf, format="BMP")
            data = buf.getvalue()

            resp = client.post("/estimate", files={"file": ("test.bmp", data)})
            assert resp.status_code == 413
        finally:
            Image.MAX_IMAGE_PIXELS = original_limit
            settings.max_image_pixels = original_limit
