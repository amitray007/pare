"""Tests for memory guards: pixel limits, decompressed size validation, frame limits."""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from exceptions import BackpressureError, ImageTooLargeError
from utils.concurrency import CompressionGate
from utils.image_validation import validate_image_dimensions


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


class TestDecompressedSizeValidation:
    """validate_image_dimensions() checks."""

    def test_small_image_passes(self):
        """Normal-sized images pass validation."""
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        # Should not raise
        validate_image_dimensions(buf.getvalue())

    def test_huge_dimensions_rejected(self):
        """Images with decompressed size > 512MB are rejected."""
        # Create a minimal JPEG with huge claimed dimensions via mock
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        # Patch Image.open to return a mock with huge dimensions
        with patch("utils.image_validation.Image.open") as mock_open:
            mock_img = mock_open.return_value
            mock_img.size = (20000, 20000)  # 400MP * 3 bpp = 1.2GB > 512MB
            mock_img.mode = "RGB"
            mock_img.n_frames = 1

            with pytest.raises(ImageTooLargeError) as exc_info:
                validate_image_dimensions(data)
            assert exc_info.value.status_code == 413

    def test_frame_count_limit(self):
        """Animated images with > 500 frames are rejected."""
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        with patch("utils.image_validation.Image.open") as mock_open:
            mock_img = mock_open.return_value
            mock_img.size = (100, 100)
            mock_img.mode = "RGB"
            mock_img.n_frames = 600

            with pytest.raises(ImageTooLargeError, match="frames"):
                validate_image_dimensions(data)

    def test_svg_passes_gracefully(self):
        """SVG files (which can't be opened by Pillow) pass through."""
        svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>'
        # Should not raise — Image.open fails, caught by except handler
        validate_image_dimensions(svg_data)

    def test_single_frame_within_limit(self):
        """Single-frame image within 512MB decompressed passes."""
        img = Image.new("RGB", (100, 100))
        buf = io.BytesIO()
        img.save(buf, format="BMP")
        data = buf.getvalue()

        with patch("utils.image_validation.Image.open") as mock_open:
            mock_img = mock_open.return_value
            mock_img.size = (5000, 5000)  # 25MP * 3 = 75MB, under 512MB
            mock_img.mode = "RGB"
            mock_img.n_frames = 1

            # Should not raise
            validate_image_dimensions(data)


class TestCompressionGate:
    """Memory-aware CompressionGate tests."""

    @pytest.mark.asyncio
    async def test_acquire_release_basic(self):
        """Basic acquire/release without memory tracking."""
        gate = CompressionGate(semaphore_size=2, max_queue=4, memory_budget_bytes=1024)
        await gate.acquire()
        assert gate.active_jobs == 1
        gate.release()
        assert gate.active_jobs == 0

    @pytest.mark.asyncio
    async def test_memory_budget_rejection(self):
        """Requests exceeding memory budget are rejected with 503."""
        gate = CompressionGate(semaphore_size=4, max_queue=8, memory_budget_bytes=100)

        # First request with 60 bytes estimated — should pass
        await gate.acquire(estimated_memory=60)

        # Second request with 60 bytes — would exceed 100 byte budget
        with pytest.raises(BackpressureError):
            await gate.acquire(estimated_memory=60)

        gate.release(estimated_memory=60)

    @pytest.mark.asyncio
    async def test_memory_tracking_decrements(self):
        """Memory used decrements correctly on release."""
        gate = CompressionGate(semaphore_size=4, max_queue=8, memory_budget_bytes=200)

        await gate.acquire(estimated_memory=80)
        assert gate._memory_used == 80

        await gate.acquire(estimated_memory=50)
        assert gate._memory_used == 130

        gate.release(estimated_memory=80)
        assert gate._memory_used == 50

        gate.release(estimated_memory=50)
        assert gate._memory_used == 0

    @pytest.mark.asyncio
    async def test_backward_compatible_no_memory(self):
        """acquire() without estimated_memory still works (slot-based only)."""
        gate = CompressionGate(semaphore_size=2, max_queue=4, memory_budget_bytes=100)

        await gate.acquire()  # No estimated_memory
        assert gate._memory_used == 0
        gate.release()

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        """Queue depth limit still works with memory tracking."""
        gate = CompressionGate(semaphore_size=2, max_queue=2, memory_budget_bytes=10000)

        await gate.acquire(estimated_memory=10)
        await gate.acquire(estimated_memory=10)

        # Queue is full (2/2)
        with pytest.raises(BackpressureError, match="queue full"):
            await gate.acquire(estimated_memory=10)

        gate.release(estimated_memory=10)
        gate.release(estimated_memory=10)

    @pytest.mark.asyncio
    async def test_first_request_always_admitted(self):
        """First request passes even if its estimate exceeds budget (budget=0 used)."""
        gate = CompressionGate(semaphore_size=2, max_queue=4, memory_budget_bytes=10)

        # First request — memory_used is 0, so condition `self._memory_used > 0` is false
        await gate.acquire(estimated_memory=50)
        assert gate._memory_used == 50

        gate.release(estimated_memory=50)


class TestEstimateEndpointGuard:
    """Estimate endpoint semaphore integration."""

    def test_estimate_normal_request(self, client, sample_png):
        """Normal estimate requests work with the semaphore."""
        resp = client.post("/estimate", files={"file": ("img.png", sample_png)})
        assert resp.status_code == 200
