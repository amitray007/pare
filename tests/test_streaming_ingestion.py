"""Tests for streaming multipart upload ingestion (_read_upload_streaming).

Verifies that oversized uploads are rejected after the first chunk that crosses
max_file_size_bytes, before the full body is buffered in RAM.
"""

import pytest

from exceptions import FileTooLargeError
from routers.optimize import _UPLOAD_CHUNK_SIZE, _read_upload_streaming


class _FakeUploadFile:
    """Minimal UploadFile stand-in that feeds fixed-size chunks from a bytearray."""

    def __init__(self, data: bytes, chunk_size: int = _UPLOAD_CHUNK_SIZE):
        self._data = memoryview(data)
        self._pos = 0
        self._chunk_size = chunk_size

    async def read(self, size: int = -1) -> bytes:
        if size < 0:
            chunk = bytes(self._data[self._pos :])
            self._pos = len(self._data)
        else:
            chunk = bytes(self._data[self._pos : self._pos + size])
            self._pos += len(chunk)
        return chunk


@pytest.mark.asyncio
async def test_streaming_rejects_oversized_upload():
    """A payload of small_limit + 1 bytes must raise FileTooLargeError."""
    from unittest.mock import patch

    small_limit = 64 * 1024  # 64 KiB — avoids allocating 32 MB+ in CI
    oversized = b"x" * (small_limit + 1)
    fake_file = _FakeUploadFile(oversized)

    with patch("routers.optimize.settings") as mock_settings:
        mock_settings.max_file_size_bytes = small_limit
        with pytest.raises(FileTooLargeError) as exc_info:
            await _read_upload_streaming(fake_file)

    assert exc_info.value.details["limit"] == small_limit
    assert exc_info.value.details["file_size"] > small_limit


@pytest.mark.asyncio
async def test_streaming_rejects_before_full_read():
    """Rejection fires on the first chunk that crosses the limit, not after full read."""
    # Payload is exactly 2 chunks; limit is set so the first chunk alone is already over.
    small_limit = _UPLOAD_CHUNK_SIZE - 1  # less than one chunk
    payload = b"y" * (_UPLOAD_CHUNK_SIZE * 2)
    fake_file = _FakeUploadFile(payload)

    from unittest.mock import patch

    with patch("routers.optimize.settings") as mock_settings:
        mock_settings.max_file_size_bytes = small_limit
        mock_settings.max_file_size_mb = 0
        with pytest.raises(FileTooLargeError):
            await _read_upload_streaming(fake_file)

    # Only the first chunk should have been consumed; position advances by at most chunk_size
    assert fake_file._pos <= _UPLOAD_CHUNK_SIZE


@pytest.mark.asyncio
async def test_streaming_accepts_valid_upload():
    """A payload within the size limit is returned as bytes without error."""
    # Use a tiny 100-byte payload well under any realistic limit.
    payload = b"z" * 100
    fake_file = _FakeUploadFile(payload)

    result = await _read_upload_streaming(fake_file)

    assert result == payload


@pytest.mark.asyncio
async def test_streaming_empty_upload():
    """An empty upload (0 bytes) is returned as empty bytes."""
    fake_file = _FakeUploadFile(b"")

    result = await _read_upload_streaming(fake_file)

    assert result == b""
