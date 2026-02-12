"""Tests for GCS upload integration (mocked)."""

from unittest.mock import MagicMock

import pytest

from schemas import StorageConfig
from storage.gcs import GCSUploader


@pytest.fixture
def storage_config():
    return StorageConfig(
        provider="gcs",
        bucket="test-bucket",
        path="images/optimized.png",
    )


@pytest.fixture
def public_storage_config():
    return StorageConfig(
        provider="gcs",
        bucket="test-bucket",
        path="images/optimized.png",
        public=True,
    )


@pytest.mark.asyncio
async def test_gcs_upload_mock(storage_config):
    """Upload bytes to mocked GCS -> correct bucket/path."""
    uploader = GCSUploader()

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    uploader._client = mock_client

    result = await uploader.upload(b"fake image data", "png", storage_config)

    mock_client.bucket.assert_called_once_with("test-bucket", user_project=None)
    mock_bucket.blob.assert_called_once_with("images/optimized.png")
    mock_blob.upload_from_string.assert_called_once_with(
        b"fake image data", content_type="image/png"
    )
    assert result.provider == "gcs"
    assert result.url == "gs://test-bucket/images/optimized.png"
    assert result.public_url is None


@pytest.mark.asyncio
async def test_gcs_upload_public(public_storage_config):
    """public=True -> blob.make_public() called, public_url returned."""
    uploader = GCSUploader()

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    uploader._client = mock_client

    result = await uploader.upload(b"fake image data", "png", public_storage_config)

    mock_blob.make_public.assert_called_once()
    assert result.public_url == "https://storage.googleapis.com/test-bucket/images/optimized.png"


@pytest.mark.asyncio
async def test_gcs_upload_private(storage_config):
    """public=False -> no public_url in result."""
    uploader = GCSUploader()

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    uploader._client = mock_client

    result = await uploader.upload(b"fake image data", "png", storage_config)

    mock_blob.make_public.assert_not_called()
    assert result.public_url is None


@pytest.mark.asyncio
async def test_gcs_upload_custom_project():
    """project field passed to GCS client."""
    config = StorageConfig(
        provider="gcs",
        bucket="test-bucket",
        path="images/opt.png",
        project="my-gcp-project",
    )
    uploader = GCSUploader()

    mock_blob = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    uploader._client = mock_client

    await uploader.upload(b"data", "png", config)

    mock_client.bucket.assert_called_once_with("test-bucket", user_project="my-gcp-project")


@pytest.mark.asyncio
async def test_gcs_upload_failure_handling(storage_config):
    """GCS error -> PareError raised with details."""
    uploader = GCSUploader()

    mock_client = MagicMock()
    mock_client.bucket.side_effect = Exception("Bucket not found")
    uploader._client = mock_client

    from exceptions import PareError

    with pytest.raises(PareError, match="GCS upload failed"):
        await uploader.upload(b"data", "png", storage_config)
