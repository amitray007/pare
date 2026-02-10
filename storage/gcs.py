import logging

from google.cloud import storage as gcs_lib

from exceptions import PareError
from schemas import StorageConfig, StorageResult
from utils.format_detect import MIME_TYPES

logger = logging.getLogger(__name__)


class GCSUploader:
    """Google Cloud Storage upload handler.

    Authentication:
    - Cloud Run (production): Workload identity (automatic)
    - Local development: GOOGLE_APPLICATION_CREDENTIALS env var
    - CI/CD: Service account key or workload identity federation
    """

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Lazy-initialized GCS client."""
        if self._client is None:
            self._client = gcs_lib.Client()
        return self._client

    async def upload(
        self,
        data: bytes,
        fmt: str,
        config: StorageConfig,
    ) -> StorageResult:
        """Upload optimized bytes to GCS.

        Args:
            data: Optimized image bytes.
            fmt: Image format string (e.g., "png", "jpeg").
            config: Storage configuration from the request.

        Returns:
            StorageResult with GCS URLs.

        Raises:
            PareError: If upload fails.
        """
        try:
            bucket = self.client.bucket(
                config.bucket,
                user_project=config.project,
            )
            blob = bucket.blob(config.path)

            content_type = MIME_TYPES.get(fmt, "application/octet-stream")
            blob.upload_from_string(data, content_type=content_type)

            if config.public:
                blob.make_public()

            gs_url = f"gs://{config.bucket}/{config.path}"
            public_url = (
                f"https://storage.googleapis.com/{config.bucket}/{config.path}"
                if config.public
                else None
            )

            return StorageResult(
                provider="gcs",
                url=gs_url,
                public_url=public_url,
            )

        except Exception as e:
            raise PareError(
                f"GCS upload failed: {e}",
                bucket=config.bucket,
                path=config.path,
            )


# Module-level singleton
gcs_uploader = GCSUploader()
