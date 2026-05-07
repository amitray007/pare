import os
from typing import Literal

from PIL import Image
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- Server ---
    port: int = 8080
    workers: int = 1
    graceful_shutdown_timeout: int = 30

    # --- File Limits ---
    max_file_size_mb: int = 32
    max_file_size_bytes: int = 0  # Computed in model_post_init
    max_image_pixels: int = 100_000_000  # 100 megapixels

    # --- Optimization Defaults ---
    default_quality: int = 80
    tool_timeout_seconds: int = 60

    # --- Concurrency ---
    compression_semaphore_size: int = 0  # 0 = use CPU count
    max_queue_depth: int = 0  # 0 = 2 * CPU count
    estimate_semaphore_size: int = 0  # 0 = 2 * compression_semaphore_size
    estimate_queue_depth: int = 0  # 0 = 2 * estimate_semaphore_size
    memory_budget_mb: int = 6144  # Memory budget for concurrent compressions (MB)

    # --- Security ---
    redis_url: str = ""
    rate_limit_public_rpm: int = 60
    rate_limit_public_burst: int = 10
    rate_limit_auth_enabled: bool = False
    rate_limit_auth_rpm: int = 0
    api_key: str = ""
    allowed_origins: str = "*"

    # --- URL Fetching ---
    url_fetch_timeout: int = 30
    url_fetch_max_redirects: int = 5

    # --- Encoder Selection ---
    jpeg_encoder: str = "pillow"  # "pillow" (default) or "cjpeg" (MozJPEG fallback)

    # --- Format Support ---
    enable_jxl: bool = False  # Requires libjxl build (cjxl, djxl, jpegli, jxlpy)

    # --- Fitted Estimator ---
    # "off": existing direct-encode-sample path unchanged (default, safe rollout gate).
    # "active": fitted BPP curve runs instead of direct-encode-sample for supported formats.
    # Env-var rollout only — do not flip in application code; use env FITTED_ESTIMATOR_MODE=active.
    # Production flips require the staged rollout playbook.
    fitted_estimator_mode: Literal["off", "active"] = "off"

    # --- Range-Fetch / Header-Only Estimation ---
    header_only_min_size_bytes: int = 1_048_576  # 1 MB
    """Minimum file size for the header-only Range-fetch path. Smaller files round-trip
    the Range request in similar time to a full small download, so we just download.
    Used by routers/estimate.py URL-mode dispatch."""

    large_file_threshold_bytes: int = (
        10 * 1024 * 1024
    )  # 10 MB (unchanged from prior hardcoded value)
    """Threshold above which the legacy thumbnail_url fast path activates.
    Configurable now (was hardcoded 10MB constant). Default preserves prior behavior.
    The new header-only path uses its own threshold (header_only_min_size_bytes,
    default 1 MB) independent of this."""

    # --- Logging ---
    log_level: str = "ERROR"

    model_config = {"env_prefix": "", "case_sensitive": False}

    def model_post_init(self, __context) -> None:
        if self.max_file_size_bytes == 0:
            self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        if self.compression_semaphore_size == 0:
            cpu = os.cpu_count() or 4
            self.compression_semaphore_size = max(2, cpu)
        if self.max_queue_depth == 0:
            self.max_queue_depth = 2 * self.compression_semaphore_size
        if self.estimate_semaphore_size == 0:
            self.estimate_semaphore_size = 2 * self.compression_semaphore_size
        if self.estimate_queue_depth == 0:
            self.estimate_queue_depth = 2 * self.estimate_semaphore_size
        Image.MAX_IMAGE_PIXELS = self.max_image_pixels


settings = Settings()
