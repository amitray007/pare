import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- Server ---
    port: int = 8080
    workers: int = 4
    graceful_shutdown_timeout: int = 30

    # --- File Limits ---
    max_file_size_mb: int = 32
    max_file_size_bytes: int = 0  # Computed in model_post_init

    # --- Optimization Defaults ---
    default_quality: int = 80
    tool_timeout_seconds: int = 60

    # --- Concurrency ---
    compression_semaphore_size: int = 0  # 0 = use CPU count
    max_queue_depth: int = 0  # 0 = 2 * CPU count

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

    # --- Logging ---
    log_level: str = "ERROR"

    model_config = {"env_prefix": "", "case_sensitive": False}

    def model_post_init(self, __context) -> None:
        if self.max_file_size_bytes == 0:
            self.max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        if self.compression_semaphore_size == 0:
            self.compression_semaphore_size = os.cpu_count() or 4
        if self.max_queue_depth == 0:
            self.max_queue_depth = 2 * self.compression_semaphore_size


settings = Settings()
