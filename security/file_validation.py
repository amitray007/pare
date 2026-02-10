from config import settings
from exceptions import FileTooLargeError
from utils.format_detect import ImageFormat, detect_format


def validate_file(data: bytes) -> ImageFormat:
    """Validate file size and format.

    Args:
        data: Raw file bytes.

    Returns:
        Detected ImageFormat.

    Raises:
        FileTooLargeError: If file exceeds max_file_size_mb.
        UnsupportedFormatError: If magic bytes don't match any known format.
    """
    if len(data) > settings.max_file_size_bytes:
        raise FileTooLargeError(
            f"File size {len(data)} bytes exceeds limit of {settings.max_file_size_mb} MB",
            file_size=len(data),
            limit=settings.max_file_size_bytes,
        )

    return detect_format(data)
