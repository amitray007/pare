class PareError(Exception):
    """Base exception for all Pare errors."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, **kwargs):
        self.message = message
        self.details = kwargs
        super().__init__(message)


class BadRequestError(PareError):
    """Malformed request body, invalid JSON, missing required fields."""

    status_code = 400
    error_code = "bad_request"


class FileTooLargeError(PareError):
    """File exceeds maximum allowed size."""

    status_code = 413
    error_code = "file_too_large"


class UnsupportedFormatError(PareError):
    """File format not recognized via magic bytes."""

    status_code = 415
    error_code = "unsupported_format"


class OptimizationError(PareError):
    """Optimization failed (tool crash, larger output, etc.)."""

    status_code = 422
    error_code = "optimization_failed"


class SSRFError(PareError):
    """URL targets a private/reserved IP range."""

    status_code = 422
    error_code = "ssrf_blocked"


class URLFetchError(PareError):
    """Failed to fetch image from URL."""

    status_code = 422
    error_code = "url_fetch_failed"


class ToolTimeoutError(PareError):
    """Compression tool exceeded timeout."""

    status_code = 500
    error_code = "tool_timeout"


class RateLimitError(PareError):
    """Rate limit exceeded."""

    status_code = 429
    error_code = "rate_limit_exceeded"


class AuthenticationError(PareError):
    """Invalid or missing API key."""

    status_code = 401
    error_code = "unauthorized"


class BackpressureError(PareError):
    """Compression queue is full."""

    status_code = 503
    error_code = "service_overloaded"
