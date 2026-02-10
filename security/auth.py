from fastapi import Request

from config import settings
from exceptions import AuthenticationError


def authenticate(request: Request) -> bool:
    """Check if request has a valid API key.

    Args:
        request: FastAPI request object.

    Returns:
        True if authenticated, False if no auth header present.

    Raises:
        AuthenticationError: If auth header present but key is invalid.
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        return False  # No auth — treated as public request

    if not auth_header.startswith("Bearer "):
        raise AuthenticationError(
            "Invalid Authorization header format. Expected: Bearer <key>"
        )

    provided_key = auth_header[7:]  # Strip "Bearer "

    if not settings.api_key:
        # No API key configured — accept all authenticated requests (dev mode)
        return True

    if provided_key != settings.api_key:
        raise AuthenticationError("Invalid API key")

    return True
