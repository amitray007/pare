import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from exceptions import PareError
from security.auth import authenticate
from security.rate_limiter import safe_check_rate_limit


class SecurityMiddleware(BaseHTTPMiddleware):
    """Combined middleware for auth, rate limiting, and request ID.

    Order of operations per request:
    1. Inject request ID (UUID)
    2. Authenticate (check Bearer token)
    3. Rate limit check (skip for authenticated if configured)
    4. Process request
    5. Add X-Request-ID to response
    """

    async def dispatch(self, request: Request, call_next):
        # 1. Request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        try:
            # 2. Authentication
            is_authenticated = authenticate(request)
            request.state.is_authenticated = is_authenticated

            # 3. Rate limiting
            client_ip = _get_client_ip(request)
            await safe_check_rate_limit(client_ip, is_authenticated)

            # 4. Process request
            response = await call_next(request)

        except PareError as exc:
            response = JSONResponse(
                status_code=exc.status_code,
                content={
                    "success": False,
                    "error": exc.error_code,
                    "message": exc.message,
                    **exc.details,
                },
            )

        # 5. Request ID header
        response.headers["X-Request-ID"] = request_id
        return response


def _get_client_ip(request: Request) -> str:
    """Extract client IP from X-Forwarded-For or direct connection."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
