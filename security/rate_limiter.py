import logging
import time

from config import settings
from exceptions import RateLimitError

logger = logging.getLogger(__name__)

# Redis client initialized lazily
_redis = None

# Cumulative count of Redis errors since process start. Operators can alert on this
# counter (e.g., via log scraping or /health) rather than on the absence of rate limiting.
_rate_limit_redis_error_count = 0


def get_rate_limit_redis_error_count() -> int:
    """Return the number of Redis errors encountered by the rate limiter since process start."""
    return _rate_limit_redis_error_count


async def get_redis():
    """Get or create async Redis connection."""
    global _redis
    if _redis is None:
        import redis.asyncio as aioredis

        _redis = aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
    return _redis


async def check_rate_limit(client_ip: str, is_authenticated: bool) -> None:
    """Check if the request should be rate-limited.

    Uses a sliding window counter algorithm:
    1. INCR current minute key
    2. Read previous minute key
    3. Weighted count = prev * (1 - elapsed_fraction) + current
    4. Reject if weighted > limit

    Args:
        client_ip: Client IP address.
        is_authenticated: Whether the request has a valid API key.

    Raises:
        RateLimitError: If rate limit exceeded (429).
    """
    if is_authenticated and not settings.rate_limit_auth_enabled:
        return

    limit = settings.rate_limit_auth_rpm if is_authenticated else settings.rate_limit_public_rpm

    if limit == 0:
        return  # Unlimited

    r = await get_redis()
    now = time.time()
    current_minute = int(now // 60)
    elapsed_fraction = (now % 60) / 60

    current_key = f"rate:{client_ip}:{current_minute}"
    prev_key = f"rate:{client_ip}:{current_minute - 1}"

    pipe = r.pipeline()
    pipe.incr(current_key)
    pipe.expire(current_key, 120)
    pipe.get(prev_key)
    results = await pipe.execute()

    current_count = results[0]
    prev_count = int(results[2] or 0)

    weighted = prev_count * (1 - elapsed_fraction) + current_count

    if weighted > limit:
        raise RateLimitError(
            "Rate limit exceeded",
            retry_after=int(60 - now % 60),
            limit=limit,
        )


async def check_burst_limit(client_ip: str) -> None:
    """Check burst limit (requests in a 10-second window).

    Raises:
        RateLimitError: If burst limit exceeded.
    """
    if settings.rate_limit_public_burst == 0:
        return

    r = await get_redis()
    now = time.time()
    window = int(now // 10)
    key = f"burst:{client_ip}:{window}"

    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 20)
    results = await pipe.execute()

    if results[0] > settings.rate_limit_public_burst:
        raise RateLimitError(
            "Burst rate limit exceeded",
            retry_after=int(10 - now % 10),
            limit=settings.rate_limit_public_burst,
        )


async def safe_check_rate_limit(client_ip: str, is_authenticated: bool) -> None:
    """Rate limit check with fail-open on Redis errors.

    If Redis is unavailable, requests are allowed through to prevent
    Redis outages from taking down the service.
    """
    if not settings.redis_url:
        return  # Rate limiting disabled (no Redis configured)

    try:
        await check_rate_limit(client_ip, is_authenticated)
        await check_burst_limit(client_ip)
    except RateLimitError:
        raise  # Propagate actual rate limit hits
    # Policy: fail-open on Redis errors (preserves availability). Visibility
    # is via the error log + counter — operators page on the counter, not
    # the absence of rate limiting. To switch to fail-closed, change to
    # `raise RateLimitError(...)` here.
    except Exception as e:
        global _rate_limit_redis_error_count
        _rate_limit_redis_error_count += 1
        logger.exception(
            f"Rate limiter unavailable — fail-open allowed request: {e}",
            extra={
                "context": {
                    "redis_error_count": _rate_limit_redis_error_count,
                    "exc_type": type(e).__name__,
                }
            },
        )
