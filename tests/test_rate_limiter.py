"""Tests for rate_limiter module — Redis-backed sliding window + burst."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from exceptions import RateLimitError


def _make_redis_mock(pipe_execute_result):
    """Create a properly-mocked async Redis with pipeline support.

    pipeline() is sync, pipeline methods (incr/expire/get) are sync,
    execute() is async.
    """
    mock_pipe = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=pipe_execute_result)

    mock_redis = MagicMock()
    mock_redis.pipeline.return_value = mock_pipe

    return mock_redis


# --- safe_check_rate_limit ---


@pytest.mark.asyncio
async def test_safe_check_no_redis_url():
    """No redis_url configured -> skip rate limiting."""
    from security.rate_limiter import safe_check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = ""
        await safe_check_rate_limit("1.2.3.4", False)


@pytest.mark.asyncio
async def test_safe_check_rate_limit_propagates():
    """RateLimitError propagated from check_rate_limit."""
    from security.rate_limiter import safe_check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = "redis://localhost"
        with patch(
            "security.rate_limiter.check_rate_limit",
            side_effect=RateLimitError("exceeded", retry_after=30, limit=60),
        ):
            with pytest.raises(RateLimitError):
                await safe_check_rate_limit("1.2.3.4", False)


@pytest.mark.asyncio
async def test_safe_check_redis_error_failopen():
    """Redis connection error -> fail open (no exception)."""
    from security.rate_limiter import safe_check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = "redis://localhost"
        with patch(
            "security.rate_limiter.check_rate_limit", side_effect=ConnectionError("redis down")
        ):
            await safe_check_rate_limit("1.2.3.4", False)


# --- check_rate_limit ---


@pytest.mark.asyncio
async def test_check_rate_limit_authenticated_disabled():
    """Authenticated + auth rate limiting disabled -> skip."""
    from security.rate_limiter import check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_auth_enabled = False
        await check_rate_limit("1.2.3.4", is_authenticated=True)


@pytest.mark.asyncio
async def test_check_rate_limit_zero_limit():
    """Limit=0 means unlimited."""
    from security.rate_limiter import check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_auth_enabled = True
        mock_settings.rate_limit_auth_rpm = 0
        await check_rate_limit("1.2.3.4", is_authenticated=True)


@pytest.mark.asyncio
async def test_check_rate_limit_under_limit():
    """Under the limit -> no error."""
    from security import rate_limiter

    mock_redis = _make_redis_mock([1, True, "0"])

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_auth_enabled = True
        mock_settings.rate_limit_public_rpm = 60
        mock_settings.rate_limit_auth_rpm = 120
        with patch.object(rate_limiter, "get_redis", new=AsyncMock(return_value=mock_redis)):
            await rate_limiter.check_rate_limit("1.2.3.4", is_authenticated=False)


@pytest.mark.asyncio
async def test_check_rate_limit_over_limit():
    """Over the limit -> RateLimitError."""
    from security import rate_limiter

    mock_redis = _make_redis_mock([100, True, "100"])

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_auth_enabled = True
        mock_settings.rate_limit_public_rpm = 60
        mock_settings.rate_limit_auth_rpm = 120
        with patch.object(rate_limiter, "get_redis", new=AsyncMock(return_value=mock_redis)):
            with pytest.raises(RateLimitError):
                await rate_limiter.check_rate_limit("1.2.3.4", is_authenticated=False)


# --- check_burst_limit ---


@pytest.mark.asyncio
async def test_burst_limit_zero_disabled():
    """burst=0 -> disabled."""
    from security.rate_limiter import check_burst_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_public_burst = 0
        await check_burst_limit("1.2.3.4")


@pytest.mark.asyncio
async def test_burst_limit_under():
    """Under burst limit -> no error."""
    from security import rate_limiter

    mock_redis = _make_redis_mock([1, True])

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_public_burst = 10
        with patch.object(rate_limiter, "get_redis", new=AsyncMock(return_value=mock_redis)):
            await rate_limiter.check_burst_limit("1.2.3.4")


@pytest.mark.asyncio
async def test_burst_limit_over():
    """Over burst limit -> RateLimitError."""
    from security import rate_limiter

    mock_redis = _make_redis_mock([20, True])

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.rate_limit_public_burst = 10
        with patch.object(rate_limiter, "get_redis", new=AsyncMock(return_value=mock_redis)):
            with pytest.raises(RateLimitError):
                await rate_limiter.check_burst_limit("1.2.3.4")


# --- get_redis ---


@pytest.mark.asyncio
async def test_get_redis_creates_client():
    """get_redis creates a redis client lazily."""
    from security import rate_limiter

    rate_limiter._redis = None
    mock_redis = MagicMock()
    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = "redis://localhost:6379"
        with patch("redis.asyncio.from_url", return_value=mock_redis) as mock_from_url:
            result = await rate_limiter.get_redis()
            assert result is mock_redis
            mock_from_url.assert_called_once()
    rate_limiter._redis = None


# --- safe_check_rate_limit burst call ---


@pytest.mark.asyncio
async def test_rate_limiter_burst_check():
    """Cover safe_check_rate_limit calling check_burst_limit."""
    from security.rate_limiter import safe_check_rate_limit

    with patch("security.rate_limiter.settings") as mock_settings:
        mock_settings.redis_url = "redis://localhost:6379"
        with patch("security.rate_limiter.check_rate_limit", new_callable=AsyncMock):
            with patch(
                "security.rate_limiter.check_burst_limit", new_callable=AsyncMock
            ) as mock_bl:
                await safe_check_rate_limit("1.2.3.4", False)
                mock_bl.assert_awaited_once_with("1.2.3.4")
