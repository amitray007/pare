import asyncio
import logging

import httpx

from config import settings
from exceptions import FileTooLargeError, URLFetchError
from security.ssrf import validate_url

logger = logging.getLogger("pare.utils.url_fetch")

# Module-level shared client — created once at first call, reused across requests.
# Avoids per-request TLS handshake + connection setup overhead (~50-200ms on cold hosts).
_client: httpx.AsyncClient | None = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None:
        async with _client_lock:
            if _client is None:
                _client = httpx.AsyncClient(
                    # Per-request timeout is passed to client.stream() instead,
                    # so the client-level timeout is a wide backstop only.
                    timeout=httpx.Timeout(settings.url_fetch_timeout * 4),
                    follow_redirects=False,
                    verify=True,
                )
    return _client


async def close_client() -> None:
    """Close the shared AsyncClient. Called from FastAPI lifespan shutdown."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def fetch_image(url: str, is_authenticated: bool = False) -> bytes:
    """Fetch image from a URL with SSRF protection and streaming size limits.

    Follows redirects manually (max 5 hops) with SSRF validation at each
    hop to prevent redirect-based SSRF attacks. Streams the response body
    and aborts early if the size limit is exceeded.

    Args:
        url: User-supplied HTTPS URL.
        is_authenticated: Affects timeout (auth=base, public=2x base).

    Returns:
        Raw image bytes.

    Raises:
        SSRFError: URL targets private/reserved IP.
        URLFetchError: Fetch failed (timeout, non-2xx, redirect limit).
        FileTooLargeError: Response body exceeds max file size.
    """
    # 1. SSRF validation on initial URL
    validate_url(url)

    timeout = settings.url_fetch_timeout if is_authenticated else settings.url_fetch_timeout * 2
    max_redirects = settings.url_fetch_max_redirects
    max_size = settings.max_file_size_bytes

    client = await _get_client()
    current_url = url

    try:
        for _hop in range(max_redirects + 1):
            async with client.stream(
                "GET", current_url, timeout=httpx.Timeout(timeout)
            ) as response:
                if response.is_redirect:
                    if response.next_request is None:
                        raise URLFetchError(
                            "Redirect without Location header",
                            url=current_url,
                        )
                    redirect_url = str(response.next_request.url)
                    validate_url(redirect_url)
                    current_url = redirect_url
                    continue

                if not response.is_success:
                    raise URLFetchError(
                        f"URL returned HTTP {response.status_code}",
                        url=url,
                        http_status=response.status_code,
                    )

                # 2. Check Content-Length header for early rejection
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > max_size:
                    raise FileTooLargeError(
                        f"URL content too large: {content_length} bytes",
                        file_size=int(content_length),
                        limit=max_size,
                    )

                # 3. Stream body with running size check — abort on the first
                # chunk that pushes past the limit, avoiding full download of
                # oversized payloads before rejection.
                buf = bytearray()
                async for chunk in response.aiter_bytes():
                    buf.extend(chunk)
                    if len(buf) > max_size:
                        raise FileTooLargeError(
                            f"URL content exceeds {settings.max_file_size_mb} MB limit",
                            file_size=len(buf),
                            limit=max_size,
                        )

                return bytes(buf)

        raise URLFetchError(
            f"Too many redirects (>{max_redirects})",
            url=url,
        )

    except httpx.TimeoutException:
        raise URLFetchError(
            f"URL fetch timed out after {timeout}s",
            url=url,
        )
    except httpx.RequestError as e:
        raise URLFetchError(
            f"URL fetch failed: {e}",
            url=url,
        )


async def fetch_partial(
    url: str,
    *,
    byte_range: tuple[int, int] = (0, 8191),
    is_authenticated: bool = False,
) -> tuple[bytes, int | None]:
    """Issue a Range request for `byte_range`. Returns (partial_bytes, total_size).

    `total_size` is parsed from `Content-Range: bytes a-b/total` (206 response)
    or from `Content-Length` (200 fallback when origin doesn't honor Range).
    Returns `total_size=None` if neither header is parseable.
    Hard-caps the read at `byte_range[1] + 1` regardless of server behavior.
    SSRF-validates each redirect hop. Reuses the lifespan-pooled httpx client.

    Args:
        url: User-supplied HTTPS URL.
        byte_range: Inclusive (start, end) byte positions to request.
        is_authenticated: Affects timeout (auth=base, public=2x base).

    Returns:
        Tuple of (partial_bytes, total_size). total_size is None if undetermined.

    Raises:
        SSRFError: URL targets private/reserved IP.
        URLFetchError: Fetch failed (timeout, non-2xx non-416, redirect limit).
    """
    start, end = byte_range
    if start < 0 or end < start:
        raise ValueError(f"Invalid byte_range: start={start}, end={end}")
    max_bytes = end - start + 1

    # 1. SSRF validation on initial URL
    validate_url(url)

    timeout = settings.url_fetch_timeout if is_authenticated else settings.url_fetch_timeout * 2
    max_redirects = settings.url_fetch_max_redirects

    client = await _get_client()
    current_url = url

    try:
        for _hop in range(max_redirects + 1):
            async with client.stream(
                "GET",
                current_url,
                headers={"Range": f"bytes={start}-{end}"},
                timeout=httpx.Timeout(timeout),
            ) as response:
                if response.is_redirect:
                    if response.next_request is None:
                        raise URLFetchError(
                            "Redirect without Location header",
                            url=current_url,
                        )
                    redirect_url = str(response.next_request.url)
                    validate_url(redirect_url)
                    current_url = redirect_url
                    continue

                # 416 Range Not Satisfiable — caller handles
                if response.status_code == 416:
                    return (b"", None)

                if response.status_code == 206:
                    # Parse total size from Content-Range: bytes start-end/total
                    total_size: int | None = None
                    content_range = response.headers.get("content-range", "")
                    # Format: bytes <start>-<end>/<total>  or  bytes */<total>
                    if content_range.startswith("bytes ") and "/" in content_range:
                        try:
                            total_str = content_range.split("/", 1)[1].strip()
                            if total_str != "*":
                                total_size = int(total_str)
                        except (ValueError, IndexError):
                            total_size = None

                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = max_bytes - len(buf)
                        if remaining <= 0:
                            break
                        buf.extend(chunk[:remaining])
                        if len(buf) >= max_bytes:
                            break

                    return (bytes(buf), total_size)

                if response.status_code == 200:
                    # Origin ignored the Range header — log and fall back gracefully
                    logger.info(
                        "range_not_supported_origin: %s returned 200 instead of 206",
                        current_url,
                    )
                    # Parse total from Content-Length
                    total_size = None
                    cl_header = response.headers.get("content-length")
                    if cl_header:
                        try:
                            total_size = int(cl_header)
                        except ValueError:
                            total_size = None

                    # Read only up to max_bytes — never hold full body in memory
                    buf = bytearray()
                    async for chunk in response.aiter_bytes():
                        remaining = max_bytes - len(buf)
                        if remaining <= 0:
                            break
                        buf.extend(chunk[:remaining])
                        if len(buf) >= max_bytes:
                            break

                    return (bytes(buf), total_size)

                raise URLFetchError(
                    f"URL returned HTTP {response.status_code}",
                    url=url,
                    http_status=response.status_code,
                )

        raise URLFetchError(
            f"Too many redirects (>{max_redirects})",
            url=url,
        )

    except httpx.TimeoutException:
        raise URLFetchError(
            f"URL fetch timed out after {timeout}s",
            url=url,
        )
    except httpx.RequestError as e:
        raise URLFetchError(
            f"URL fetch failed: {e}",
            url=url,
        )
