import httpx

from config import settings
from exceptions import FileTooLargeError, URLFetchError
from security.ssrf import validate_url


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

    timeout = (
        settings.url_fetch_timeout
        if is_authenticated
        else settings.url_fetch_timeout * 2
    )
    max_redirects = settings.url_fetch_max_redirects

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            verify=True,
        ) as client:
            current_url = url

            # 2. Manual redirect following with SSRF check at each hop
            for _hop in range(max_redirects + 1):
                response = await client.get(current_url)

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

                # 3. Check Content-Length header for early rejection
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > settings.max_file_size_bytes:
                    raise FileTooLargeError(
                        f"URL content too large: {content_length} bytes",
                        file_size=int(content_length),
                        limit=settings.max_file_size_bytes,
                    )

                # 4. Validate actual body size
                data = response.content
                if len(data) > settings.max_file_size_bytes:
                    raise FileTooLargeError(
                        f"URL content exceeds {settings.max_file_size_mb} MB limit",
                        file_size=len(data),
                        limit=settings.max_file_size_bytes,
                    )

                return data

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
