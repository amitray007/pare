import ipaddress
import socket
from urllib.parse import urlparse

from exceptions import SSRFError

# Blocked IP networks (private, loopback, link-local, unspecified)
BLOCKED_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

# Blocked hostnames (case-insensitive)
BLOCKED_HOSTNAMES = {
    "metadata.google.internal",
    "metadata.google.internal.",
}


def validate_url(url: str) -> str:
    """Validate a URL is safe to fetch (not targeting internal resources).

    Args:
        url: User-supplied URL string.

    Returns:
        The validated URL (unchanged if safe).

    Raises:
        SSRFError: If URL targets a private/reserved IP, metadata endpoint,
                   or uses a non-HTTPS scheme.
    """
    parsed = urlparse(url)

    # 1. Scheme check — HTTPS only
    if parsed.scheme != "https":
        raise SSRFError(
            f"Only HTTPS URLs are allowed, got {parsed.scheme}://",
            url=url,
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFError("URL has no hostname", url=url)

    # 2. Hostname blocklist
    if hostname.lower() in BLOCKED_HOSTNAMES:
        raise SSRFError(
            "URL targets a blocked metadata endpoint",
            url=url,
        )

    # 3. DNS resolution — resolve before any HTTP request to prevent rebinding
    try:
        addr_infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        raise SSRFError(f"Could not resolve hostname: {hostname}", url=url)

    # 4. IP validation — check every resolved address
    for family, _, _, _, sockaddr in addr_infos:
        ip = ipaddress.ip_address(sockaddr[0])
        for network in BLOCKED_NETWORKS:
            if ip in network:
                raise SSRFError(
                    "URL resolves to a private/reserved IP address",
                    url=url,
                    resolved_ip=str(ip),
                )

    return url
