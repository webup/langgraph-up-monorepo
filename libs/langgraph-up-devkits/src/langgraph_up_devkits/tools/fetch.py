"""HTTP fetch tool for retrieving web content."""

import asyncio
from typing import Any

from langchain_core.tools import tool


def _get_requests() -> Any:
    """Dynamically import requests to avoid hard dependency."""
    try:
        import requests

        return requests
    except ImportError:
        return None


def _get_aiohttp() -> Any:
    """Dynamically import aiohttp for async requests."""
    try:
        import aiohttp

        return aiohttp
    except ImportError:
        return None


@tool
async def fetch_url(url: str, timeout_seconds: float = 10.0) -> str:
    """Fetch text content from a URL.

    This function retrieves the text content from a given URL, with async support
    as the preferred method. Falls back to synchronous requests if aiohttp
    is not available.

    Args:
        url: The URL to fetch content from
        timeout_seconds: Request timeout in seconds (default: 10.0)

    Returns:
        Text content of the URL response
    """
    # Try async first (preferred)
    aiohttp = _get_aiohttp()
    if aiohttp:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                text_content = await response.text()
                return str(text_content)

    # Fallback to sync requests
    requests = _get_requests()
    if not requests:
        raise ImportError("HTTP fetch requires: uv add aiohttp (preferred) or uv add requests")

    # Run sync request in thread pool to avoid blocking
    loop = asyncio.get_event_loop()

    def _sync_fetch() -> str:
        response = requests.get(url, timeout=timeout_seconds)
        response.raise_for_status()
        return str(response.text)

    return await loop.run_in_executor(None, _sync_fetch)
