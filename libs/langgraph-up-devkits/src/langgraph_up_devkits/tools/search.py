"""Web search tool with context awareness."""

import os
from typing import Any, cast

from langchain_core.tools import tool
from langgraph.runtime import get_runtime


def _get_tavily_client() -> Any:
    """Dynamically import Tavily client to avoid hard dependency."""
    try:
        import importlib

        tavily_module = importlib.import_module("tavily")
        return getattr(tavily_module, "TavilyClient", None)
    except (ImportError, AttributeError, ModuleNotFoundError):
        return None


@tool
async def web_search(query: str) -> dict[str, Any]:
    """Search for general web results using Tavily.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.

    Args:
        query: The search query string

    Returns:
        Dictionary containing search results

    Raises:
        ImportError: If tavily-python is not installed
        ValueError: If TAVILY_API_KEY is not set
    """
    TavilyClient = _get_tavily_client()
    if not TavilyClient:
        raise ImportError("Tavily search requires: uv add tavily-python")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")

    runtime = get_runtime()

    # Get max_search_results from context, with fallback
    max_results = getattr(runtime.context, "max_search_results", 5)

    client = TavilyClient(api_key=api_key)

    # Perform the search (sync method, run in thread pool for async behavior)
    import asyncio

    loop = asyncio.get_event_loop()

    def _sync_search() -> Any:
        return client.search(query=query, max_results=max_results)

    results = await loop.run_in_executor(None, _sync_search)

    return cast(dict[str, Any], results)
