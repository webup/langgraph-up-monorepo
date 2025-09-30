"""Web search tools with context awareness and VFS support."""

import base64
import os
import re
import uuid
from datetime import datetime
from typing import Annotated, Any, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.runtime import get_runtime
from langgraph.types import Command

try:
    # Try new location first (LangGraph 0.2+)
    from langgraph.prebuilt.tool_node import InjectedState
except ImportError:
    try:
        # Fallback to old location for compatibility
        from langchain.agents.tool_node import InjectedState  # type: ignore[import-not-found,no-redef]
    except ImportError as e:
        raise ImportError(
            "InjectedState is required for deep_web_search tool. "
            "Please update LangChain and LangGraph to compatible versions."
        ) from e

try:
    from langchain_core.tools import InjectedToolCallId
except ImportError as e:
    raise ImportError(
        "InjectedToolCallId is required for deep_web_search tool. Please update LangChain to a compatible version."
    ) from e


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


def _get_today_str() -> str:
    """Get current date in human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


async def _process_search_result(result: dict[str, Any], query: str) -> dict[str, Any]:
    """Process single search result using Tavily's content and raw_content."""
    url = result.get("url", "")
    title = result.get("title", "Untitled")
    tavily_summary = result.get("content", "")  # This is the AI-generated summary from Tavily
    raw_content = result.get("raw_content", "")  # This is the full markdown content

    # Create a simple filename from title
    safe_title = re.sub(r"[^\w\s-]", "", title)[:50].replace(" ", "_")
    filename = f"{safe_title}.md"

    # Make filename unique
    uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
    name, ext = os.path.splitext(filename)
    unique_filename = f"{name}_{uid}{ext}"

    return {
        "url": url,
        "title": title,
        "summary": tavily_summary,  # Use Tavily's AI-generated summary
        "filename": unique_filename,
        "raw_content": raw_content,  # Use raw_content as the full content
        "query": query,
    }


def _create_file_content(processed_result: dict[str, Any]) -> str:
    """Create comprehensive file content for VFS storage."""
    return f"""# {processed_result["title"]}

**URL:** {processed_result["url"]}
**Search Query:** {processed_result["query"]}
**Date:** {_get_today_str()}

## Tavily Summary
{processed_result["summary"]}

## Full Content
{processed_result["raw_content"] or "Content not available"}
"""


async def _run_tavily_search(query: str, max_results: int = 1) -> dict[str, Any]:
    """Execute Tavily search with enhanced content."""
    TavilyClient = _get_tavily_client()
    if not TavilyClient:
        raise ImportError("Tavily search requires: uv add tavily-python")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is required")

    client = TavilyClient(api_key=api_key)

    # Perform the search with configurable raw content
    import asyncio

    loop = asyncio.get_event_loop()

    def _sync_search() -> Any:
        # Get include_raw_content from context
        include_raw_content_value = "markdown"  # Default value
        try:
            runtime = get_runtime()
            include_raw_content_value = getattr(runtime.context, "include_raw_content", "markdown")
        except Exception:
            pass  # Use default if runtime not available

        search_params = {"query": query, "max_results": max_results, "search_depth": "advanced"}

        # Only add include_raw_content if not "none"
        if include_raw_content_value != "none":
            search_params["include_raw_content"] = include_raw_content_value

        return client.search(**search_params)

    results = await loop.run_in_executor(None, _sync_search)
    return cast(dict[str, Any], results)


@tool(parse_docstring=True)
async def deep_web_search(
    query: str,
    max_results: int | None = None,
    state: Annotated[dict[str, Any] | None, InjectedState] = None,
    tool_call_id: Annotated[str | None, InjectedToolCallId] = None,
) -> Command[Any]:
    """Enhanced web search with VFS integration.

    Uses Tavily's advanced search with AI-generated summaries and full markdown content.
    Stores full results in virtual file system for detailed reference.

    Args:
        query: Search query to execute
        max_results: Maximum number of results (default: uses context.max_search_results or 1)
        state: Injected agent state for file storage (auto-injected)
        tool_call_id: Injected tool call identifier (auto-injected)

    Returns:
        Command that saves full results to files and provides minimal summary
    """
    # Get max_results from context if not provided
    if max_results is None:
        try:
            runtime = get_runtime()
            max_results = getattr(runtime.context, "max_search_results", 1)
        except Exception:
            max_results = 1  # Final fallback

    # Execute Tavily search
    search_results = await _run_tavily_search(query, max_results)

    # Process each result
    processed_results = []
    for result in search_results.get("results", []):
        processed = await _process_search_result(result, query)
        processed_results.append(processed)

    # Handle state injection fallback - get from runtime if injection failed
    if state is None:
        try:
            runtime = get_runtime()
            # Try to get the actual state from runtime
            state = getattr(runtime, "state", {})
        except Exception:
            # If runtime is not available, create empty state
            state = {}

    # Update files in state
    files = state.get("files", {})
    saved_files = []
    summaries = []

    for processed in processed_results:
        filename = processed["filename"]
        file_content = _create_file_content(processed)

        files[filename] = file_content
        saved_files.append(filename)
        summaries.append(f"- {filename}: {processed['summary']}")

    # Create minimal response
    summary_text = f"""🔍 Found {len(processed_results)} result(s) for '{query}':

{chr(10).join(summaries)}

📁 Files saved: {", ".join(saved_files)}
💡 Access full details with file system tools"""

    # Handle tool_call_id fallback
    if tool_call_id is None:
        # Generate a fallback tool call ID
        import uuid

        tool_call_id = f"deep_search_{uuid.uuid4().hex[:8]}"

    return Command(update={"files": files, "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)]})
