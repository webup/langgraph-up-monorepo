"""MCP Client setup and management for LangGraph ReAct Agent."""

import logging
from collections.abc import Callable
from typing import Any, cast

try:
    from langchain_mcp_adapters.client import (
        MultiServerMCPClient,
    )

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MultiServerMCPClient = None

logger = logging.getLogger(__name__)

# Global MCP client and tools cache
_mcp_client: MultiServerMCPClient | None = None
_mcp_tools_cache: dict[str, list[Callable[..., Any]]] = {}

# MCP Server configurations
MCP_SERVERS = {
    "deepwiki": {
        "url": "https://mcp.deepwiki.com/mcp",
        "transport": "streamable_http",
    },
    "context7": {
        "url": "https://mcp.context7.com/sse",
        "transport": "sse",
    },
    # Add more MCP servers here as needed
}


async def get_mcp_client(
    server_configs: dict[str, Any] | None = None,
) -> MultiServerMCPClient | None:
    """Get or initialize MCP client with given server configurations.

    If server_configs is provided, creates a new client for those specific servers.
    If no server_configs provided, uses the global client with all configured servers.
    """
    global _mcp_client

    # If specific server configs provided, create a dedicated client for them
    if server_configs is not None:
        try:
            client = MultiServerMCPClient(server_configs)  # pyright: ignore[reportArgumentType]
            logger.info(f"Created MCP client with servers: {list(server_configs.keys())}")
            return client
        except Exception as e:
            logger.error("Failed to create MCP client: %s", e)
            return None

    # Otherwise, use global client for all servers (backward compatibility)
    if _mcp_client is None:
        try:
            _mcp_client = MultiServerMCPClient(MCP_SERVERS)  # pyright: ignore[reportArgumentType]
            logger.info(f"Initialized global MCP client with servers: {list(MCP_SERVERS.keys())}")
        except Exception as e:
            logger.error("Failed to initialize global MCP client: %s", e)
            return None
    return _mcp_client


async def get_mcp_tools(server_name: str) -> list[Callable[..., Any]]:
    """Get MCP tools for a specific server, initializing client if needed."""
    if not MCP_AVAILABLE:
        logger.warning("MCP not available - install langchain-mcp-adapters")
        return []

    global _mcp_tools_cache

    # Return cached tools if available
    if server_name in _mcp_tools_cache:
        return _mcp_tools_cache[server_name]

    # Check if server exists in configuration
    if server_name not in MCP_SERVERS:
        logger.warning(f"MCP server '{server_name}' not found in configuration")
        _mcp_tools_cache[server_name] = []
        return []

    try:
        # Create server-specific client instead of using global singleton
        server_config = {server_name: MCP_SERVERS[server_name]}
        client = await get_mcp_client(server_config)
        if client is None:
            _mcp_tools_cache[server_name] = []
            return []

        # Get all tools from this specific server
        all_tools = await client.get_tools()
        tools = cast(list[Callable[..., Any]], all_tools)

        _mcp_tools_cache[server_name] = tools
        logger.info(f"Loaded {len(tools)} tools from MCP server '{server_name}'")
        return tools
    except Exception as e:
        logger.warning(f"Failed to load tools from MCP server '{server_name}': %s", e)
        _mcp_tools_cache[server_name] = []
        return []


async def get_deepwiki_tools() -> list[Callable[..., Any]]:
    """Get DeepWiki MCP tools."""
    return await get_mcp_tools("deepwiki")


async def get_context7_tools() -> list[Callable[..., Any]]:
    """Get Context7 MCP tools."""
    return await get_mcp_tools("context7")


async def get_all_mcp_tools() -> list[Callable[..., Any]]:
    """Get all tools from all configured MCP servers."""
    all_tools = []
    for server_name in MCP_SERVERS.keys():
        tools = await get_mcp_tools(server_name)
        all_tools.extend(tools)
    return all_tools


def add_mcp_server(name: str, config: dict[str, Any]) -> None:
    """Add a new MCP server configuration."""
    MCP_SERVERS[name] = config
    # Clear client to force reinitialization with new config
    clear_mcp_cache()


def remove_mcp_server(name: str) -> None:
    """Remove an MCP server configuration."""
    if name in MCP_SERVERS:
        del MCP_SERVERS[name]
        # Clear client to force reinitialization with new config
        clear_mcp_cache()


def clear_mcp_cache() -> None:
    """Clear the MCP client and tools cache (useful for testing)."""
    global _mcp_client, _mcp_tools_cache
    _mcp_client = None
    _mcp_tools_cache = {}
