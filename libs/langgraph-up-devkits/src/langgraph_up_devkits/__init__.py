"""LangGraph Up Development Kits.

A comprehensive development toolkit for LangGraph agents providing:
- Context schemas for agent configuration
- Middleware components for behavior modification
- Reusable tools with context awareness
- Model provider integrations via langchain-dev-utils
- Development utilities and testing helpers
"""

from .context import (
    BaseAgentContext,
    DataAnalystContext,
    DataContext,
    ResearchContext,
    SearchContext,
)
from .middleware import ModelProviderMiddleware, SummarizationMiddleware
from .prompts import (
    DATA_ANALYST_PROMPT,
    RESEARCH_ASSISTANT_PROMPT,
    SYSTEM_PROMPT,
)
from .tools import (
    MCP_SERVERS,
    add_mcp_server,
    clear_mcp_cache,
    fetch_url,
    get_all_mcp_tools,
    get_context7_tools,
    get_deepwiki_tools,
    get_mcp_client,
    get_mcp_tools,
    remove_mcp_server,
    web_search,
)
from .utils import (
    AVAILABLE_PROVIDERS,
    create_context_aware_prompt,
    load_chat_model,
)

# Provider registration is handled automatically by load_chat_model()
# No manual registration needed - just use load_chat_model("provider:model")

__version__ = "0.1.0"
__all__ = [
    # Context schemas
    "BaseAgentContext",
    "SearchContext",
    "DataContext",
    "DataAnalystContext",
    "ResearchContext",
    # Middleware
    "ModelProviderMiddleware",
    "SummarizationMiddleware",
    # HTTP tools
    "fetch_url",
    "web_search",
    # MCP tools
    "get_context7_tools",
    "get_deepwiki_tools",
    "get_mcp_tools",
    "get_all_mcp_tools",
    "get_mcp_client",
    # MCP configuration
    "MCP_SERVERS",
    "add_mcp_server",
    "remove_mcp_server",
    "clear_mcp_cache",
    # Prompts
    "create_context_aware_prompt",
    "SYSTEM_PROMPT",
    "DATA_ANALYST_PROMPT",
    "RESEARCH_ASSISTANT_PROMPT",
    # Provider registration
    "load_chat_model",
    "AVAILABLE_PROVIDERS",
]
