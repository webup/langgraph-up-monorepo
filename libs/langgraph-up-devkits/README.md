# LangGraph-UP DevKits

A comprehensive development toolkit for LangGraph agents (LangChain v1) providing reusable components, middleware, context schemas, and testing utilities.

## Features

- 🎯 **Context Schemas**: Composable configuration schemas for different agent types
- 🔧 **Middleware Components**: Model provider switching with `wrap_model_call` pattern (LangChain v1)
- 🛠️ **Reusable Tools**: Context-aware tools for web search and MCP integration
- 🧪 **Testing Utilities**: Mock objects and helpers for agent testing
- 📦 **Provider Integration**: Seamless model provider support with automatic registration
- ⚡ **Async Support**: Full async/await support for all middleware and tools

## Version

**Current Version: 0.3.0** - LangChain v1 compatible

This version uses the modern LangChain v1 API with `wrap_model_call` middleware pattern.

## Installation

```bash
# Core installation
uv add langgraph-up-devkits

# With provider support
uv add langgraph-up-devkits[providers]

# With tools
uv add langgraph-up-devkits[tools]

# Complete installation
uv add langgraph-up-devkits[all]

# Development
uv add langgraph-up-devkits[dev]
```

## LangChain v1 Migration

This package is fully compatible with LangChain v1. Key changes from v0:

- ✅ Uses `wrap_model_call` and `awrap_model_call` instead of `modify_model_request`
- ✅ Imports from `langchain.agents` namespace (not `langchain_core.agents`)
- ✅ Uses `system_prompt` parameter in `create_agent` (not `prompt` for middleware)
- ✅ Provider registration uses `"openai-compatible"` for custom providers
- ✅ Full async/await support throughout

**Important Note on Provider Packages:**

Since `langchain-qwq` and `langchain-siliconflow` have not yet been upgraded to LangChain v1, we use the OpenAI-compatible fallback for these providers. This means:

- Qwen/QwQ models are accessed via OpenAI-compatible API with `DASHSCOPE_API_KEY`
- SiliconFlow models are accessed via OpenAI-compatible API with `SILICONFLOW_API_KEY`
- All functionality works seamlessly - the fallback is transparent to users
- Once these packages are upgraded to v1, we can switch to native implementations

## Quick Start

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph_up_devkits import (
    DataAnalystContext,
    ModelProviderMiddleware,
    web_search,
    create_context_aware_prompt
)

# Create agent using devkit components (LangChain v1)
agent = create_agent(
    model="openai:gpt-4o-mini",  # Fallback model - middleware will switch
    tools=[web_search],
    system_prompt=create_context_aware_prompt,  # v1: use system_prompt with middleware
    context_schema=DataAnalystContext,
    middleware=[ModelProviderMiddleware()]
)

# Use with context - middleware will switch to the specified model
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Analyze current market trends")]},
    context=DataAnalystContext(
        model="siliconflow:THUDM/glm-4-9b-chat",  # Middleware switches to this
        max_search_results=10,
        max_data_rows=5000
    )
)
```

## Components

### Context Schemas

- `BaseAgentContext` - Core configuration
- `SearchContext` - Search capabilities
- `DataContext` - Data analysis features
- `DataAnalystContext` - Composed context for data analysts
- `ResearchContext` - Research assistant configuration

### Middleware

- `ModelProviderMiddleware` - Automatic model provider switching with `wrap_model_call` and `awrap_model_call` (LangChain v1)
- `FileSystemMaskMiddleware` - Shadows virtual file system from model context

### Tools

- `web_search` - Context-aware web search using Tavily
- `deep_web_search` - Advanced web search with depth control and context awareness
- `think_tool` - Strategic reflection and planning tool for agents
- `fetch_url` - HTTP content fetching with async support
- `get_deepwiki_tools` - Built-in MCP integration for GitHub repository documentation
- `get_context7_tools` - Built-in MCP integration for library documentation
- `get_mcp_tools()` - Get tools from any configured MCP server
- `add_mcp_server()` - Add new MCP servers dynamically

### Utilities

- `create_context_aware_prompt` - Dynamic prompt generation
- Provider registration utilities
- Testing helpers and mocks

## Examples

### Basic Context Usage

```python
from langgraph_up_devkits.context import BaseAgentContext, SearchContext

# Basic agent context
basic_context = BaseAgentContext(
    model="openrouter:openai/gpt-4o",
    user_id="user_123"
)

# Search-enabled context
search_context = SearchContext(
    model="siliconflow:THUDM/glm-4-9b-chat",
    max_search_results=8,
    enable_deepwiki=True,
    user_id="researcher_001"
)
```

### Middleware for Model Switching (LangChain v1)

The `ModelProviderMiddleware` uses the modern `wrap_model_call` pattern from LangChain v1, supporting both sync and async operations:

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph_up_devkits.middleware import ModelProviderMiddleware
from dataclasses import dataclass

@dataclass
class ModelTestContext:
    model: str
    user_id: str = "test_user"

# Example 1: Fallback model with middleware switching
middleware = ModelProviderMiddleware()
fallback_model = init_chat_model("openai:gpt-4o-mini")

agent = create_agent(
    model=fallback_model,  # Fallback model
    tools=[],
    system_prompt="You are a helpful assistant.",
    context_schema=ModelTestContext,
    middleware=[middleware]
)

# Context specifies SiliconFlow GLM model - middleware will switch automatically
context = ModelTestContext(model="siliconflow:THUDM/glm-4-9b-chat")
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Hello")]},
    context=context
)

# Example 2: Direct SiliconFlow model in create_agent
agent_direct = create_agent(
    model="siliconflow:THUDM/glm-4-9b-chat",  # Direct model specification
    tools=[],
    system_prompt="You are a helpful assistant.",
    context_schema=ModelTestContext,
    middleware=[middleware]  # Still useful for context-based switching
)
```

**Key Features:**
- ✅ Implements both `wrap_model_call` (sync) and `awrap_model_call` (async)
- ✅ Automatic provider registration (OpenRouter, Qwen, SiliconFlow)
- ✅ Context-based model switching at runtime
- ✅ Graceful fallback on errors

### FileSystemMaskMiddleware for Virtual File Systems

The `FileSystemMaskMiddleware` automatically shadows the `files` field from the agent state before passing to the model, then restores it after the model execution. This is useful when you have virtual file systems in your state that should not be sent to the LLM.

```python
from typing import Annotated
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, add_messages
from langgraph_up_devkits.middleware import FileSystemMaskMiddleware

# Define a middleware to extend state with files field
class FilesStateMiddleware(AgentMiddleware[AgentState]):
    """Middleware that extends state with files field."""

    class FilesState(AgentState):  # type: ignore[type-arg]
        """State with files field for virtual file system."""
        messages: Annotated[list, add_messages]
        files: dict  # Virtual file system - will be masked from model

    state_schema = FilesState

# Create both middlewares
files_state_middleware = FilesStateMiddleware()
filesystem_mask_middleware = FileSystemMaskMiddleware()

# Create agent with both middlewares
agent = create_agent(
    model="siliconflow:THUDM/glm-4-9b-chat",
    tools=[],
    system_prompt="You are a helpful assistant.",
    middleware=[files_state_middleware, filesystem_mask_middleware]
)

# Use with files in state - files are masked from model but preserved in state
result = await agent.ainvoke({
    "messages": [HumanMessage(content="Hello")],
    "files": {
        "config.json": '{"setting": "value"}',
        "data.csv": "name,age\\nAlice,30\\nBob,25"
    }
})

# Files are restored in the result
assert "files" in result
assert result["files"]["config.json"] == '{"setting": "value"}'
```

**Key Features:**
- **Automatic masking**: Removes `files` field before model execution
- **Automatic restoration**: Restores original `files` after model execution
- **Stateless**: Each middleware instance can handle sequential invocations
- **Composable**: Works with other middlewares via middleware chaining
- **Use case**: Prevents large file contents from consuming model context window

### Context-Aware Tools

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph_up_devkits.context import SearchContext
from langgraph_up_devkits.tools import web_search, fetch_url, get_deepwiki_tools, get_context7_tools

# Create search context with specific settings
search_context = SearchContext(
    model="openrouter:anthropic/claude-sonnet-4",
    max_search_results=5,
    enable_deepwiki=True
)

# Get built-in MCP tools
deepwiki_tools = await get_deepwiki_tools()  # GitHub repository docs
context7_tools = await get_context7_tools()  # Library documentation

# Create agent with context-aware tools
agent = create_agent(
    model="openai:gpt-4o",
    tools=[web_search, fetch_url] + deepwiki_tools + context7_tools,
    system_prompt="You are a research assistant with web search, GitHub documentation, and library documentation access.",
    context_schema=SearchContext
)

# Tools automatically respect context limits
result = await agent.ainvoke(
    {"messages": [HumanMessage(content="Research React.js documentation and latest developments using both GitHub and library docs")]},
    context=search_context  # web_search will return max 5 results
)
```

### Complete Research Assistant Example

```python
import os
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph_up_devkits import (
    ResearchContext,
    ModelProviderMiddleware,
    web_search,
    create_context_aware_prompt
)
# Set up environment (ensure API keys are available)
os.environ["TAVILY_API_KEY"] = "your_tavily_key"
os.environ["SILICONFLOW_API_KEY"] = "your_siliconflow_key"

# Provider registration is automatic - no manual setup needed!

# Create research assistant
research_agent = create_agent(
    model="openai:gpt-4o-mini",  # Fallback model
    tools=[web_search],
    system_prompt=create_context_aware_prompt,  # Context-aware prompting
    context_schema=ResearchContext,
    middleware=[ModelProviderMiddleware()]  # Automatic model switching
)

# Configure research context
research_context = ResearchContext(
    model="siliconflow:THUDM/glm-4-9b-chat",  # Switch to SiliconFlow GLM
    max_search_results=10,
    enable_deepwiki=True,
    user_id="researcher_001",
    thread_id="research_session_123"
)

# Perform research task
result = await research_agent.ainvoke(
    {"messages": [HumanMessage(content="""
        Research the current state of large language models in 2024.
        Focus on recent developments, benchmark results, and industry adoption.
        Provide a comprehensive analysis with sources.
    """)]},
    context=research_context
)

print(f"Research Result: {result['messages'][-1].content}")
```

## MCP Integration

The devkits provide seamless integration with Model Context Protocol (MCP) servers. **DeepWiki** and **Context7** are included as built-in servers, and you can easily add more servers.

### Built-in Servers

#### DeepWiki (GitHub Repositories)

DeepWiki provides documentation and insights for GitHub repositories:

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph_up_devkits.tools import get_deepwiki_tools

# Get built-in DeepWiki tools
deepwiki_tools = await get_deepwiki_tools()

agent = create_agent(
    model="openai:gpt-4o",
    tools=deepwiki_tools,
    system_prompt="You are a helpful assistant with access to GitHub repository documentation."
)

# Query GitHub repository information
result = await agent.ainvoke({
    "messages": [HumanMessage(content="What is React? Use your ask_question tool to query facebook/react.")]
})
```

#### Context7 (Library Documentation)

Context7 provides up-to-date documentation for popular libraries and frameworks:

```python
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph_up_devkits.tools import get_context7_tools

# Get built-in Context7 tools
context7_tools = await get_context7_tools()

agent = create_agent(
    model="openai:gpt-4o",
    tools=context7_tools,
    system_prompt="You are a helpful assistant with access to library documentation."
)

# Query library documentation
result = await agent.ainvoke({
    "messages": [HumanMessage(content="How do I use React hooks? Use resolve-library-id and get-library-docs to find React documentation.")]
})
```

### Adding New MCP Servers

You can easily add new MCP servers to extend functionality:

```python
from langgraph_up_devkits.tools import add_mcp_server, get_mcp_tools

# Add your own custom MCP server
add_mcp_server("my_custom_server", {
    "url": "https://my-mcp-server.com/mcp",
    "transport": "streamable_http"
})

custom_tools = await get_mcp_tools("my_custom_server")
```

### Combined MCP Usage

```python
from langgraph_up_devkits.tools import get_deepwiki_tools, get_context7_tools, get_mcp_tools, add_mcp_server

# Add additional custom servers
add_mcp_server("my_custom_server", {
    "url": "https://my-custom-mcp.com/api",
    "transport": "streamable_http"
})

# Get tools from multiple servers
deepwiki_tools = await get_deepwiki_tools()        # Built-in GitHub docs
context7_tools = await get_context7_tools()        # Built-in library docs
custom_tools = await get_mcp_tools("my_custom_server")  # Custom server

all_tools = deepwiki_tools + context7_tools + custom_tools

comprehensive_agent = create_agent(
    model="openai:gpt-4o",
    tools=all_tools,
    system_prompt="""You are a comprehensive research assistant with access to:
    - GitHub repository documentation (via ask_question for repos)
    - Library documentation (via resolve-library-id and get-library-docs)
    - Custom MCP server tools
    Use the appropriate tools based on the user's needs."""
)
```

### Available MCP Functions

```python
from langgraph_up_devkits.tools import (
    get_deepwiki_tools,    # Built-in GitHub repository tools
    get_context7_tools,    # Built-in library documentation tools
    get_mcp_tools,         # Get tools from any server by name
    get_all_mcp_tools,     # Get tools from all configured servers
    add_mcp_server,        # Add new MCP server configuration
    remove_mcp_server,     # Remove MCP server configuration
    clear_mcp_cache        # Clear cached tools (useful for development)
)
```

## Supported Providers

Automatic provider registration (no manual setup required):

- **OpenRouter**: `openrouter:openai/gpt-4o`, `openrouter:anthropic/claude-sonnet-4`
  - Requires: `OPENROUTER_API_KEY` in environment or `.env` file
- **Qwen**: `qwen:qwen-flash`, `qwq:qwq-32b-preview`
  - Requires: `DASHSCOPE_API_KEY` in environment or `.env` file
- **SiliconFlow**: `siliconflow:THUDM/glm-4-9b-chat`, `siliconflow:Qwen/Qwen3-8B`
  - Requires: `SILICONFLOW_API_KEY` in environment or `.env` file
- Any provider supported by `init_chat_model`

### Standalone Usage of `load_chat_model`

You can also use our `load_chat_model` utility directly for standalone model loading:

```python
from langgraph_up_devkits import load_chat_model

# Automatic provider registration and model loading
model = load_chat_model("siliconflow:THUDM/glm-4-9b-chat")

# Use the model directly
response = await model.ainvoke("Hello, how are you?")
print(response.content)

# Load with configuration parameters
configured_model = load_chat_model(
    model="openrouter:anthropic/claude-sonnet-4",
    temperature=0.7,
    max_tokens=1000
)

# Load different providers seamlessly
qwen_model = load_chat_model("qwen:qwen-flash")
glm_model = load_chat_model("siliconflow:THUDM/glm-4-9b-chat")
openrouter_model = load_chat_model("openrouter:openai/gpt-4o")
```

**Key Benefits:**
- **Zero setup**: No manual provider registration required
- **Smart registration**: Only registers the provider you actually need
- **Efficient**: Minimal overhead, fast loading
- **Simple API**: Clean interface with no confusing parameters

### Environment Setup

Create a `.env` file in your project root:

```bash
# Required for web search functionality
TAVILY_API_KEY=your_tavily_api_key

# Provider API keys (add as needed)
OPENROUTER_API_KEY=your_openrouter_api_key
DASHSCOPE_API_KEY=your_qwen_dashscope_api_key
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

## Testing

The library includes comprehensive testing utilities:

```python
from langgraph_up_devkits.testing import AgentTestBuilder, MockRuntime

# Test your agents easily
result = await (AgentTestBuilder()
    .with_context(DataAnalystContext())
    .with_messages(["Analyze quarterly sales"])
    .run_test(agent))
```

## Development

```bash
# Install in development mode
uv add -e .[dev]

# Run all tests (recommended)
make test

# Run unit tests only
uv run pytest tests/unit/

# Run integration tests only (requires API keys)
uv run pytest tests/integration/

# Lint and format
make lint
make format

# Individual commands
uv run pytest --cov=src                      # With coverage
uv run ruff check . && uv run ruff format .  # Lint and format
uv run mypy src/                              # Type checking
```

### Environment Setup for Integration Tests

```bash
# Required for integration tests
export TAVILY_API_KEY="your_tavily_api_key"
export SILICONFLOW_API_KEY="your_siliconflow_api_key"

# Optional for enhanced testing
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

## License

MIT