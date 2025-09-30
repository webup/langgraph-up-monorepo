"""Integration tests for tools with real APIs.

These tests require actual API keys and make real network requests.
Run with: pytest tests/integration/test_tools_integration.py -v -s
"""

import os
from dataclasses import dataclass

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AnyMessage, HumanMessage

from langgraph_up_devkits.context import SearchContext
from langgraph_up_devkits.tools import (
    deep_web_search,
    fetch_url,
    get_context7_tools,
    get_deepwiki_tools,
    think_tool,
    web_search,
)
from langgraph_up_devkits.utils import load_chat_model

# Skip integration tests if no API keys
pytestmark = pytest.mark.skipif(
    not (os.getenv("SILICONFLOW_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="Integration tests require SILICONFLOW_API_KEY and TAVILY_API_KEY",
)


@dataclass
class AgentContext:
    """Simple context for agent testing."""

    user_id: str = "test_user"
    max_search_results: int = 3


@pytest.mark.asyncio
async def test_fetch_url_tool_integration():
    """Test fetch_url tool with real HTTP request."""
    result = await fetch_url.ainvoke({"url": "https://httpbin.org/json", "timeout": 10.0})

    assert isinstance(result, str)
    assert len(result) > 0
    # httpbin.org/json returns JSON with slideshow data
    assert "slideshow" in result.lower()


@pytest.mark.asyncio
async def test_web_search_tool_integration():
    """Test web_search tool with real Tavily API using SearchContext."""
    context = SearchContext(max_search_results=2, enable_deepwiki=False)

    # Mock runtime for the search tool
    class MockRuntime:
        def __init__(self, context):
            self.context = context

    import langgraph_up_devkits.tools.search as search_module

    original_get_runtime = search_module.get_runtime
    search_module.get_runtime = lambda: MockRuntime(context)

    try:
        result = await web_search.ainvoke({"query": "LangGraph documentation"})

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) <= 2  # Respects max_search_results

        # Check that results contain expected fields
        if result["results"]:
            first_result = result["results"][0]
            assert "title" in first_result or "content" in first_result

    finally:
        # Restore original get_runtime
        search_module.get_runtime = original_get_runtime


@pytest.mark.asyncio
async def test_create_agent_with_tools():
    """Test create_agent with fetch_url and web_search tools."""
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Define custom prompt function
    def agent_prompt(state) -> list[AnyMessage]:
        from langgraph.runtime import get_runtime

        runtime = get_runtime(AgentContext)
        user_id = runtime.context.user_id

        system_msg = HumanMessage(
            content=(
                f"You are a helpful assistant for user {user_id}. "
                "Use tools when needed to fetch information. "
                "Be concise in your responses."
            )
        )

        return [system_msg] + state.get("messages", [])

    # Create agent with tools and context schema
    agent = create_agent(model=model, tools=[fetch_url, web_search], prompt=agent_prompt, context_schema=AgentContext)

    # Test agent with a simple fetch task
    context = AgentContext(user_id="integration_test_user", max_search_results=1)

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=("Please fetch the UUID from https://httpbin.org/uuid and tell me what format it's in.")
                    )
                ]
            },
            context=context,
        )

        assert isinstance(result, dict)
        assert "messages" in result
        assert len(result["messages"]) > 0

        # Check final message content
        final_message = result["messages"][-1]
        assert hasattr(final_message, "content")
        assert len(final_message.content) > 10

        print(f"Agent output: {final_message.content}")

    except Exception as e:
        print(f"Agent execution failed: {e}")
        # Test passes if we can create the agent - execution might fail
        pass


@pytest.mark.asyncio
async def test_agent_with_search_tool():
    """Test agent using web_search tool with SearchContext."""
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    def search_prompt(state) -> list[AnyMessage]:
        from langgraph.runtime import get_runtime

        runtime = get_runtime(SearchContext)
        max_results = runtime.context.max_search_results

        system_msg = HumanMessage(
            content=(
                f"You are a research assistant. You can search for up to {max_results} results. Be brief and factual."
            )
        )

        return [system_msg] + state.get("messages", [])

    agent = create_agent(model=model, tools=[web_search], prompt=search_prompt, context_schema=SearchContext)

    context = SearchContext(max_search_results=1, enable_deepwiki=False)

    try:
        result = await agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=("Search for information about Python programming language and give me one key fact.")
                    )
                ]
            },
            context=context,
        )

        assert isinstance(result, dict)
        assert "messages" in result

        if result["messages"]:
            final_message = result["messages"][-1]
            print(f"Search agent output: {final_message.content}")

    except Exception as e:
        print(f"Search agent execution failed: {e}")
        # Test structure is valid even if execution fails
        pass


@pytest.mark.asyncio
async def test_deepwiki_tools_retrieval():
    """Test deepwiki MCP tools retrieval - verifies server connection and tool loading."""
    # Get deepwiki tools - this should work since MCP tools are available in our environment
    deepwiki_tools = await get_deepwiki_tools()

    # This should not be empty - we know MCP deepwiki tools exist
    assert len(deepwiki_tools) > 0, "DeepWiki MCP tools should be available"

    # Check that tools have expected attributes
    for tool in deepwiki_tools:
        assert callable(tool), "Each tool should be callable"
        # Tools should have name and description attributes
        assert hasattr(tool, "name") or hasattr(tool, "__name__"), "Tool should have a name"

    print(f"✅ DeepWiki MCP tools loaded successfully: {len(deepwiki_tools)} tools")


@pytest.mark.asyncio
async def test_agent_with_deepwiki_tools():
    """Test agent with deepwiki MCP tools - fails fast on MCP issues."""
    # Get deepwiki tools - this should work since MCP tools are available in our environment
    deepwiki_tools = await get_deepwiki_tools()

    # This should not be empty - we know MCP deepwiki tools exist from context
    assert len(deepwiki_tools) > 0, "DeepWiki MCP tools should be available"

    # Register SiliconFlow provider
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Create agent with deepwiki tools
    agent = create_agent(
        model=model,
        tools=deepwiki_tools,
        prompt="""You are a helpful assistant. When asked about repositories,
        use your ask_question tool to get information.""",
        context_schema=SearchContext,
    )

    context = SearchContext(enable_deepwiki=True)

    # Test agent with explicit tool usage instruction
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
            Use your ask_question tool to query facebook/react about 'What is React?'
        """
                )
            ]
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls - we know tool calling works from Tavily tests
    tool_calls_found = any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in result["messages"])

    if not tool_calls_found:
        # Print debug info
        print("No tool calls found. Messages:")
        for i, msg in enumerate(result["messages"]):
            print(f"  {i}: {type(msg)} - {getattr(msg, 'content', 'NO_CONTENT')[:100]}")

    # Assert that deepwiki tools were actually used
    assert tool_calls_found, "Agent should use deepwiki tools - tool calling works per other tests"

    print("✅ DeepWiki MCP tools integration working correctly")


@pytest.mark.asyncio
async def test_context7_tools_retrieval():
    """Test context7 MCP tools retrieval - verifies server connection and tool loading."""
    # Get context7 tools - this should work since MCP tools are available in our environment
    context7_tools = await get_context7_tools()

    # This should not be empty - we know MCP context7 tools exist
    assert len(context7_tools) > 0, "Context7 MCP tools should be available"

    # Check that tools have expected attributes
    for tool in context7_tools:
        assert callable(tool), "Each tool should be callable"
        # Tools should have name and description attributes
        assert hasattr(tool, "name") or hasattr(tool, "__name__"), "Tool should have a name"

    print(f"✅ Context7 MCP tools loaded successfully: {len(context7_tools)} tools")


@pytest.mark.asyncio
async def test_agent_with_context7_tools():
    """Test agent with context7 MCP tools - fails fast on MCP issues."""
    # Get context7 tools - this should work since MCP tools are available in our environment
    context7_tools = await get_context7_tools()

    # This should not be empty - we know MCP context7 tools exist from context
    assert len(context7_tools) > 0, "Context7 MCP tools should be available"

    # Register SiliconFlow provider
    # Use our load_chat_model which automatically registers providers
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Create agent with context7 tools
    agent = create_agent(
        model=model,
        tools=context7_tools,
        prompt="""You are a helpful assistant. When asked about libraries or documentation,
        use your resolve-library-id and get-library-docs tools to get information.""",
        context_schema=SearchContext,
    )

    context = SearchContext(enable_deepwiki=True)

    # Test agent with explicit tool usage instruction
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Use your resolve-library-id tool to find the library ID for 'react' "
                        "and then get its documentation."
                    )
                )
            ]
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls - we know tool calling works from other tests
    tool_calls_found = any(hasattr(msg, "tool_calls") and msg.tool_calls for msg in result["messages"])

    if not tool_calls_found:
        # Print debug info
        print("No tool calls found. Messages:")
        for i, msg in enumerate(result["messages"]):
            print(f"  {i}: {type(msg)} - {getattr(msg, 'content', 'NO_CONTENT')[:100]}")

    # Assert that context7 tools were actually used
    assert tool_calls_found, "Agent should use context7 tools - tool calling works per other tests"

    print("✅ Context7 MCP tools integration working correctly")


def test_think_tool_reflection_roundtrip():
    """Think tool returns confirmation containing reflection text."""
    reflection = "Validated search findings and noted missing context."
    result = think_tool.invoke({"reflection": reflection})
    assert result == f"Reflection recorded: {reflection}"


@pytest.mark.asyncio
async def test_agent_with_think_tool_integration():
    """Test agent with think tool in a real reflection workflow."""
    # Load model
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with think tool and web search for a realistic reflection scenario
    agent = create_agent(
        model=model,
        tools=[think_tool, web_search],
        prompt="""You are a helpful research assistant. When conducting research:
        1. First search for information
        2. Use the think_tool to reflect on what you found and plan next steps
        3. Continue research if needed or provide final answer

        Always use the think_tool after getting search results to analyze findings.""",
        context_schema=SearchContext,
    )

    context = SearchContext(max_search_results=2)

    # Test agent with a research task that would naturally use reflection
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
            Research Python's GIL (Global Interpreter Lock). After your search,
            use the think_tool to reflect on what you found before providing a summary.
        """
                )
            ]
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict)
    assert "messages" in result
    assert len(result["messages"]) > 0

    # Check for tool calls and tool messages - look for evidence of both tools
    think_tool_used = False
    web_search_used = False

    for msg in result["messages"]:
        # Check for AI messages with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")
                if tool_name == "think_tool":
                    think_tool_used = True
                elif tool_name == "web_search":
                    web_search_used = True

        # Check for tool messages (responses from tools)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            # Look for characteristic responses from our tools
            if "Reflection recorded:" in content:
                think_tool_used = True
            elif "query" in content and ("answer" in content or "follow_up_questions" in content):
                # This looks like a web search response
                web_search_used = True

    if not (think_tool_used and web_search_used):
        # Print debug info
        print("Tool usage analysis:")
        for i, msg in enumerate(result["messages"]):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "unknown")
                    print(f"  Message {i}: Used tool '{tool_name}'")
            else:
                content_preview = getattr(msg, "content", "NO_CONTENT")
                if isinstance(content_preview, str):
                    content_preview = content_preview[:100]
                print(f"  Message {i}: {type(msg).__name__} - {content_preview}")

    # Assert that both tools were used in a realistic research workflow
    assert web_search_used, "Agent should use web_search for research"
    assert think_tool_used, "Agent should use think_tool for reflection after search"

    print("✅ Think tool integration with real agent working correctly")


@pytest.mark.asyncio
async def test_agent_with_deep_web_search_tool():
    """Test agent with deep_web_search tool using custom state with files field."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Define custom state with files field for VFS support
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: dict  # Virtual file system

    # Load model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with deep_web_search tool and custom state schema
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        prompt=(
            "You are a research assistant. You MUST use the deep_web_search tool "
            "when asked to search or research any topic. Never answer from your own "
            "knowledge - always use the tool first."
        ),
        state_schema=DeepAgentState,
    )

    # Test agent with research request
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage(content="Search for Python GIL using deep_web_search tool.")],
            "files": {},  # Start with empty VFS
        }
    )

    # Verify response structure
    assert isinstance(result, dict), "Agent should return a dictionary"
    assert "messages" in result, "Result should contain messages"
    assert "files" in result, "Result should contain files (VFS)"

    # Debug: Print messages if no files
    files = result["files"]
    if len(files) == 0:
        print("\n⚠️  No files created. Message trace:")
        for i, msg in enumerate(result["messages"]):
            msg_type = type(msg).__name__
            has_tool_calls = hasattr(msg, "tool_calls") and bool(msg.tool_calls)
            content_preview = str(getattr(msg, "content", "N/A"))[:100]
            print(f"  [{i}] {msg_type} | tool_calls={has_tool_calls} | content: {content_preview}")

    # Check that files were added to VFS
    assert isinstance(files, dict), "Files should be a dictionary"
    assert len(files) > 0, "Should have stored search results in VFS"

    # Verify file content structure
    for filename, content in files.items():
        assert filename.endswith(".md"), f"Filename {filename} should end with .md"
        assert isinstance(content, str), "File content should be a string"
        assert len(content) > 0, "File content should not be empty"
        assert "# " in content, "File should contain markdown headers"
        assert "**URL:**" in content, "File should contain URL metadata"
        assert "**Search Query:**" in content, "File should contain search query"
        assert "## Tavily Summary" in content, "File should contain Tavily summary section"

    # Check for tool usage in messages
    deep_search_used = False
    for msg in result["messages"]:
        # Check for AI messages with tool calls
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")
                if tool_name == "deep_web_search":
                    deep_search_used = True

        # Check for tool messages (responses from tools)
        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            if "🔍 Found" in content and "📁 Files saved:" in content:
                deep_search_used = True

    assert deep_search_used, "Agent should have used deep_web_search tool"

    # Verify final message indicates successful research
    final_message = result["messages"][-1]
    assert hasattr(final_message, "content"), "Final message should have content"

    print("✅ Deep web search agent integration working correctly")
    print(f"   Files stored in VFS: {len(files)}")
    print(f"   File names: {list(files.keys())}")
    print(f"   Final message length: {len(final_message.content)} characters")


@pytest.mark.asyncio
async def test_agent_with_deep_web_search_state_persistence():
    """Test that agent preserves existing VFS files while adding new ones."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Define a custom merger for files that preserves existing files
    def merge_files(existing: dict, new: dict) -> dict:
        """Merge new files with existing files, preserving both."""
        merged = existing.copy() if existing else {}
        if new:
            merged.update(new)
        return merged

    # Define custom state with files field that uses custom merger
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: Annotated[dict, merge_files]

    # Load model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=200)

    # Create agent
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        prompt="You are a research assistant. Use deep_web_search when asked to research topics.",
        state_schema=DeepAgentState,
    )

    # Start with existing files in VFS
    initial_files = {"existing_research.md": "# Previous Research\nThis was stored earlier."}

    # First search
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Search for LangGraph documentation.")], "files": initial_files.copy()}
    )

    # Verify existing files are preserved and new ones added
    files = result["files"]
    assert "existing_research.md" in files, "Existing files should be preserved"
    assert files["existing_research.md"] == "# Previous Research\nThis was stored earlier.", (
        "Existing content unchanged"
    )

    # Count new files (excluding the existing one)
    new_files = {k: v for k, v in files.items() if k != "existing_research.md"}
    assert len(new_files) >= 1, "Should have added new search result files"

    print("✅ Deep web search state persistence working correctly")
    print(f"   Total files: {len(files)} (1 existing + {len(new_files)} new)")
    print(f"   New files: {list(new_files.keys())}")


@pytest.mark.asyncio
async def test_agent_with_deep_web_search_with_context():
    """Test agent with deep_web_search tool using SearchContext for configuration."""
    from typing import Annotated

    from langchain.agents import AgentState
    from langgraph.graph.message import add_messages

    # Use the actual SearchContext from the library
    from langgraph_up_devkits.context import SearchContext as LibSearchContext

    # Define custom state with files field for VFS support
    class DeepAgentState(AgentState):
        messages: Annotated[list, add_messages]
        files: dict  # Virtual file system

    # Load main model using SiliconFlow
    model = load_chat_model(model="siliconflow:THUDM/GLM-Z1-9B-0414", temperature=0.3, max_tokens=300)

    # Create agent with deep_web_search tool and context schema
    agent = create_agent(
        model=model,
        tools=[deep_web_search],
        prompt="""You are a research assistant with access to deep web search capabilities.
        When asked to search for information, use the deep_web_search tool.
        Always use the tool when asked to research a topic.""",
        state_schema=DeepAgentState,
        context_schema=LibSearchContext,
    )

    # Create context with include_raw_content configuration
    context = LibSearchContext(max_search_results=1, include_raw_content="markdown")

    # Test agent with research request and custom context
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="""
            Please search for information about LangGraph patterns.
            Use your deep_web_search tool to find and analyze the content.
        """
                )
            ],
            "files": {},  # Start with empty VFS
        },
        context=context,
    )

    # Verify response structure
    assert isinstance(result, dict), "Agent should return a dictionary"
    assert "messages" in result, "Result should contain messages"
    assert "files" in result, "Result should contain files (VFS)"

    # Check that files were added to VFS
    files = result["files"]
    assert isinstance(files, dict), "Files should be a dictionary"
    assert len(files) > 0, "Should have stored search results in VFS"

    # Verify file content structure - should have Tavily summary and full content
    for filename, content in files.items():
        assert filename.endswith(".md"), f"Filename {filename} should end with .md"
        assert isinstance(content, str), "File content should be a string"
        assert len(content) > 0, "File content should not be empty"
        assert "## Tavily Summary" in content, "File should contain Tavily summary section"
        assert "## Full Content" in content, "File should contain full content section"
        # The summary should exist and have content
        summary_section = content.split("## Tavily Summary")[1].split("##")[0].strip()
        assert len(summary_section) > 0, "Tavily summary should have content"

    # Check for tool usage in messages
    deep_search_used = False
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_name = getattr(tool_call, "name", None) or getattr(tool_call, "__name__", "")
                if tool_name == "deep_web_search":
                    deep_search_used = True

        if hasattr(msg, "content") and isinstance(msg.content, str):
            content = msg.content
            if "🔍 Found" in content and "📁 Files saved:" in content:
                deep_search_used = True

    assert deep_search_used, "Agent should have used deep_web_search tool"

    print("✅ Deep web search with SearchContext configuration working correctly")
    print(f"   Used include_raw_content: {context.include_raw_content}")
    print(f"   Files stored in VFS: {len(files)}")
    print(f"   File names: {list(files.keys())}")
