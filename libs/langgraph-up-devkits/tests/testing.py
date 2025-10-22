"""Comprehensive testing utilities for LangGraph agents."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage

from langgraph_up_devkits.context import (
    BaseAgentContext,
)

# ===== MOCK OBJECTS =====


class MockRuntime:
    """Mock LangGraph runtime for testing."""

    def __init__(self, context: Any = None, store: Any = None):
        """Initialize mock runtime.

        Args:
            context: Context object to return
            store: Store object to return
        """
        self.context = context or Mock()
        self.store = store or Mock()
        self.stream_writer = Mock()


class MockChatModel:
    """Mock chat model for testing."""

    def __init__(self, responses: list[str] | None = None):
        """Initialize mock chat model.

        Args:
            responses: List of response strings to cycle through
        """
        self.responses = responses or ["Mock response"]
        self.call_count = 0
        self.bound_tools: list[BaseTool] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Mock async invoke."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        """Mock sync invoke."""
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return AIMessage(content=response)

    def bind_tools(self, tools: list[BaseTool]):
        """Mock tool binding."""
        bound_model = MockChatModel(self.responses)
        bound_model.bound_tools = tools
        return bound_model


class MockTool:
    """Mock tool for testing."""

    def __init__(self, name: str, return_value: Any = "Mock tool result"):
        """Initialize mock tool.

        Args:
            name: Tool name
            return_value: Value to return when invoked
        """
        self.name = name
        self.return_value = return_value
        self.call_count = 0
        self.last_args = None

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        """Mock async tool invocation."""
        self.call_count += 1
        self.last_args = args
        return self.return_value

    def invoke(self, args: dict[str, Any]) -> Any:
        """Mock sync tool invocation."""
        self.call_count += 1
        self.last_args = args
        return self.return_value


class MockMiddleware:
    """Mock middleware for testing middleware chains."""

    def __init__(self, name: str):
        """Initialize mock middleware.

        Args:
            name: Middleware name for identification
        """
        self.name = name
        self.before_model_calls: list[Any] = []
        self.wrap_model_call_calls: list[tuple] = []
        self.after_model_calls: list[Any] = []

    def before_model(self, state):
        """Mock before_model hook."""
        self.before_model_calls.append(state)
        return None

    def wrap_model_call(self, request, handler):
        """Mock wrap_model_call hook."""
        self.wrap_model_call_calls.append((request, handler))
        return handler(request)

    def after_model(self, state):
        """Mock after_model hook."""
        self.after_model_calls.append(state)
        return None


# ===== TEST HELPERS =====


def create_test_messages(texts: list[str]) -> list[BaseMessage]:
    """Create test messages from text strings.

    Args:
        texts: List of text strings to convert to messages

    Returns:
        List of HumanMessage objects
    """
    return [HumanMessage(content=text) for text in texts]


async def run_agent_test(
    agent,
    messages: list[BaseMessage],
    context: BaseAgentContext,
    expected_response_contains: str | None = None,
) -> dict[str, Any]:
    """Helper to run agent with test context and assertions.

    Args:
        agent: Agent to test
        messages: Input messages
        context: Agent context
        expected_response_contains: Optional string to check in response

    Returns:
        Agent execution result
    """
    result = await agent.ainvoke({"messages": messages}, context=context)

    if expected_response_contains:
        last_message = result["messages"][-1]
        assert expected_response_contains in last_message.content

    return result


def mock_runtime_context(context: BaseAgentContext):
    """Context manager to mock get_runtime with specific context.

    Args:
        context: Context object to return from get_runtime

    Returns:
        Mock patch context manager
    """
    mock_runtime = MockRuntime(context=context)
    return patch("langgraph.runtime.get_runtime", return_value=mock_runtime)


class AgentTestBuilder:
    """Builder pattern for creating agent tests."""

    def __init__(self):
        """Initialize test builder."""
        self.context = None
        self.messages = []
        self.expected_tools = []
        self.mock_responses = ["Test response"]

    def with_context(self, context: BaseAgentContext):
        """Set test context.

        Args:
            context: Agent context to use

        Returns:
            Self for method chaining
        """
        self.context = context
        return self

    def with_messages(self, texts: list[str]):
        """Set test messages.

        Args:
            texts: List of message text strings

        Returns:
            Self for method chaining
        """
        self.messages = create_test_messages(texts)
        return self

    def with_mock_responses(self, responses: list[str]):
        """Set mock model responses.

        Args:
            responses: List of response strings

        Returns:
            Self for method chaining
        """
        self.mock_responses = responses
        return self

    def expect_tool_calls(self, tool_names: list[str]):
        """Expect specific tools to be called.

        Args:
            tool_names: List of expected tool names

        Returns:
            Self for method chaining
        """
        self.expected_tools = tool_names
        return self

    async def run_test(self, agent) -> dict[str, Any]:
        """Execute the test with configured parameters.

        Args:
            agent: Agent to test

        Returns:
            Test execution result

        Raises:
            ValueError: If context not set
        """
        if not self.context:
            raise ValueError("Context must be set with with_context()")

        if not self.messages:
            self.messages = create_test_messages(["Test message"])

        return await run_agent_test(agent=agent, messages=self.messages, context=self.context)


# ===== PYTEST FIXTURES =====




@pytest.fixture
def mock_web_search_tool():
    """Create mock web search tool."""
    return MockTool(
        name="web_search",
        return_value={"results": [{"title": "Test Result", "content": "Mock search result"}]},
    )


@pytest.fixture
def mock_chat_model():
    """Create mock chat model with realistic responses."""
    return MockChatModel(
        [
            "I'll help you with that analysis.",
            "Based on the data, here are the key findings:",
            "Let me search for more information.",
            "Here's my final recommendation:",
        ]
    )


@pytest.fixture
def sample_conversation():
    """Create sample conversation messages."""
    return [
        HumanMessage(content="Hello, I need help analyzing some data."),
        HumanMessage(content="Can you search for recent trends in AI development?"),
    ]


# ===== EXPORTS =====

__all__ = [
    # Mock objects
    "MockRuntime",
    "MockChatModel",
    "MockTool",
    "MockMiddleware",
    # Test helpers
    "create_test_messages",
    "run_agent_test",
    "mock_runtime_context",
    "AgentTestBuilder",
    # Fixtures (automatically available when imported in conftest.py)
    "mock_web_search_tool",
    "mock_chat_model",
    "sample_conversation",
]
