"""Pytest configuration for sample_agent tests."""

import asyncio
import os
import sys
from collections.abc import Generator
from unittest.mock import patch

import pytest
from langchain.chat_models import BaseChatModel
from langchain.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult


class MockChatModel(BaseChatModel):
    """Mock chat model for testing without API keys."""
    
    def __init__(self, **kwargs):
        """Initialize MockChatModel."""
        super().__init__(**kwargs)
        
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(self, messages, stop=None, **kwargs):
        # Mock response that simulates agent behavior
        last_message = messages[-1].content.lower() if messages else ""

        if "faang" in last_message and ("headcount" in last_message or "combined" in last_message):
            response = """I need to research FAANG company headcounts and calculate the total.

Based on my research:
- Facebook (Meta): 67,317 employees
- Apple: 164,000 employees
- Amazon: 1,551,000 employees
- Netflix: 14,000 employees
- Google (Alphabet): 181,269 employees

Calculating the sum: 67,317 + 164,000 + 1,551,000 + 14,000 + 181,269 = 1,977,586

The combined headcount of FAANG companies in 2024 is 1,977,586 employees."""
        elif "research" in last_message or ("headcount" in last_message and "calculate" not in last_message):
            response = """Here are the headcounts for each of the FAANG companies in 2024:
1. **Facebook (Meta)**: 67,317 employees.
2. **Apple**: 164,000 employees.
3. **Amazon**: 1,551,000 employees.
4. **Netflix**: 14,000 employees.
5. **Google (Alphabet)**: 181,269 employees."""
        elif any(num in last_message for num in ["67317", "164000", "1551000"]) or "calculate" in last_message:
            response = "The sum is 67,317 + 164,000 + 1,551,000 + 14,000 + 181,269 = 1,977,586"
        elif "2+2" in last_message:
            response = "2 + 2 = 4"
        else:
            response = "I'll help you with that task. Let me analyze what you need."

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return LLMResult(generations=[[generation]])
    
    async def _agenerate(self, messages, stop=None, **kwargs):
        return self._generate(messages, stop, **kwargs)
    
    def bind_tools(self, tools, **kwargs):
        """Mock bind_tools method for compatibility with agent creation."""
        return self


@pytest.fixture(scope="session") 
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Set up test environment variables and mocks before any imports."""
    # Set test mode environment variable
    original_test_mode = os.environ.get("PYTEST_RUNNING")
    os.environ["PYTEST_RUNNING"] = "true"

    # Clear any cached sample_agent modules to ensure fresh imports with mocks
    modules_to_clear = [name for name in sys.modules.keys() if name.startswith('sample_agent')]
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]

    yield

    # Cleanup
    if original_test_mode is None:
        os.environ.pop("PYTEST_RUNNING", None)
    else:
        os.environ["PYTEST_RUNNING"] = original_test_mode


@pytest.fixture(autouse=True)
def mock_chat_model():
    """Mock load_chat_model to return MockChatModel for tests."""
    # Skip mocking if LANGCHAIN_TRACING_V2 is set for real API calls
    if os.environ.get("LANGCHAIN_TRACING_V2"):
        yield None
        return

    mock_model = MockChatModel()

    def mock_load_chat_model(model_name, **kwargs):
        # Return mock model for any model name during tests
        return mock_model

    # Patch all the load_chat_model import paths used throughout the app
    patches = [
        patch('langgraph_up_devkits.utils.providers._load_chat_model', side_effect=mock_load_chat_model),
        patch('langgraph_up_devkits.utils.providers.load_chat_model', side_effect=mock_load_chat_model),
        patch('langgraph_up_devkits.load_chat_model', side_effect=mock_load_chat_model),
        patch('sample_agent.graph.load_chat_model', side_effect=mock_load_chat_model),
        patch('sample_agent.subagents.math.load_chat_model', side_effect=mock_load_chat_model),
        patch('sample_agent.subagents.research.load_chat_model', side_effect=mock_load_chat_model),
    ]

    # Start all patches
    for patcher in patches:
        patcher.start()

    yield mock_model

    # Stop all patches
    for patcher in patches:
        patcher.stop()


@pytest.fixture(autouse=True)
async def reset_environment():
    """Reset environment before each test."""
    # Clean up any state between tests  
    yield
    # Post-test cleanup if needed