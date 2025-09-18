"""Global test configuration for langgraph-up-devkits."""

import asyncio

import pytest

from langgraph_up_devkits.context.schemas import BaseAgentContext, SearchContext
from tests.testing import MockRuntime

# Note: Environment variables are now loaded via pytest-dotenv plugin
# Configuration: env_files = ["../../.env"] in pyproject.toml


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def base_context():
    """Create base agent context for testing."""
    return BaseAgentContext(
        model="openai:openai/gpt-4o", user_id="test_user", session_id="test_session"
    )


@pytest.fixture
def search_context():
    """Create search context for testing."""
    return SearchContext(
        model="openai:openai/gpt-4o",
        max_search_results=10,
        enable_deepwiki=True,
        user_id="test_user",
    )


@pytest.fixture
def mock_runtime(base_context):
    """Create mock runtime with base context."""
    return MockRuntime(context=base_context)
