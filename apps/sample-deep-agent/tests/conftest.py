"""Test configuration and fixtures."""

import os
from unittest.mock import Mock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_openai_key(monkeypatch):
    """Mock OPENAI_API_KEY to prevent API key errors in unit tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake-key-for-unit-tests")


@pytest.fixture
def mock_model():
    """Mock chat model for testing."""
    mock = Mock()
    mock.invoke.return_value = Mock(content="Test response")
    return mock


@pytest.fixture
def mock_load_chat_model(mock_model):
    """Mock load_chat_model function."""
    with patch("sample_deep_agent.graph.load_chat_model", return_value=mock_model):
        yield mock_model


@pytest.fixture
def mock_deepagents():
    """Mock deepagents create_deep_agent function."""
    with patch("sample_deep_agent.graph.create_deep_agent") as mock:
        mock_agent = Mock()
        mock_agent.compile.return_value = Mock()
        mock.return_value = mock_agent
        yield mock


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "configurable": {
            "model_name": "siliconflow:zai-org/GLM-4.5"
        }
    }