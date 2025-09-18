"""Tests for prompt templates and utilities."""

from datetime import UTC
from unittest.mock import Mock, patch

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_up_devkits.prompts import (
    DATA_ANALYST_PROMPT,
    RESEARCH_ASSISTANT_PROMPT,
    SYSTEM_PROMPT,
)
from langgraph_up_devkits.utils.prompts import create_context_aware_prompt


class TestPromptTemplates:
    """Test pure prompt templates."""

    def test_system_prompt_format(self):
        """Test system prompt has correct format."""
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT.strip()) > 0
        assert "{system_time}" in SYSTEM_PROMPT
        assert "helpful ai assistant" in SYSTEM_PROMPT.lower()

    def test_data_analyst_prompt_format(self):
        """Test data analyst prompt has correct format."""
        assert isinstance(DATA_ANALYST_PROMPT, str)
        assert len(DATA_ANALYST_PROMPT.strip()) > 0
        assert "{system_time}" in DATA_ANALYST_PROMPT
        assert "data analyst assistant" in DATA_ANALYST_PROMPT.lower()
        assert "data analysis" in DATA_ANALYST_PROMPT.lower()
        assert "visualization" in DATA_ANALYST_PROMPT.lower()

    def test_research_assistant_prompt_format(self):
        """Test research assistant prompt has correct format."""
        assert isinstance(RESEARCH_ASSISTANT_PROMPT, str)
        assert len(RESEARCH_ASSISTANT_PROMPT.strip()) > 0
        assert "{system_time}" in RESEARCH_ASSISTANT_PROMPT
        assert "research assistant" in RESEARCH_ASSISTANT_PROMPT.lower()
        assert "finding" in RESEARCH_ASSISTANT_PROMPT.lower()
        assert "analyzing" in RESEARCH_ASSISTANT_PROMPT.lower()

    def test_prompt_template_substitution(self):
        """Test that prompt templates can be formatted with system time."""
        test_time = "2024-01-01T12:00:00Z"

        formatted_system = SYSTEM_PROMPT.format(system_time=test_time)
        assert test_time in formatted_system
        assert "{system_time}" not in formatted_system

        formatted_data_analyst = DATA_ANALYST_PROMPT.format(system_time=test_time)
        assert test_time in formatted_data_analyst
        assert "{system_time}" not in formatted_data_analyst

        formatted_research = RESEARCH_ASSISTANT_PROMPT.format(system_time=test_time)
        assert test_time in formatted_research
        assert "{system_time}" not in formatted_research

    def test_prompt_uniqueness(self):
        """Test that each prompt template is unique."""
        prompts = [SYSTEM_PROMPT, DATA_ANALYST_PROMPT, RESEARCH_ASSISTANT_PROMPT]
        assert len(set(prompts)) == len(prompts), "All prompts should be unique"

    def test_prompt_lengths(self):
        """Test that prompts have reasonable lengths."""
        # All prompts should be substantial but not excessively long
        for prompt in [SYSTEM_PROMPT, DATA_ANALYST_PROMPT, RESEARCH_ASSISTANT_PROMPT]:
            assert 50 < len(prompt) < 1000, (
                f"Prompt length should be reasonable: {len(prompt)}"
            )


class TestCreateContextAwarePrompt:
    """Test dynamic prompt creation utility."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_runtime = Mock()
        self.mock_context = Mock()
        self.mock_runtime.context = self.mock_context

        # Default context attributes
        self.mock_context.system_prompt = "You are a helpful assistant."
        self.mock_context.user_id = None
        self.mock_context.session_id = None
        self.mock_context.enable_deepwiki = False
        self.mock_context.max_search_results = None
        self.mock_context.max_data_rows = None

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_basic_prompt_creation(self, mock_get_runtime):
        """Test basic prompt creation with minimal context."""
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": [HumanMessage(content="Hello")]}
        result = create_context_aware_prompt(state)

        assert isinstance(result, list)
        assert len(result) == 2  # System message + original message
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], HumanMessage)

        system_content = result[0].content
        assert "You are a helpful assistant." in system_content
        assert "Current time:" in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_context_with_user_info(self, mock_get_runtime):
        """Test prompt creation with user information."""
        self.mock_context.user_id = "test_user_123"
        self.mock_context.session_id = "session_456"
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        system_content = result[0].content
        assert "User ID: test_user_123" in system_content
        assert "Session ID: session_456" in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_context_with_tool_info(self, mock_get_runtime):
        """Test prompt creation with tool availability information."""
        self.mock_context.enable_deepwiki = True
        self.mock_context.max_search_results = 10
        self.mock_context.max_data_rows = 5000
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        system_content = result[0].content
        assert "DeepWiki for open source documentation" in system_content
        assert "Web searches return up to 10 results" in system_content
        assert "Data analysis operations are limited to 5000 rows" in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_custom_system_prompt(self, mock_get_runtime):
        """Test prompt creation with custom system prompt."""
        custom_prompt = "You are a specialized AI for testing purposes."
        self.mock_context.system_prompt = custom_prompt
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        system_content = result[0].content
        assert custom_prompt in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_empty_state_handling(self, mock_get_runtime):
        """Test prompt creation with empty or missing state."""
        mock_get_runtime.return_value = self.mock_runtime

        # Test with empty state
        result1 = create_context_aware_prompt({})
        assert len(result1) == 1  # Only system message

        # Test with state without messages
        result2 = create_context_aware_prompt({"other_field": "value"})
        assert len(result2) == 1  # Only system message

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    @patch("langgraph_up_devkits.utils.prompts.datetime")
    def test_timestamp_formatting(self, mock_datetime, mock_get_runtime):
        """Test that timestamp is properly formatted."""
        # Mock the datetime to return a predictable time
        mock_now = Mock()
        mock_now.isoformat.return_value = "2024-01-01T12:00:00Z"
        mock_datetime.now.return_value = mock_now
        mock_datetime.UTC = UTC

        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        system_content = result[0].content
        assert "Current time: 2024-01-01T12:00:00Z" in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_getattr_fallbacks(self, mock_get_runtime):
        """Test that getattr fallbacks work when context attributes are missing."""
        # Create a context with missing attributes
        minimal_context = Mock(spec=[])  # No attributes defined
        self.mock_runtime.context = minimal_context
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        # Should not raise AttributeError, should use defaults
        system_content = result[0].content
        assert "You are a helpful assistant." in system_content  # Default prompt
        assert "Current time:" in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_none_values_handling(self, mock_get_runtime):
        """Test handling of None values in context attributes."""
        self.mock_context.user_id = None
        self.mock_context.session_id = None
        self.mock_context.max_search_results = None
        self.mock_context.max_data_rows = None
        mock_get_runtime.return_value = self.mock_runtime

        state = {"messages": []}
        result = create_context_aware_prompt(state)

        system_content = result[0].content
        # None values should not add their respective sections
        assert "User ID:" not in system_content
        assert "Session ID:" not in system_content
        assert "Web searches return up to" not in system_content
        assert "Data analysis operations are limited to" not in system_content

    @patch("langgraph_up_devkits.utils.prompts.get_runtime")
    def test_message_preservation(self, mock_get_runtime):
        """Test that original messages are preserved and appended."""
        mock_get_runtime.return_value = self.mock_runtime

        original_messages = [
            HumanMessage(content="First message"),
            HumanMessage(content="Second message"),
        ]
        state = {"messages": original_messages}

        result = create_context_aware_prompt(state)

        assert len(result) == 3  # System + 2 original messages
        assert result[1] == original_messages[0]
        assert result[2] == original_messages[1]
