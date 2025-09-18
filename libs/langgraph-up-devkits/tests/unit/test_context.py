"""Tests for context schemas."""

import os
from unittest.mock import patch

from langgraph_up_devkits.context.schemas import (
    BaseAgentContext,
    DataAnalystContext,
    DataContext,
    ResearchContext,
    SearchContext,
)


class TestBaseAgentContext:
    """Test base context functionality."""

    def test_default_values(self):
        """Test default context values."""
        context = BaseAgentContext()
        assert context.model == "openai:openai/gpt-4o"
        assert context.max_iterations == 10
        assert context.user_id is None

    @patch.dict(
        os.environ, {"MODEL": "qwen:qwen-flash", "USER_ID": "test_user"}, clear=False
    )
    def test_env_var_loading(self):
        """Test environment variable loading."""
        context = BaseAgentContext()
        assert context.model == "qwen:qwen-flash"
        assert context.user_id == "test_user"

    def test_explicit_values_override_env(self):
        """Test that explicit values override environment variables."""
        with patch.dict(os.environ, {"MODEL": "env_model"}, clear=False):
            context = BaseAgentContext(model="explicit_model")
            assert context.model == "explicit_model"


class TestSearchContext:
    """Test search context functionality."""

    def test_search_defaults(self):
        """Test search context defaults."""
        context = SearchContext()
        assert context.max_search_results == 5
        assert context.enable_deepwiki is False

    @patch.dict(
        os.environ,
        {"MAX_SEARCH_RESULTS": "15", "ENABLE_DEEPWIKI": "true"},
        clear=False,
    )
    def test_search_env_loading(self):
        """Test search-specific environment loading."""
        context = SearchContext()
        assert context.max_search_results == 15
        assert context.enable_deepwiki is True


class TestComposedContexts:
    """Test composed context schemas."""

    def test_data_analyst_context_composition(self):
        """Test that DataAnalystContext inherits from both SearchContext and DataContext."""  # noqa: E501
        context = DataAnalystContext()

        # Should have BaseAgentContext fields
        assert hasattr(context, "model")
        assert hasattr(context, "user_id")

        # Should have SearchContext fields
        assert hasattr(context, "max_search_results")
        assert hasattr(context, "enable_deepwiki")

        # Should have DataContext fields
        assert hasattr(context, "max_data_rows")

        # Should have its own fields
        assert "data analyst assistant" in context.system_prompt.lower()

        # Test default values for data analyst
        assert context.max_search_results == 8  # Data analyst override
        assert context.enable_data_viz is True
        assert context.max_data_rows == 1000

    def test_research_context_composition(self):
        """Test that ResearchContext has proper inheritance."""
        context = ResearchContext()

        # Should have BaseAgentContext fields
        assert hasattr(context, "model")
        assert hasattr(context, "user_id")

        # Should have SearchContext fields
        assert hasattr(context, "max_search_results")
        assert hasattr(context, "enable_deepwiki")

        # Should have its own fields
        assert "research assistant" in context.system_prompt.lower()

        # Test default values for research assistant
        assert context.max_search_results == 10  # Research override
        assert context.enable_deepwiki is True  # Research default

    def test_data_context_composition(self):
        """Test DataContext has correct fields and defaults."""
        context = DataContext()

        # Should have BaseAgentContext fields
        assert hasattr(context, "model")
        assert hasattr(context, "user_id")

        # Should have its own fields
        assert hasattr(context, "max_data_rows")
        assert hasattr(context, "enable_data_viz")

        # Test defaults
        assert context.max_data_rows == 1000
        assert context.enable_data_viz is True

    def test_context_field_validation(self):
        """Test context field validation and type checking."""
        # Test valid max_iterations
        context = BaseAgentContext(max_iterations=5)
        assert context.max_iterations == 5

        # Test valid max_search_results
        search_context = SearchContext(max_search_results=15)
        assert search_context.max_search_results == 15

    def test_context_serialization(self):
        """Test context can be serialized to dict."""
        context = DataAnalystContext(
            model="test-model", user_id="test-user", max_search_results=15
        )

        # Should be serializable
        from dataclasses import asdict
        context_dict = asdict(context)
        assert isinstance(context_dict, dict)
        assert context_dict["model"] == "test-model"
        assert context_dict["user_id"] == "test-user"
        assert context_dict["max_search_results"] == 15

    def test_context_inheritance_chain(self):
        """Test the inheritance chain works correctly."""
        # DataAnalystContext should inherit from all parent classes
        da_context = DataAnalystContext()

        assert isinstance(da_context, BaseAgentContext)
        assert isinstance(da_context, SearchContext)
        assert isinstance(da_context, DataContext)
        assert isinstance(da_context, DataAnalystContext)

        # ResearchContext should inherit correctly
        r_context = ResearchContext()

        assert isinstance(r_context, BaseAgentContext)
        assert isinstance(r_context, SearchContext)
        assert isinstance(r_context, ResearchContext)

    def test_context_field_override_precedence(self):
        """Test that explicit values override environment and defaults."""
        with patch.dict(
            os.environ, {"MODEL": "env_model", "USER_ID": "env_user"}, clear=False
        ):
            # Explicit values should win
            context = BaseAgentContext(
                model="explicit_model", user_id="explicit_user", max_iterations=15
            )

            assert context.model == "explicit_model"
            assert context.user_id == "explicit_user"
            assert context.max_iterations == 15

    def test_search_context_boolean_env_parsing(self):
        """Test boolean environment variable parsing for SearchContext."""
        # Test various boolean representations
        bool_tests = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False),
        ]

        for env_val, expected in bool_tests:
            with patch.dict(os.environ, {"ENABLE_DEEPWIKI": env_val}, clear=False):
                context = SearchContext()
                assert context.enable_deepwiki == expected, f"Failed for {env_val}"

    def test_context_default_values_comprehensive(self):
        """Test all default values are set correctly."""
        # BaseAgentContext defaults
        base_context = BaseAgentContext()
        assert base_context.model == "openai:openai/gpt-4o"
        assert base_context.max_iterations == 10
        assert base_context.user_id is None
        assert base_context.session_id is None

        # SearchContext defaults
        search_context = SearchContext()
        assert search_context.max_search_results == 5
        assert search_context.enable_deepwiki is False

        # DataContext defaults
        data_context = DataContext()
        assert data_context.max_data_rows == 1000
        assert data_context.enable_data_viz is True

    @patch.dict(
        os.environ,
        {
            "MODEL": "qwen:qwen-flash",
            "MAX_SEARCH_RESULTS": "20",
            "ENABLE_DATA_VIZ": "false",
            "MAX_DATA_ROWS": "5000",
        },
        clear=False,
    )
    def test_data_analyst_env_loading(self):
        """Test DataAnalystContext environment variable loading."""
        context = DataAnalystContext()
        assert context.model == "qwen:qwen-flash"
        assert context.max_search_results == 20
        assert context.enable_data_viz is False
        assert context.max_data_rows == 5000

    def test_explicit_override_precedence(self):
        """Test that explicit parameters override environment variables."""
        with patch.dict(
            os.environ,
            {"MAX_SEARCH_RESULTS": "100", "ENABLE_DEEPWIKI": "false"},
            clear=False,
        ):
            context = DataAnalystContext(max_search_results=5, enable_deepwiki=True)
            assert context.max_search_results == 5  # Explicit override
            assert context.enable_deepwiki is True  # Explicit override


class TestDataContext:
    """Test data context functionality."""

    def test_data_context_defaults(self):
        """Test data context default values."""
        context = DataContext()
        assert context.max_data_rows == 1000
        assert context.enable_data_viz is True

    @patch.dict(
        os.environ,
        {"MAX_DATA_ROWS": "50000", "ENABLE_DATA_VIZ": "false"},
        clear=False,
    )
    def test_data_context_env_loading(self):
        """Test data context environment variable loading."""
        context = DataContext()
        assert context.max_data_rows == 50000
        assert context.enable_data_viz is False


class TestEnvironmentVariableHandling:
    """Test comprehensive environment variable handling."""

    def test_boolean_env_parsing(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("", False),
            ("invalid", False),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"ENABLE_DEEPWIKI": env_value}, clear=False):
                context = SearchContext()
                assert context.enable_deepwiki == expected, f"Failed for '{env_value}'"

    def test_integer_env_parsing(self):
        """Test integer environment variable parsing."""
        test_cases = [
            ("10", 10),
            ("0", 0),
            ("1000", 1000),
        ]

        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MAX_SEARCH_RESULTS": env_value}, clear=False):
                context = SearchContext()
                assert context.max_search_results == expected

    def test_invalid_integer_env_fallback(self):
        """Test that invalid integer env vars fall back to defaults."""
        with patch.dict(os.environ, {"MAX_SEARCH_RESULTS": "invalid"}, clear=False):
            context = SearchContext()
            assert context.max_search_results == 5  # Default value

    def test_empty_env_vars(self):
        """Test behavior with empty environment variables."""
        with patch.dict(
            os.environ, {"MODEL": "", "USER_ID": "", "SESSION_ID": ""}, clear=False
        ):
            context = BaseAgentContext()
            # Empty strings should be treated as None/default
            assert context.model == "openai:openai/gpt-4o"  # Default
            assert context.user_id is None
            assert context.session_id is None


class TestContextValidation:
    """Test context field validation."""

    def test_model_field_validation(self):
        """Test model field accepts valid formats."""
        valid_models = [
            "openai:openai/gpt-4o",
            "qwen:qwen-flash",
            "anthropic:claude-3-sonnet",
            "local-model",
        ]

        for model in valid_models:
            context = BaseAgentContext(model=model)
            assert context.model == model

    def test_positive_integer_fields(self):
        """Test that integer fields accept positive values."""
        context = SearchContext(max_search_results=100, max_iterations=50)
        assert context.max_search_results == 100
        assert context.max_iterations == 50

    def test_zero_values_allowed(self):
        """Test that zero values are allowed for appropriate fields."""
        context = SearchContext(max_search_results=0)
        assert context.max_search_results == 0
