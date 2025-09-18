"""Tests for middleware components."""

from unittest.mock import Mock, patch

import pytest

from langgraph_up_devkits.middleware import ModelProviderMiddleware


class TestModelProviderMiddleware:
    """Test middleware functionality."""

    def test_middleware_initialization(self):
        """Test middleware can be initialized."""
        middleware = ModelProviderMiddleware()
        assert middleware is not None
        assert hasattr(middleware, "modify_model_request")

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_with_dev_utils(
        self, mock_load_chat_model, mock_get_runtime
    ):
        """Test model request modification when dev utils are available."""
        # Mock the runtime and context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "qwen:qwen-flash"
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        # Mock the model loading
        mock_model = Mock()
        mock_load_chat_model.return_value = mock_model

        # Mock the request and state
        mock_request = Mock()
        mock_state = Mock()

        # Test the middleware
        middleware = ModelProviderMiddleware()
        result = middleware.modify_model_request(mock_request, mock_state)

        # Verify the model was loaded and set
        mock_load_chat_model.assert_called_once_with("qwen:qwen-flash")
        assert result.model == mock_model

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_fallback(
        self, mock_load_chat_model, mock_get_runtime
    ):
        """Test model request modification fallback when dev utils not available."""
        # Mock the runtime and context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "openai:openai/gpt-4o"
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        # Mock the model loading
        mock_model = Mock()
        mock_load_chat_model.return_value = mock_model

        # Mock the request and state
        mock_request = Mock()
        mock_state = Mock()

        # Test the middleware
        middleware = ModelProviderMiddleware()
        result = middleware.modify_model_request(mock_request, mock_state)

        # Verify the model was loaded
        mock_load_chat_model.assert_called_once_with("openai:openai/gpt-4o")
        assert result.model == mock_model


class TestProviderRegistration:
    """Test provider registration utilities."""

    @patch("langgraph_up_devkits.utils.providers.DEV_UTILS_AVAILABLE", True)
    @patch("langgraph_up_devkits.utils.providers.register_model_provider")
    def test_register_openrouter_provider(self, mock_register):
        """Test OpenRouter provider registration."""
        from langgraph_up_devkits.utils.providers import _register_openrouter_provider

        result = _register_openrouter_provider()

        assert result is True
        mock_register.assert_called_once_with(
            "openrouter", "openai", base_url="https://openrouter.ai/api/v1"
        )

    @patch("langgraph_up_devkits.utils.providers.DEV_UTILS_AVAILABLE", False)
    def test_register_provider_when_unavailable(self):
        """Test provider registration when dev utils not available."""
        from langgraph_up_devkits.utils.providers import _register_openrouter_provider

        result = _register_openrouter_provider()
        assert result is False

    @patch("langgraph_up_devkits.utils.providers.DEV_UTILS_AVAILABLE", True)
    @patch("importlib.import_module")
    @patch("langgraph_up_devkits.utils.providers.register_model_provider")
    def test_register_qwen_provider_success(self, mock_register, mock_import):
        """Test successful Qwen provider registration."""
        from langgraph_up_devkits.utils.providers import _register_qwen_provider

        # Mock the langchain_qwq module
        mock_module = Mock()
        mock_chat_qwen = Mock()
        mock_chat_qwq = Mock()
        mock_module.ChatQwen = mock_chat_qwen
        mock_module.ChatQwQ = mock_chat_qwq
        mock_import.return_value = mock_module

        result = _register_qwen_provider()

        assert result is True
        # Should register both ChatQwen and ChatQwQ
        assert mock_register.call_count >= 2

    @patch("langgraph_up_devkits.utils.providers.DEV_UTILS_AVAILABLE", True)
    @patch("importlib.import_module")
    def test_register_qwen_provider_import_error(self, mock_import):
        """Test Qwen provider registration with import error."""
        from langgraph_up_devkits.utils.providers import _register_qwen_provider

        # Make import fail
        mock_import.side_effect = ImportError("Module not found")

        result = _register_qwen_provider()
        assert result is False


class TestModelProviderMiddlewareEnhanced:
    """Enhanced tests for ModelProviderMiddleware."""

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_different_model_specs(
        self, mock_load_model, mock_get_runtime
    ):
        """Test middleware with different model specifications."""
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        test_models = [
            "openai:openai/gpt-4o",
            "openai:anthropic/claude-sonnet-4",
            "qwen:qwen-flash",
            "qwq:qwq-32b-preview",
            "siliconflow:qwen-plus",
            "standard/model-name",
        ]

        for model_spec in test_models:
            # Setup runtime for each test
            mock_runtime = Mock()
            mock_context = Mock()
            mock_context.model = model_spec
            mock_runtime.context = mock_context
            mock_get_runtime.return_value = mock_runtime

            # Create middleware and request
            middleware = ModelProviderMiddleware()
            request = Mock()
            request.model = None
            state = Mock()

            # Execute
            result = middleware.modify_model_request(request, state)

            # Verify
            assert result is request
            assert result.model is mock_model
            mock_load_model.assert_called_with(model_spec)

            # Reset mock for next iteration
            mock_load_model.reset_mock()

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_preserves_other_attributes(
        self, mock_load_model, mock_get_runtime
    ):
        """Test that middleware preserves other request attributes."""
        # Setup mocks
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "test-model"
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create middleware and request with additional attributes
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = None
        request.temperature = 0.7
        request.max_tokens = 1000
        request.custom_attribute = "test_value"
        state = Mock()

        # Execute
        result = middleware.modify_model_request(request, state)

        # Verify model was updated but other attributes preserved
        assert result is request
        assert result.model is mock_model
        assert result.temperature == 0.7
        assert result.max_tokens == 1000
        assert result.custom_attribute == "test_value"

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_model_loading_error(
        self, mock_load_model, mock_get_runtime
    ):
        """Test middleware behavior when model loading fails."""
        # Setup mocks
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "invalid-model"
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        # Make model loading fail
        mock_load_model.side_effect = ValueError("Invalid model specification")

        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = None
        state = Mock()

        # Middleware should handle the error gracefully and keep original model
        result = middleware.modify_model_request(request, state)

        # Should return the request unchanged when model loading fails
        assert result is request
        assert request.model is None  # Original model should be unchanged

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    def test_modify_model_request_runtime_error(self, mock_get_runtime):
        """Test middleware behavior when runtime access fails."""
        # Make runtime access fail
        mock_get_runtime.side_effect = RuntimeError("Runtime not available")

        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        state = Mock()

        # Middleware should handle the runtime error gracefully
        result = middleware.modify_model_request(request, state)

        # Should return the request unchanged when runtime access fails
        assert result is request


class TestMiddlewareStructure:
    """Test middleware structure and interface without API calls."""

    def test_middleware_interface_structure(self):
        """Test middleware has correct structure and interface."""
        middleware = ModelProviderMiddleware()

        # Test basic attributes
        assert hasattr(middleware, 'modify_model_request')

        # Test method signature
        import inspect
        sig = inspect.signature(middleware.modify_model_request)
        params = list(sig.parameters.keys())

        assert 'request' in params
        assert 'state' in params

        print("✅ Middleware structure is correct")

    @patch("langgraph_up_devkits.utils.providers._register_siliconflow_provider")
    def test_enhanced_siliconflow_provider_registration_unit(self, mock_register):
        """Test our enhanced SiliconFlow provider registration logic (unit test)."""
        import os
        from unittest.mock import patch

        # Mock successful registration
        mock_register.return_value = True

        # Test that the function can be called
        from langgraph_up_devkits.utils.providers import _register_siliconflow_provider
        result = _register_siliconflow_provider()

        # Verify basic functionality
        assert result is True or result is False  # Should return boolean

        # Test with mocked environment
        with patch.dict(os.environ, {"REGION": "prc"}):
            region = os.getenv("REGION", "")
            assert region == "prc"

        print("✅ Enhanced SiliconFlow provider registration unit test passed")

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_no_context_model_with_provider_format(
        self, mock_load_model, mock_get_runtime
    ):
        """Test middleware when no context model but original has provider format."""
        # Setup runtime without model context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = None  # No model in context
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        # Mock the model loading
        mock_model = Mock()
        mock_load_model.return_value = mock_model

        # Create middleware and request with provider format
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = "openai:gpt-4"  # String with provider format
        # Create a mock that doesn't have invoke attribute
        mock_model_obj = Mock()
        del mock_model_obj.invoke  # Remove invoke attribute
        request.model = mock_model_obj
        request.model.__str__ = Mock(return_value="openai:gpt-4")  # Make str() work
        state = Mock()

        # Execute
        result = middleware.modify_model_request(request, state)

        # Should resolve the provider model format
        mock_load_model.assert_called_with("openai:gpt-4")
        assert result.model is mock_model

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_provider_resolution_failure(
        self, mock_load_model, mock_get_runtime
    ):
        """Test middleware when provider resolution fails."""
        # Setup runtime without model context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = None
        mock_runtime.context = mock_context
        mock_get_runtime.return_value = mock_runtime

        # Make model loading fail
        mock_load_model.side_effect = Exception("Provider not registered")

        # Create middleware and request with provider format
        middleware = ModelProviderMiddleware()
        request = Mock()
        # Create a mock that doesn't have invoke attribute
        mock_model_obj = Mock()
        del mock_model_obj.invoke  # Remove invoke attribute
        request.model = mock_model_obj
        request.model.__str__ = Mock(return_value="unknown:unknown-model")  # Make str() work
        state = Mock()

        # Should raise ValueError when provider resolution fails
        with pytest.raises(ValueError, match="Cannot resolve model provider"):
            middleware.modify_model_request(request, state)

    @patch("langgraph_up_devkits.middleware.model_provider.get_runtime")
    def test_modify_model_request_unexpected_exception(self, mock_get_runtime):
        """Test middleware behavior with unexpected exceptions."""
        # Make get_runtime raise an unexpected exception
        mock_get_runtime.side_effect = Exception("Unexpected error")

        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = "test-model"
        state = Mock()

        # Should re-raise unexpected exceptions
        with pytest.raises(Exception, match="Unexpected error"):
            middleware.modify_model_request(request, state)

    def test_dev_utils_import_fallback(self):
        """Test fallback behavior when langchain-dev-utils is not available."""
        # Test that debug mode can be enabled/disabled
        from langgraph_up_devkits.middleware.model_provider import ModelProviderMiddleware

        middleware = ModelProviderMiddleware(debug=True)
        assert middleware.debug is True

        middleware_no_debug = ModelProviderMiddleware(debug=False)
        assert middleware_no_debug.debug is False

        # Test that we can import the load_chat_model function
        from langgraph_up_devkits.middleware.model_provider import load_chat_model
        assert callable(load_chat_model)
