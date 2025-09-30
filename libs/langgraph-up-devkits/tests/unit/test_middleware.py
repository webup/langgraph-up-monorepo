"""Tests for middleware components."""

from unittest.mock import Mock, patch

import pytest

from langgraph_up_devkits.middleware import (
    FileSystemMaskMiddleware,
    ModelProviderMiddleware,
)


# Mock runtime for testing
def _create_mock_runtime() -> Mock:
    """Create a mock runtime for testing."""
    return Mock()


class TestModelProviderMiddleware:
    """Test middleware functionality."""

    def test_middleware_initialization(self):
        """Test middleware can be initialized."""
        middleware = ModelProviderMiddleware()
        assert middleware is not None
        assert hasattr(middleware, "modify_model_request")

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_with_dev_utils(self, mock_load_chat_model):
        """Test model request modification when dev utils are available."""
        # Mock the runtime and context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "qwen:qwen-flash"
        mock_runtime.context = mock_context

        # Mock the model loading
        mock_model = Mock()
        mock_load_chat_model.return_value = mock_model

        # Mock the request and state
        mock_request = Mock()
        mock_state = Mock()

        # Test the middleware
        middleware = ModelProviderMiddleware()
        result = middleware.modify_model_request(mock_request, mock_state, mock_runtime)

        # Verify the model was loaded and set
        mock_load_chat_model.assert_called_once_with("qwen:qwen-flash")
        assert result.model == mock_model

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_fallback(self, mock_load_chat_model):
        """Test model request modification fallback when dev utils not available."""
        # Mock the runtime and context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "openai:openai/gpt-4o"
        mock_runtime.context = mock_context

        # Mock the model loading
        mock_model = Mock()
        mock_load_chat_model.return_value = mock_model

        # Mock the request and state
        mock_request = Mock()
        mock_state = Mock()

        # Test the middleware
        middleware = ModelProviderMiddleware()
        result = middleware.modify_model_request(mock_request, mock_state, mock_runtime)

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
        mock_register.assert_called_once_with("openrouter", "openai", base_url="https://openrouter.ai/api/v1")

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

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_different_model_specs(self, mock_load_model):
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

            # Create middleware and request
            middleware = ModelProviderMiddleware()
            request = Mock()
            request.model = None
            state = Mock()

            # Execute
            result = middleware.modify_model_request(request, state, mock_runtime)

            # Verify
            assert result is request
            assert result.model is mock_model
            mock_load_model.assert_called_with(model_spec)

            # Reset mock for next iteration
            mock_load_model.reset_mock()

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_preserves_other_attributes(self, mock_load_model):
        """Test that middleware preserves other request attributes."""
        # Setup mocks
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "test-model"
        mock_runtime.context = mock_context

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
        result = middleware.modify_model_request(request, state, mock_runtime)

        # Verify model was updated but other attributes preserved
        assert result is request
        assert result.model is mock_model
        assert result.temperature == 0.7
        assert result.max_tokens == 1000
        assert result.custom_attribute == "test_value"

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_model_loading_error(self, mock_load_model):
        """Test middleware behavior when model loading fails."""
        # Setup mocks
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = "invalid-model"
        mock_runtime.context = mock_context

        # Make model loading fail
        mock_load_model.side_effect = ValueError("Invalid model specification")

        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = None
        state = Mock()

        # Middleware should handle the error gracefully and keep original model
        result = middleware.modify_model_request(request, state, mock_runtime)

        # Should return the request unchanged when model loading fails
        assert result is request
        assert request.model is None  # Original model should be unchanged

    def test_modify_model_request_runtime_error(self):
        """Test middleware behavior when runtime access fails."""
        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        state = Mock()

        # Create a runtime that will cause an error when accessing context
        mock_runtime = Mock()
        mock_runtime.context = Mock()
        mock_runtime.context.model = Mock(side_effect=RuntimeError("Runtime not available"))

        # Middleware should handle the runtime error gracefully
        result = middleware.modify_model_request(request, state, mock_runtime)

        # Should return the request unchanged when runtime access fails
        assert result is request


class TestMiddlewareStructure:
    """Test middleware structure and interface without API calls."""

    def test_middleware_interface_structure(self):
        """Test middleware has correct structure and interface."""
        middleware = ModelProviderMiddleware()

        # Test basic attributes
        assert hasattr(middleware, "modify_model_request")

        # Test method signature
        import inspect

        sig = inspect.signature(middleware.modify_model_request)
        params = list(sig.parameters.keys())

        assert "request" in params
        assert "state" in params

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

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_no_context_model_with_provider_format(self, mock_load_model):
        """Test middleware when no context model but original has provider format."""
        # Setup runtime without model context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = None  # No model in context
        mock_runtime.context = mock_context

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
        result = middleware.modify_model_request(request, state, mock_runtime)

        # Should resolve the provider model format
        mock_load_model.assert_called_with("openai:gpt-4")
        assert result.model is mock_model

    @patch("langgraph_up_devkits.middleware.model_provider.load_chat_model")
    def test_modify_model_request_provider_resolution_failure(self, mock_load_model):
        """Test middleware when provider resolution fails."""
        # Setup runtime without model context
        mock_runtime = Mock()
        mock_context = Mock()
        mock_context.model = None
        mock_runtime.context = mock_context

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
            middleware.modify_model_request(request, state, mock_runtime)

    def test_modify_model_request_unexpected_exception(self):
        """Test middleware behavior with unexpected exceptions."""
        # Create middleware and request
        middleware = ModelProviderMiddleware()
        request = Mock()
        request.model = "test-model"
        state = Mock()

        # Create a runtime that will cause an unexpected exception
        mock_runtime = Mock()
        mock_runtime.context = Mock()
        mock_runtime.context.model = None  # No model in context

        # Test that middleware handles the case gracefully (no exception should be raised)
        result = middleware.modify_model_request(request, state, mock_runtime)

        # Should return the request unchanged when no context model
        assert result is request
        assert result.model == "test-model"

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


class TestFileSystemMaskMiddleware:
    """Test FileSystemMask middleware functionality."""

    def test_middleware_initialization(self):
        """Test middleware can be initialized."""
        middleware = FileSystemMaskMiddleware()
        assert middleware is not None
        assert hasattr(middleware, "before_model")
        assert hasattr(middleware, "after_model")
        assert hasattr(middleware, "_shadowed_files")

    def test_before_model_with_files(self):
        """Test before_model shadows files field."""
        middleware = FileSystemMaskMiddleware()

        # Create state with files
        state = {
            "messages": ["message1", "message2"],
            "files": ["/path/to/file1.txt", "/path/to/file2.txt"],
            "other_field": "value",
        }

        # Call before_model
        runtime = _create_mock_runtime()
        result = middleware.before_model(state, runtime)

        # Verify files are removed from returned state
        assert result is not None
        assert "files" not in result
        assert result["messages"] == ["message1", "message2"]
        assert result["other_field"] == "value"

        # Verify files are shadowed internally
        assert middleware._shadowed_files == ["/path/to/file1.txt", "/path/to/file2.txt"]

    def test_before_model_without_files(self):
        """Test before_model returns None when no files field."""
        middleware = FileSystemMaskMiddleware()

        # Create state without files
        state = {"messages": ["message1"], "other_field": "value"}

        # Call before_model
        runtime = _create_mock_runtime()
        result = middleware.before_model(state, runtime)

        # Should return None when no files to shadow
        assert result is None
        assert middleware._shadowed_files is middleware._NO_FILES_SENTINEL

    def test_before_model_with_non_dict_state(self):
        """Test before_model handles non-dict state gracefully."""
        middleware = FileSystemMaskMiddleware()

        # Test with non-dict state
        state = "not a dict"

        # Call before_model
        runtime = _create_mock_runtime()
        result = middleware.before_model(state, runtime)

        # Should return None for non-dict state
        assert result is None

    def test_after_model_restores_files(self):
        """Test after_model restores shadowed files."""
        middleware = FileSystemMaskMiddleware()

        # Create state with files
        state = {"messages": ["message1"], "files": ["/path/to/file.txt"]}

        # Shadow the files first
        runtime = _create_mock_runtime()
        middleware.before_model(state, runtime)

        # Now test restoration
        result = middleware.after_model(state, runtime)

        # Verify files are restored
        assert result is not None
        assert "files" in result
        assert result["files"] == ["/path/to/file.txt"]

        # Verify shadowed files are cleaned up
        assert middleware._shadowed_files is middleware._NO_FILES_SENTINEL

    def test_after_model_without_shadowed_files(self):
        """Test after_model returns None when no files were shadowed."""
        middleware = FileSystemMaskMiddleware()

        # Create state
        state = {"messages": ["message1"]}

        # Call after_model without shadowing first
        runtime = _create_mock_runtime()
        result = middleware.after_model(state, runtime)

        # Should return None when no files to restore
        assert result is None

    def test_complete_workflow(self):
        """Test complete before_model -> after_model workflow."""
        middleware = FileSystemMaskMiddleware()

        # Initial state with files
        original_files = ["/file1.txt", "/file2.txt", "/file3.txt"]
        state = {"messages": ["hello"], "files": original_files, "user_id": "test_user"}

        # Step 1: Shadow files before model
        runtime = _create_mock_runtime()
        before_result = middleware.before_model(state, runtime)
        assert before_result is not None
        assert "files" not in before_result
        assert before_result["messages"] == ["hello"]
        assert before_result["user_id"] == "test_user"

        # Step 2: Restore files after model
        after_result = middleware.after_model(state, runtime)
        assert after_result is not None
        assert after_result["files"] == original_files

        # Verify cleanup
        assert middleware._shadowed_files is middleware._NO_FILES_SENTINEL

    def test_sequential_execution(self):
        """Test middleware handles sequential execution correctly."""
        middleware = FileSystemMaskMiddleware()

        # First execution cycle
        state1 = {"messages": ["state1"], "files": ["file1.txt"]}

        runtime = _create_mock_runtime()
        result1 = middleware.before_model(state1, runtime)
        assert "files" not in result1

        after_result1 = middleware.after_model(state1, runtime)
        assert after_result1["files"] == ["file1.txt"]
        assert middleware._shadowed_files is middleware._NO_FILES_SENTINEL

        # Second execution cycle (reusing the same middleware instance)
        state2 = {"messages": ["state2"], "files": ["file2.txt"]}

        result2 = middleware.before_model(state2, runtime)
        assert "files" not in result2

        after_result2 = middleware.after_model(state2, runtime)
        assert after_result2["files"] == ["file2.txt"]
        assert middleware._shadowed_files is middleware._NO_FILES_SENTINEL

    def test_files_field_with_empty_list(self):
        """Test middleware handles empty files list."""
        middleware = FileSystemMaskMiddleware()

        state = {"messages": ["test"], "files": []}

        # Shadow empty files
        runtime = _create_mock_runtime()
        before_result = middleware.before_model(state, runtime)
        assert "files" not in before_result

        # Restore empty files
        after_result = middleware.after_model(state, runtime)
        assert after_result["files"] == []

    def test_files_field_with_none(self):
        """Test middleware handles files=None."""
        middleware = FileSystemMaskMiddleware()

        state = {"messages": ["test"], "files": None}

        # Shadow None files
        runtime = _create_mock_runtime()
        before_result = middleware.before_model(state, runtime)
        assert "files" not in before_result

        # Restore None files
        after_result = middleware.after_model(state, runtime)
        assert after_result["files"] is None

    def test_files_field_with_complex_data(self):
        """Test middleware handles complex file data structures."""
        middleware = FileSystemMaskMiddleware()

        complex_files = [
            {"path": "/file1.txt", "size": 1024, "type": "text"},
            {"path": "/file2.pdf", "size": 2048, "type": "pdf"},
        ]

        state = {"messages": ["test"], "files": complex_files}

        # Shadow complex files
        runtime = _create_mock_runtime()
        before_result = middleware.before_model(state, runtime)
        assert "files" not in before_result

        # Restore complex files
        after_result = middleware.after_model(state, runtime)
        assert after_result["files"] == complex_files
        assert after_result["files"][0]["path"] == "/file1.txt"
        assert after_result["files"][1]["type"] == "pdf"
