"""Model provider middleware using our load_chat_model utility."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import ModelRequest, ModelResponse

from ..utils.providers import load_chat_model
from .base import BaseMiddleware


class ModelProviderMiddleware(BaseMiddleware[Any, Any]):
    """Middleware that automatically loads models with provider registration.

    This middleware handles model loading and provider switching based on the
    model specification in the runtime context or request.

    Supports:
    - OpenRouter models: openrouter:openai/gpt-4o, openrouter:anthropic/claude-sonnet-4
    - Qwen models: qwen:qwen-flash, qwq:qwq-32b-preview
    - SiliconFlow models: siliconflow:qwen-plus
    - Standard models via init_chat_model fallback

    Provider registration is handled automatically - no manual setup required.
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the middleware.

        Args:
            debug: Enable debug logging for model loading operations.
        """
        super().__init__(debug=debug)

    def _process_model_request(self, request: ModelRequest) -> None:
        """Process and modify the model request (shared logic for sync and async)."""
        self._log(
            "Processing model request",
            request.runtime,
            model=str(request.model),
            model_type=type(request.model).__name__,
        )

        try:
            # Check if runtime context has a model specification
            model_spec = getattr(request.runtime.context, "model", None)
            self._log("Context model spec", request.runtime, model_spec=model_spec)

            if model_spec:
                # Context specifies a model - load it with automatic provider registration
                try:
                    new_model = load_chat_model(model_spec)
                    request.model = new_model
                    self._log(
                        "Switched to context model",
                        request.runtime,
                        model_spec=model_spec,
                        new_model_type=type(new_model).__name__,
                    )
                except Exception as model_err:
                    self._log(
                        "Failed to load context model",
                        request.runtime,
                        model_spec=model_spec,
                        error=str(model_err),
                    )
                    # Fall back to original model

            else:
                # No model in context - check if original model needs provider resolution
                original_model_str = str(request.model)
                self._log("No context model, checking original", request.runtime, original_model=original_model_str)

                # If original model has provider:model format, resolve it
                if ":" in original_model_str and not hasattr(request.model, "invoke"):
                    try:
                        resolved_model = load_chat_model(original_model_str)
                        request.model = resolved_model
                        self._log(
                            "Resolved provider model",
                            request.runtime,
                            original_model=original_model_str,
                            resolved_type=type(resolved_model).__name__,
                        )
                    except Exception as resolve_err:
                        self._log(
                            "Failed to resolve provider model",
                            request.runtime,
                            original_model=original_model_str,
                            error=str(resolve_err),
                        )
                        raise ValueError(f"Cannot resolve model provider for: {original_model_str}") from resolve_err

        except RuntimeError as runtime_err:
            self._log("Runtime error (no context available)", error=str(runtime_err))
            # When no runtime context, fallback to original model

        except Exception as e:
            self._log("Unexpected error", error=str(e))
            # Re-raise unexpected errors
            raise

        self._log(
            "Final model",
            request.runtime,
            model=str(request.model),
            model_type=type(request.model).__name__,
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Load model using our load_chat_model utility with automatic provider registration (sync)."""
        self._process_model_request(request)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Load model using our load_chat_model utility with automatic provider registration (async)."""
        self._process_model_request(request)
        return await handler(request)
