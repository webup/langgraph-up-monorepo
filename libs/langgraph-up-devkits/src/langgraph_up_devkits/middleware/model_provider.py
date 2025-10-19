"""Model provider middleware using our load_chat_model utility."""

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from ..utils.providers import load_chat_model


class ModelProviderMiddleware(AgentMiddleware[Any]):
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
        super().__init__()
        self.debug = debug

    def _log(self, message: str) -> None:
        """Log debug messages if debug mode is enabled."""
        if self.debug:
            print(f"ModelProviderMiddleware: {message}")

    def _process_model_request(self, request: ModelRequest) -> None:
        """Process and modify the model request (shared logic for sync and async)."""
        self._log(f"Processing model request: {request.model} (type: {type(request.model)})")

        try:
            # Check if runtime context has a model specification
            model_spec = getattr(request.runtime.context, "model", None)
            self._log(f"Context model spec: {model_spec}")

            if model_spec:
                # Context specifies a model - load it with automatic provider registration
                try:
                    new_model = load_chat_model(model_spec)
                    request.model = new_model
                    self._log(f"Switched to context model: {model_spec} -> {type(new_model)}")
                except Exception as model_err:
                    self._log(f"Failed to load context model {model_spec}: {model_err}")
                    # Fall back to original model

            else:
                # No model in context - check if original model needs provider resolution
                original_model_str = str(request.model)
                self._log(f"No context model, checking original: {original_model_str}")

                # If original model has provider:model format, resolve it
                if ":" in original_model_str and not hasattr(request.model, "invoke"):
                    try:
                        resolved_model = load_chat_model(original_model_str)
                        request.model = resolved_model
                        self._log(f"Resolved provider model: {original_model_str} -> {type(resolved_model)}")
                    except Exception as resolve_err:
                        self._log(f"Failed to resolve provider model {original_model_str}: {resolve_err}")
                        raise ValueError(f"Cannot resolve model provider for: {original_model_str}") from resolve_err

        except RuntimeError as runtime_err:
            self._log(f"Runtime error (no context available): {runtime_err}")
            # When no runtime context, fallback to original model

        except Exception as e:
            self._log(f"Unexpected error: {e}")
            # Re-raise unexpected errors
            raise

        self._log(f"Final model: {request.model} (type: {type(request.model)})")

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
