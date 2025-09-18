"""Model provider registration using langchain-dev-utils."""

import os
from typing import Any

# Type hint for chat models - using Any since we don't want to enforce specific LangChain types
ChatModel = Any

try:
    from langchain_dev_utils import load_chat_model as _load_chat_model_original
    from langchain_dev_utils import register_model_provider as _register_model_provider

    DEV_UTILS_AVAILABLE = True

    def register_model_provider(provider_name: str, chat_model: Any, base_url: str | None = None) -> Any:
        """Wrapper for langchain_dev_utils.register_model_provider."""
        if base_url is not None:
            return _register_model_provider(provider_name, chat_model, base_url=base_url)
        return _register_model_provider(provider_name, chat_model)

    def _load_chat_model(model: str, *, model_provider: str | None = None, **kwargs: Any) -> Any:
        """Wrapper for langchain_dev_utils.load_chat_model."""
        return _load_chat_model_original(model, model_provider=model_provider, **kwargs)

except ImportError:
    DEV_UTILS_AVAILABLE = False
    from langchain.chat_models import init_chat_model

    def register_model_provider(provider_name: str, chat_model: Any, base_url: str | None = None) -> Any:
        """Mock function when langchain-dev-utils is not available."""
        return None

    def _load_chat_model(model: str, *, model_provider: str | None = None, **kwargs: Any) -> Any:
        """Fallback to init_chat_model when langchain-dev-utils is not available."""
        if model_provider:
            return init_chat_model(model, model_provider=model_provider, **kwargs)
        return init_chat_model(model, **kwargs)


def normalize_region(region: str) -> str | None:
    """Normalize region aliases to standard values.

    Args:
        region: Region string to normalize

    Returns:
        Normalized region ('prc' or 'international') or None if invalid
    """
    if not region:
        return None

    region_lower = region.lower()
    if region_lower in ("prc", "cn"):
        return "prc"
    elif region_lower in ("international", "en"):
        return "international"
    return None


def _register_qwen_provider() -> bool:
    """Register Qwen provider using langchain-dev-utils."""
    if not DEV_UTILS_AVAILABLE:
        return False

    try:
        # Dynamic imports to avoid hard dependencies
        import importlib

        qwq_module = importlib.import_module("langchain_qwq")
        ChatQwen = getattr(qwq_module, "ChatQwen", None)
        ChatQwQ = getattr(qwq_module, "ChatQwQ", None)

        if not ChatQwen or not ChatQwQ:
            return False

        # Register both ChatQwen and ChatQwQ
        register_model_provider("qwen", ChatQwen)  # For regular Qwen models
        register_model_provider("qwq", ChatQwQ)  # For QwQ models

        # Handle region-based URLs
        region = normalize_region(os.getenv("REGION", ""))
        if region == "prc":
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        elif region == "international":
            base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        else:
            base_url = None

        # Register with base URL if needed
        if base_url:
            register_model_provider("qwen", ChatQwen, base_url=base_url)
            register_model_provider("qwq", ChatQwQ, base_url=base_url)

        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False


def _register_siliconflow_provider() -> bool:
    """Register SiliconFlow provider using langchain-dev-utils with proper base URL."""
    if not DEV_UTILS_AVAILABLE:
        return False

    try:
        # Dynamic import to avoid hard dependency
        import importlib

        siliconflow_module = importlib.import_module("langchain_siliconflow")
        ChatSiliconFlow = getattr(siliconflow_module, "ChatSiliconFlow", None)
        if not ChatSiliconFlow:
            return False

        # Handle region-based URLs during registration
        region = normalize_region(os.getenv("REGION", ""))
        if region == "prc":
            base_url = "https://api.siliconflow.cn/v1"
        elif region == "international":
            base_url = "https://api.siliconflow.com/v1"
        else:
            base_url = "https://api.siliconflow.cn/v1"  # Default to PRC

        # Create a factory function that returns ChatSiliconFlow with correct base URL
        def siliconflow_factory(**kwargs: Any) -> Any:
            # Always use our region-specific base URL
            kwargs["base_url"] = base_url
            return ChatSiliconFlow(**kwargs)

        # Register the factory function instead of the raw class
        register_model_provider("siliconflow", siliconflow_factory)

        return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        return False


def _register_openrouter_provider() -> bool:
    """Register OpenRouter provider using langchain-dev-utils."""
    if not DEV_UTILS_AVAILABLE:
        return False

    register_model_provider(
        "openrouter",
        "openai",  # Use OpenAI client
        base_url="https://openrouter.ai/api/v1",
    )
    return True


def register_all_providers() -> dict[str, bool]:
    """Register all available providers."""
    providers_registered = {
        "dev_utils": DEV_UTILS_AVAILABLE,
        "qwen": _register_qwen_provider(),
        "siliconflow": _register_siliconflow_provider(),
        "openrouter": _register_openrouter_provider(),
    }

    return providers_registered


def load_chat_model(model: str, **kwargs: Any) -> ChatModel:
    """Load a chat model with automatic provider registration.

    Automatically registers only the provider needed for the specified model,
    making it efficient and zero-setup.

    Args:
        model: Model specification (e.g., "qwen:qwen-flash", "siliconflow:qwen-plus")
        **kwargs: Additional arguments passed to the underlying load_chat_model
                 (e.g., temperature, max_tokens, timeout, etc.)

    Returns:
        Initialized chat model instance.
    """
    # Extract provider prefix and register only the needed provider
    if ":" in model:
        provider_prefix = model.split(":", 1)[0].lower()

        # Map provider prefixes to registration functions
        provider_registry = {
            "qwen": _register_qwen_provider,
            "qwq": _register_qwen_provider,  # QwQ uses the same registration
            "siliconflow": _register_siliconflow_provider,
            "openrouter": _register_openrouter_provider,
        }

        # Register the specific provider if we have a registration function
        if provider_prefix in provider_registry:
            provider_registry[provider_prefix]()

    # Load the model using langchain-dev-utils or fallback, passing all kwargs
    return _load_chat_model(model, **kwargs)


# Auto-register providers on import
AVAILABLE_PROVIDERS = register_all_providers()
