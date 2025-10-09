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


def _get_qwen_base_url() -> str:
    """Get Qwen base URL based on region."""
    region = normalize_region(os.getenv("REGION", ""))
    return (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        if region == "international"
        else "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def _get_siliconflow_base_url() -> str:
    """Get SiliconFlow base URL based on region."""
    region = normalize_region(os.getenv("REGION", ""))
    return "https://api.siliconflow.com/v1" if region == "international" else "https://api.siliconflow.cn/v1"


def _register_qwen_provider() -> bool:
    """Register Qwen provider using langchain-dev-utils, fallback to OpenAI."""
    if not DEV_UTILS_AVAILABLE:
        return False

    base_url = _get_qwen_base_url()

    try:
        # Try to use native langchain_qwq package
        import importlib

        qwq_module = importlib.import_module("langchain_qwq")
        ChatQwen = getattr(qwq_module, "ChatQwen", None)
        ChatQwQ = getattr(qwq_module, "ChatQwQ", None)

        if ChatQwen and ChatQwQ:
            register_model_provider("qwen", ChatQwen, base_url=base_url)
            register_model_provider("qwq", ChatQwQ, base_url=base_url)
            return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    # Fallback to OpenAI client
    register_model_provider("qwen", "openai", base_url=base_url)
    register_model_provider("qwq", "openai", base_url=base_url)
    return True


def _register_siliconflow_provider() -> bool:
    """Register SiliconFlow provider using langchain-dev-utils, fallback to OpenAI."""
    if not DEV_UTILS_AVAILABLE:
        return False

    base_url = _get_siliconflow_base_url()

    try:
        # Try to use native langchain_siliconflow package
        import importlib

        siliconflow_module = importlib.import_module("langchain_siliconflow")
        ChatSiliconFlow = getattr(siliconflow_module, "ChatSiliconFlow", None)

        if ChatSiliconFlow:
            # Create factory to inject base_url
            def siliconflow_factory(**kwargs: Any) -> Any:
                kwargs["base_url"] = base_url
                return ChatSiliconFlow(**kwargs)

            register_model_provider("siliconflow", siliconflow_factory)
            return True
    except (ImportError, AttributeError, ModuleNotFoundError):
        pass

    # Fallback to OpenAI client
    register_model_provider("siliconflow", "openai", base_url=base_url)
    return True


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
