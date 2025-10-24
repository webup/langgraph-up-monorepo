"""Base context schema for LangGraph agent configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass(kw_only=True)
class BaseAgentContext:
    """Base context schema with common configuration for all agents.

    This base class provides:
    - model: LLM identifier for model loading (provider:model-name format)
    - temperature: Sampling temperature for generation
    - max_tokens: Optional token limit for responses
    - recursion_limit: Maximum LangGraph recursion depth
    - debug: Enable debug logging in middleware
    - user_id: Optional user identifier for personalization

    All agent-specific contexts should extend this base context.

    The context automatically loads values from environment variables if not
    explicitly set, with the following precedence:
    1. Explicitly provided values (highest priority)
    2. Environment variables (FIELD_NAME in uppercase)
    3. Default values (lowest priority)

    Example:
        ```python
        from langgraph_up_devkits.context import BaseAgentContext

        @dataclass(kw_only=True)
        class MyAgentContext(BaseAgentContext):
            custom_field: str = field(
                default="value",
                metadata={"description": "Custom configuration"}
            )
        ```
    """

    model: str = field(
        default="siliconflow:zai-org/GLM-4.5",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider:model-name (e.g., 'openrouter:anthropic/claude-sonnet-4').",
        },
    )

    temperature: float = field(
        default=0.7,
        metadata={
            "description": "Sampling temperature for the chat model (0.0 to 2.0).",
        },
    )

    max_tokens: int | None = field(
        default=None,
        metadata={
            "description": "Optional response token cap for model generation.",
        },
    )

    recursion_limit: int = field(
        default=100,
        metadata={
            "description": "Maximum LangGraph recursion depth for agent execution.",
        },
    )

    debug: bool = field(
        default=False,
        metadata={
            "description": "Enable debug logging in middleware and agent components.",
        },
    )

    user_id: str | None = field(
        default=None,
        metadata={
            "description": "User ID for personalization and logging.",
        },
    )

    def __post_init__(self) -> None:
        """Load configuration from environment variables if not explicitly set.

        This method runs after dataclass initialization and loads values from
        environment variables when:
        - The current value equals the default value (not explicitly set)
        - An environment variable exists with the field name in uppercase
        - The environment variable is not empty

        Supports automatic type conversion for:
        - bool: Accepts "true", "1", "yes", "on" (case-insensitive)
        - int: Converts string to integer, keeps default on failure
        - str: Uses value as-is
        """
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            default_value = f.default
            env_var_name = f.name.upper()
            env_value = os.environ.get(env_var_name)

            # Only override with environment variable if current value equals default
            # This preserves explicit configuration from LangGraph configurable
            # Skip empty environment variables
            if current_value == default_value and env_value is not None and env_value.strip():
                if isinstance(default_value, bool):
                    # Handle boolean environment variables
                    env_bool_value = env_value.lower() in ("true", "1", "yes", "on")
                    setattr(self, f.name, env_bool_value)
                elif isinstance(default_value, int):
                    # Handle integer environment variables
                    try:
                        setattr(self, f.name, int(env_value))
                    except ValueError:
                        pass  # Keep default if conversion fails
                elif isinstance(default_value, float):
                    # Handle float environment variables
                    try:
                        setattr(self, f.name, float(env_value))
                    except ValueError:
                        pass  # Keep default if conversion fails
                else:
                    setattr(self, f.name, env_value)


__all__ = ["BaseAgentContext"]
