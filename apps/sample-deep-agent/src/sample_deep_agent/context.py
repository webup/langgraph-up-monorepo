"""Context configuration for sample deep agent."""

from __future__ import annotations

from dataclasses import dataclass, field

from langgraph_up_devkits.context import BaseAgentContext

# Constants
MAX_TODOS = 2  # Global maximum number of TODOs allowed per session


@dataclass(kw_only=True)
class DeepAgentContext(BaseAgentContext):
    """Context configuration for deep agent runtime settings.

    Deep agent specific configuration:
    - Uses DeepSeek-V3.2-Exp model for advanced reasoning
    - Higher recursion limit (1000) for complex research workflows
    - Configurable max_todos for research task management

    Inherits from BaseAgentContext:
    - model: LLM identifier (overridden to siliconflow:deepseek-ai/DeepSeek-V3.2-Exp)
    - temperature: Sampling temperature (default 0.7)
    - max_tokens: Response token cap (default None)
    - recursion_limit: LangGraph recursion depth (overridden to 1000)
    - debug: Enable debug logging
    - user_id: Optional user identifier

    For HITL interrupts, pass interrupt_on directly to make_graph() instead of using context.
    """

    # Override model default for deep agent
    model: str = field(
        default="siliconflow:deepseek-ai/DeepSeek-V3.2-Exp",
        metadata={
            "description": "The name of the language model to use for deep reasoning tasks.",
        },
    )

    # Override recursion limit for deep agent
    recursion_limit: int = field(
        default=1000,
        metadata={
            "description": "Maximum LangGraph recursion depth for complex research workflows.",
        },
    )

    # Research workflow settings
    max_todos: int = field(
        default=MAX_TODOS,
        metadata={
            "description": "Maximum number of TODOs to create for research tasks.",
        },
    )
