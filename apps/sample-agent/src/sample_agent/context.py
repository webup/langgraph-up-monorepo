"""Context schema for supervisor configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langgraph_up_devkits.context import BaseAgentContext


@dataclass(kw_only=True)
class SupervisorContext(BaseAgentContext):
    """Context schema for supervisor configuration.

    Extends BaseAgentContext with supervisor-specific defaults.
    Uses GLM-4.5-Air model by default for efficient coordination.

    Inherits from BaseAgentContext:
    - model: LLM identifier (overridden to siliconflow:zai-org/GLM-4.5-Air)
    - temperature: Sampling temperature (default 0.7)
    - max_tokens: Response token cap (default None)
    - recursion_limit: LangGraph recursion depth (default 100)
    - debug: Enable debug logging
    - user_id: Optional user identifier
    """

    # Override model default for supervisor
    model: str = field(
        default="siliconflow:zai-org/GLM-4.5-Air",
        metadata={
            "description": "The name of the language model to use for the supervisor agent.",
        },
    )

    @classmethod
    def default(cls) -> SupervisorContext:
        """Create default supervisor context."""
        return cls()

    def to_dict(self) -> dict[str, dict[str, str | float | int | bool | None]]:
        """Convert to dictionary for RunnableConfig."""
        return {"configurable": asdict(self)}
