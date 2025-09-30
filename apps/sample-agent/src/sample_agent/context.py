"""Context schema for supervisor configuration."""

from typing import Any

from pydantic import BaseModel, Field


class SupervisorContext(BaseModel):
    """Context schema for supervisor configuration."""

    model_name: str = Field(default="siliconflow:zai-org/GLM-4.5-Air", description="Default model name")
    temperature: float = 0.7
    max_tokens: int | None = None
    debug_mode: bool = False
    recursion_limit: int = Field(default=100, description="Recursion limit for agent execution")

    @classmethod
    def default(cls) -> "SupervisorContext":
        """Create default supervisor context."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for RunnableConfig."""
        return {"configurable": self.model_dump()}
