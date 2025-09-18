"""Context schema for supervisor configuration."""

from typing import Any

from pydantic import BaseModel


class SupervisorContext(BaseModel):
    """Context schema for supervisor configuration."""

    temperature: float = 0.7
    max_tokens: int | None = None
    debug_mode: bool = False

    @classmethod
    def default(cls) -> "SupervisorContext":
        """Create default supervisor context."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for RunnableConfig."""
        return {"configurable": self.model_dump()}
