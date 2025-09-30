"""Context configuration for sample deep agent."""

from pydantic import BaseModel, Field

# Constants
MAX_TODOS = 2  # Global maximum number of TODOs allowed per session


class DeepAgentContext(BaseModel):
    """Context configuration for deep agent runtime settings."""

    # Model configuration
    model_name: str = Field(default="siliconflow:zai-org/GLM-4.5", description="Default model name")

    # Graph configuration
    recursion_limit: int = Field(default=1000, description="Recursion limit for agent execution")

    # Research workflow settings
    max_todos: int = Field(default=MAX_TODOS, description="Maximum number of TODOs to create for research tasks")
