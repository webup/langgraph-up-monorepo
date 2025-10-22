"""State definition for sample-agent extending AgentState."""

from typing import NotRequired

from langchain.agents import AgentState as BaseAgentState


class AgentState(BaseAgentState):  # type: ignore[type-arg]
    """State for sample-agent with additional fields.

    Extends langchain.agents.AgentState which provides:
    - messages: Annotated[list[BaseMessage], add_messages]
    - jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    - structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
    """

    # Required by create_react_agent
    remaining_steps: int

    # Additional fields for supervisor pattern
    task_description: NotRequired[str | None]
    active_agent: NotRequired[str | None]


__all__ = ["AgentState"]
