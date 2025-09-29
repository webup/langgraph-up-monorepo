"""Simple state definition for Agent1 extending MessagesState."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Simple state for sample-agent with task description support."""

    # Core message history
    messages: Annotated[list[BaseMessage], add_messages]

    # Required for create_react_agent
    remaining_steps: int

    # Task management - following the reference pattern
    task_description: str | None

    # Active agent tracking
    active_agent: str | None
