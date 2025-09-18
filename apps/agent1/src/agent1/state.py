"""State definition for Agent1."""

from typing import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """State for Agent1."""
    messages: Annotated[list[BaseMessage], add_messages]