"""State definition for Agent2."""

from typing import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """State for Agent2."""
    messages: Annotated[list[BaseMessage], add_messages]