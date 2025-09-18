"""Simple LangGraph agent1 for monorepo testing."""

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from agent1.state import State
from shared import get_dummy_message, get_shared_timestamp
from common import get_common_prefix


def call_model(state: State) -> dict:
    """Simple node that uses the shared libraries."""
    # Use functions from both shared packages
    dummy_message = get_dummy_message()
    timestamp = get_shared_timestamp()
    prefix = get_common_prefix()
    
    message = AIMessage(
        content=f"{prefix} Agent1 says: {dummy_message} ({timestamp})"
    )
    
    return {"messages": [message]}


def should_continue(state: State):
    """Conditional edge - end after first message."""
    messages = state["messages"]
    if len(messages) > 0:
        return END
    return "call_model"


# Build the graph
workflow = StateGraph(State)

# Add the node
workflow.add_node("call_model", call_model)

# Add edges
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue)

graph = workflow.compile()