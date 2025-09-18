"""Dynamic prompt creation utilities for LangGraph agents."""

from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.runtime import get_runtime


def create_context_aware_prompt(state: Any) -> list[AnyMessage]:
    """Create system prompt based on runtime context.

    This function uses get_runtime() to access the agent's context
    and creates a dynamic system prompt based on the configuration.

    Args:
        state: LangGraph state object

    Returns:
        List of messages with system message and existing messages
    """
    runtime = get_runtime()
    context = runtime.context

    # Get system prompt from context
    system_content = getattr(context, "system_prompt", "You are a helpful assistant.")

    # Add contextual information
    user_id = getattr(context, "user_id", None)
    if user_id:
        system_content += f"\n\nUser ID: {user_id}"

    session_id = getattr(context, "session_id", None)
    if session_id:
        system_content += f"\nSession ID: {session_id}"

    # Add current time
    system_content += f"\n\nCurrent time: {datetime.now(UTC).isoformat()}"

    # Add tool availability info
    enable_deepwiki = getattr(context, "enable_deepwiki", False)
    if enable_deepwiki:
        system_content += "\n\nYou have access to DeepWiki for open source documentation."

    max_search_results = getattr(context, "max_search_results", None)
    if max_search_results:
        system_content += f"\n\nWeb searches return up to {max_search_results} results."

    max_data_rows = getattr(context, "max_data_rows", None)
    if max_data_rows:
        system_content += f"\n\nData analysis operations are limited to {max_data_rows} rows."

    system_message = SystemMessage(content=system_content)
    existing_messages = state.get("messages", [])
    if not isinstance(existing_messages, list):
        existing_messages = []
    return [system_message] + existing_messages
