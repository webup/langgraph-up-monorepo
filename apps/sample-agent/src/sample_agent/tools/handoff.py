"""Custom handoff tools following the reference pattern."""

from typing import Annotated, Any

from langchain.messages import ToolMessage
from langchain.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langgraph_supervisor.handoff import METADATA_KEY_HANDOFF_DESTINATION


def create_custom_handoff_tool(*, agent_name: str, name: str | None = None, description: str | None = None) -> BaseTool:
    """Create a custom handoff tool with task description - follows reference pattern."""

    @tool(name or f"handoff_to_{agent_name}", description=description or f"Hand off task to {agent_name}")
    def handoff_to_agent(
        # you can add additional tool call arguments for the LLM to populate
        # for example, you can ask the LLM to populate a task description for the next agent
        task_description: Annotated[
            str, "Detailed description of what the next agent should do, including all of the relevant context."
        ],
        # you can inject the state of the agent that is calling the tool
        state: Annotated[dict[str, Any], InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:  # type: ignore[type-arg]
        tool_message = ToolMessage(
            content=f"Successfully transferred to {agent_name}",
            name=name or f"handoff_to_{agent_name}",
            tool_call_id=tool_call_id,
        )
        messages = state["messages"]
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            # NOTE: this is a state update that will be applied to the swarm multi-agent graph (i.e., the PARENT graph)
            update={
                "messages": messages + [tool_message],
                "active_agent": agent_name,
                # optionally pass the task description to the next agent
                # NOTE: individual agents would need to have `task_description` in their state schema
                # and would need to implement logic for how to consume it
                "task_description": task_description,
            },
        )

    handoff_to_agent.metadata = {METADATA_KEY_HANDOFF_DESTINATION: agent_name}
    return handoff_to_agent
