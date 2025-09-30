"""Simple supervisor pattern sample agent with custom handoff tools and message forwarding."""

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph_supervisor import create_supervisor
from langgraph_supervisor.handoff import create_forward_message_tool
from langgraph_up_devkits import load_chat_model

from sample_agent.context import SupervisorContext
from sample_agent.prompts import SUPERVISOR_PROMPT
from sample_agent.state import AgentState
from sample_agent.subagents import make_math_graph, make_research_graph
from sample_agent.tools.handoff import create_custom_handoff_tool


# This is the graph making function that will decide which graph to
# build based on the provided config
def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Make graph based on runtime configuration - rebuilds on each call."""
    if config is None:
        config = {}

    # Convert runnable config to context
    configurable = config.get("configurable", {})
    context_kwargs = {k: v for k, v in configurable.items() if k in SupervisorContext.model_fields}
    context = SupervisorContext(**context_kwargs)

    # Load model based on configuration
    model = load_chat_model(context.model_name)

    # Create agents with the configured model via make_graph functions
    math_agent = make_math_graph(config)
    research_agent = make_research_graph(config)

    # Create handoff tools
    math_handoff = create_custom_handoff_tool(
        agent_name="math_expert",
        name="assign_to_math_expert",
        description="Assign mathematical calculations and analysis tasks to the math expert",
    )
    research_handoff = create_custom_handoff_tool(
        agent_name="research_expert",
        name="assign_to_research_expert",
        description="Assign research and information gathering tasks to the research expert",
    )
    forwarding_tool = create_forward_message_tool("supervisor")

    # Create supervisor workflow
    workflow = create_supervisor(
        [research_agent, math_agent],
        model=model,
        tools=[math_handoff, research_handoff, forwarding_tool],
        state_schema=AgentState,
        prompt=SUPERVISOR_PROMPT,
        output_mode="last_message",
        add_handoff_messages=True,
    )
    return workflow.compile(name="supervisor").with_config({"recursion_limit": context.recursion_limit})
