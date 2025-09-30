"""Research agent definition."""

from typing import Any

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph_up_devkits import load_chat_model

from sample_agent.context import SupervisorContext
from sample_agent.prompts import RESEARCH_EXPERT_PROMPT

from ..tools.basic import web_search


def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Make research agent graph based on runtime configuration.

    Args:
        config: Optional runtime configuration containing model settings.

    Returns:
        Compiled research agent graph ready for deployment.
    """
    if config is None:
        config = {}

    # Convert runnable config to context
    configurable = config.get("configurable", {})
    context_kwargs = {k: v for k, v in configurable.items() if k in SupervisorContext.model_fields}
    context = SupervisorContext(**context_kwargs)

    # Load model based on configuration
    model = load_chat_model(context.model_name)

    # Create and return the research agent directly
    agent = create_agent(
        model=model,
        tools=[web_search],
        name="research_expert",
        prompt=RESEARCH_EXPERT_PROMPT,
    )
    return agent.compile() if hasattr(agent, "compile") else agent
