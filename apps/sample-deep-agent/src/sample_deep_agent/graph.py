"""Deep agent with research capabilities using deepagents framework."""

from typing import Any

from deepagents import async_create_deep_agent  # type: ignore[import-untyped]
from langchain_core.runnables import RunnableConfig
from langgraph_up_devkits import load_chat_model
from langgraph_up_devkits.tools import deep_web_search, think_tool

from sample_deep_agent.context import DeepAgentContext
from sample_deep_agent.prompts import get_research_instructions
from sample_deep_agent.subagents import RESEARCH_AGENTS

# Re-export tools for test imports
__all__ = ["make_graph", "app", "deep_web_search", "think_tool"]


def make_graph(config: RunnableConfig | None = None) -> Any:
    """Make deep agent graph based on runtime configuration.

    Args:
        config: Optional runtime configuration containing model settings.

    Returns:
        Compiled deep agent graph ready for deployment.
    """
    if config is None:
        config = {}

    # Convert runnable config to context
    configurable = config.get("configurable", {})
    context_kwargs = {k: v for k, v in configurable.items() if k in DeepAgentContext.model_fields}
    context = DeepAgentContext(**context_kwargs)

    # Load model based on context configuration
    model = load_chat_model(context.model_name)

    # Create deep agent with research capabilities (remove research_sub_agent from subagents list)
    agent = async_create_deep_agent(
        tools=[deep_web_search, think_tool],
        instructions=get_research_instructions(),
        subagents=RESEARCH_AGENTS,  # Research agent in subagents list
        model=model,
        context_schema=DeepAgentContext,
    ).with_config({"recursion_limit": context.recursion_limit})

    return agent


# Export as 'app' for LangGraph deployment
app = make_graph()
