"""Deep agent with research capabilities using deepagents framework."""

from typing import Any

from deepagents import create_deep_agent  # type: ignore[import-untyped]
from langchain.agents.middleware import InterruptOnConfig
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph_up_devkits import load_chat_model
from langgraph_up_devkits.tools import deep_web_search, think_tool

from sample_deep_agent.context import DeepAgentContext
from sample_deep_agent.prompts import get_research_instructions
from sample_deep_agent.subagents import RESEARCH_AGENTS

# Re-export tools for test imports
__all__ = ["make_graph", "app", "deep_web_search", "think_tool"]


def make_graph(
    config: RunnableConfig | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    subagent_interrupts: dict[str, dict[str, bool | InterruptOnConfig]] | None = None,
) -> Any:
    """Make deep agent graph based on runtime configuration.

    Args:
        config: Optional runtime configuration containing model settings.
        interrupt_on: Optional interrupt configuration mapping tool names to interrupt settings.
            - True: Enable interrupts with default behavior (approve, edit, reject allowed)
            - False: Disable interrupts for this tool
            - InterruptOnConfig: Custom configuration with specific allowed decisions
        subagent_interrupts: Optional subagent-specific interrupt overrides mapping subagent
            names to their interrupt_on configurations.

    Returns:
        Compiled deep agent graph ready for deployment.

    Note:
        When interrupt_on is provided, a MemorySaver checkpointer is automatically configured.
    """
    if config is None:
        config = {}

    # Convert runnable config to context
    configurable = config.get("configurable", {})
    from dataclasses import fields

    context_field_names = {f.name for f in fields(DeepAgentContext)}
    context_kwargs = {k: v for k, v in configurable.items() if k in context_field_names}
    context = DeepAgentContext(**context_kwargs)

    # Load model based on context configuration
    model = load_chat_model(context.model)

    # Automatically add checkpointer if interrupts are enabled
    checkpointer = MemorySaver() if interrupt_on is not None else None

    # Prepare subagents with interrupt overrides
    subagents = RESEARCH_AGENTS
    if subagent_interrupts:
        # Apply subagent-specific interrupt configurations
        subagents = []
        for subagent in RESEARCH_AGENTS:
            subagent_name = subagent["name"]
            if subagent_name in subagent_interrupts:
                # Create a copy with interrupt_on override
                subagent_copy = subagent.copy()
                subagent_copy["interrupt_on"] = subagent_interrupts[subagent_name]
                subagents.append(subagent_copy)
            else:
                subagents.append(subagent)

    # Create deep agent with research capabilities and interrupt support
    agent = create_deep_agent(
        model=model,
        tools=[deep_web_search, think_tool],
        system_prompt=get_research_instructions(),
        subagents=subagents,
        context_schema=DeepAgentContext,
        interrupt_on=interrupt_on,
        checkpointer=checkpointer,
    ).with_config({"recursion_limit": context.recursion_limit})

    return agent


# Export as 'app' for LangGraph deployment
app = make_graph()
