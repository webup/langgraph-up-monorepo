"""Sub-agent definitions for deep agent framework."""

from deepagents import SubAgent  # type: ignore[import-untyped]
from langgraph_up_devkits.middleware import FileSystemMaskMiddleware

from sample_deep_agent.prompts import SUB_CRITIQUE_PROMPT, SUB_RESEARCH_PROMPT

# Initialize FileSystemMaskMiddleware for sub-agents
filesystem_mask = FileSystemMaskMiddleware()

# Sub-agent configurations
research_sub_agent: SubAgent = {
    "name": "research-agent",
    "description": (
        "Used to research more in depth questions. Only give this researcher one topic at a time. "
        "Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic "
        "into the necessary components, and then call multiple research agents in parallel, "
        "one for each sub question."
    ),
    "system_prompt": SUB_RESEARCH_PROMPT,
    "middleware": [filesystem_mask],
}

critique_sub_agent: SubAgent = {
    "name": "critique-agent",
    "description": (
        "Used to critique the final report. Give this agent some information about "
        "how you want it to critique the report."
    ),
    "system_prompt": SUB_CRITIQUE_PROMPT,
    "middleware": [filesystem_mask],
}

# Agent groups for different configurations
RESEARCH_AGENTS: list[SubAgent] = [research_sub_agent]
ALL_SUBAGENTS: list[SubAgent] = [research_sub_agent, critique_sub_agent]
