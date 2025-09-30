"""Context schemas for configuring LangGraph agents."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields


@dataclass(kw_only=True)
class BaseAgentContext:
    """Base context schema with common configuration for all agents."""

    model: str = field(
        default="openai:openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "  # noqa: E501
            "Should be in the form: provider:model-name.",
        },
    )

    user_id: str | None = field(
        default=None,
        metadata={
            "description": "User ID for personalization and logging.",
        },
    )

    def __post_init__(self) -> None:
        """Load configuration from environment variables if not explicitly set."""
        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            default_value = f.default
            env_var_name = f.name.upper()
            env_value = os.environ.get(env_var_name)

            # Only override with environment variable if current value equals default
            # This preserves explicit configuration from LangGraph configurable
            # Skip empty environment variables
            if current_value == default_value and env_value is not None and env_value.strip():
                if isinstance(default_value, bool):
                    # Handle boolean environment variables
                    env_bool_value = env_value.lower() in ("true", "1", "yes", "on")
                    setattr(self, f.name, env_bool_value)
                elif isinstance(default_value, int):
                    # Handle integer environment variables
                    try:
                        setattr(self, f.name, int(env_value))
                    except ValueError:
                        pass  # Keep default if conversion fails
                else:
                    setattr(self, f.name, env_value)


@dataclass(kw_only=True)
class SearchContext(BaseAgentContext):
    """Context mixin for agents with search capabilities."""

    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return for each search query.",  # noqa: E501
        },
    )

    enable_deepwiki: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable the DeepWiki MCP tool for accessing open source project documentation.",  # noqa: E501
        },
    )

    include_raw_content: str = field(
        default="markdown",
        metadata={
            "description": "Format for raw content inclusion in search results. Options: 'none', 'text', 'markdown'.",  # noqa: E501
        },
    )


@dataclass(kw_only=True)
class DataContext(BaseAgentContext):
    """Context mixin for agents with data analysis capabilities."""

    max_data_rows: int = field(
        default=1000,
        metadata={
            "description": "Maximum number of data rows to process at once.",
        },
    )

    enable_data_viz: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable data visualization capabilities.",
        },
    )


@dataclass(kw_only=True)
class DataAnalystContext(SearchContext, DataContext):
    """Specialized context for data analyst agents."""

    system_prompt: str = field(
        default="You are a data analyst assistant specializing in data analysis, visualization, and insights. "  # noqa: E501
        "You have access to tools for analyzing data, creating visualizations, and searching for relevant information.",  # noqa: E501
        metadata={
            "description": "The system prompt to use for the data analyst agent.",
        },
    )

    # Override default for data analyst
    max_search_results: int = field(
        default=8,  # Data analysts might need more search results
        metadata={
            "description": "The maximum number of search results to return for each search query.",  # noqa: E501
        },
    )


@dataclass(kw_only=True)
class ResearchContext(SearchContext):
    """Specialized context for research assistant agents."""

    system_prompt: str = field(
        default="You are a research assistant specializing in finding, analyzing, and synthesizing information from various sources. "  # noqa: E501
        "You have access to web search and documentation tools to help with research tasks.",  # noqa: E501
        metadata={
            "description": "The system prompt to use for the research assistant agent.",
        },
    )

    # Override defaults for research assistant
    enable_deepwiki: bool = field(
        default=True,  # Research assistants typically need documentation access
        metadata={
            "description": "Whether to enable the DeepWiki MCP tool for accessing open source project documentation.",  # noqa: E501
        },
    )

    max_search_results: int = field(
        default=10,  # Research assistants need more comprehensive results
        metadata={
            "description": "The maximum number of search results to return for each search query.",  # noqa: E501
        },
    )
