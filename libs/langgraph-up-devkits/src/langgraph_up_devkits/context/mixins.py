"""Context mixins for specialized agent capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BaseAgentContext


@dataclass(kw_only=True)
class SearchContext(BaseAgentContext):
    """Context mixin for agents with search capabilities.

    Adds search-specific configuration:
    - max_search_results: Limit on number of search results
    - enable_deepwiki: Toggle for DeepWiki documentation access
    - include_raw_content: Format for raw content in search results

    Example:
        ```python
        from langgraph_up_devkits.context import SearchContext

        @dataclass(kw_only=True)
        class MySearchAgent(SearchContext):
            custom_field: str = field(default="value")
        ```
    """

    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return for each search query.",
        },
    )

    enable_deepwiki: bool = field(
        default=False,
        metadata={
            "description": "Whether to enable the DeepWiki MCP tool for accessing open source project documentation.",
        },
    )

    include_raw_content: str = field(
        default="markdown",
        metadata={
            "description": "Format for raw content inclusion in search results. Options: 'none', 'text', 'markdown'.",
        },
    )


@dataclass(kw_only=True)
class DataContext(BaseAgentContext):
    """Context mixin for agents with data analysis capabilities.

    Adds data-specific configuration:
    - max_data_rows: Limit on data rows to process
    - enable_data_viz: Toggle for data visualization features

    Example:
        ```python
        from langgraph_up_devkits.context import DataContext

        @dataclass(kw_only=True)
        class MyDataAgent(DataContext):
            custom_field: str = field(default="value")
        ```
    """

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


__all__ = ["SearchContext", "DataContext"]
