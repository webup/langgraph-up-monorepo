"""Subagents package for sample_agent."""

from .math import make_graph as make_math_graph
from .research import make_graph as make_research_graph

__all__ = [
    "make_math_graph",
    "make_research_graph",
]
