# Sample Deep Agent

A sophisticated research agent built with the [deepagents](https://github.com/langchain-ai/deepagents) framework, featuring deep web search capabilities, strategic thinking, and virtual file system management.

## Overview

This agent demonstrates the three core patterns of deep agents:

1. **Task Planning**: Structured TODO management for complex workflows
2. **Context Offloading**: Virtual file system for storing research results
3. **Context Isolation**: Specialized sub-agent delegation for focused research

## Architecture

```
sample-deep-agent/
├── Main Deep Agent (Research Coordinator)
│   ├── Task planning and TODO management
│   ├── deep_web_search (AI-powered search with VFS storage)
│   ├── think_tool (Strategic reflection and planning)
│   ├── Sub-agent delegation for research tasks
│   └── Virtual File System integration
├── Research Sub-Agent
│   ├── Structured research planning
│   ├── Systematic investigation execution
│   └── Specialized research capabilities
└── Critique Sub-Agent
    ├── Research validation planning
    └── Quality assessment and feedback
```

## Features

- **Deep Web Search**: Enhanced search with AI summarization and automatic file storage
- **Strategic Thinking**: Built-in reflection tool for research planning and task management
- **Structured Workflows**: Sub-agents follow systematic, planned approaches to research
- **Intelligent Task Management**: Efficient allocation and execution of research tasks
- **Virtual File System**: Persistent storage of research results and context
- **Sub-Agent Delegation**: Specialized research and critique agents for focused work
- **LangGraph Deployment**: Ready for production deployment with LangGraph

## Installation

1. **Install dependencies**:
   ```bash
   cd apps/sample-deep-agent
   uv sync
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys:
   # TAVILY_API_KEY=your_tavily_api_key
   # OPENROUTER_API_KEY=your_openrouter_api_key
   ```

   **Note**: The `.env` file is gitignored to protect your API keys.

## Usage

### Development Server

Start the LangGraph development server:

```bash
# From monorepo root
make dev sample-deep-agent

# Or with specific options
make dev sample-deep-agent -- --no-browser --port 3000
```

### Direct Usage

```python
from sample_deep_agent.graph import make_graph

# Create the agent
agent = make_graph()

# Run a research task
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Research the latest developments in AI safety"}]
})

print(result["messages"][-1].content)
```

### Configuration

Customize the agent behavior via runtime configuration:

```python
config = {
    "configurable": {
        "model_name": "openrouter:anthropic/claude-sonnet-4",
        "max_search_results": 3,
        "max_concurrent_research_units": 2
    }
}

agent = make_graph(config)
```

## Example Workflows

### Simple Research Query

```
User: "What are the main challenges in renewable energy adoption?"

Agent Process:
1. Main coordinator creates research plan and delegates to specialized sub-agent
2. Research sub-agent develops structured approach to investigation
3. Sub-agent systematically executes research using deep_web_search
4. Results stored in virtual file system with AI summaries
5. Main coordinator synthesizes findings into comprehensive report
6. May delegate to critique sub-agent for validation if needed
```

### Complex Multi-Faceted Research

```
User: "Compare the environmental impact, costs, and adoption rates of solar vs wind energy"

Agent Process:
1. Main coordinator develops strategic plan for complex comparison
2. Delegates to research sub-agent with specific research mandate
3. Research sub-agent plans systematic approach covering:
   - Environmental impacts (solar vs wind)
   - Cost analysis from multiple sources
   - Adoption rate comparisons
4. Sub-agent systematically executes research using deep_web_search
5. All results stored in organized VFS structure with AI summaries
6. Main coordinator synthesizes comparative analysis
7. May delegate to critique sub-agent for comprehensive validation
```

## Key Components

### Main Deep Agent (Research Coordinator)

The main agent orchestrates research through:
- **Task Management**: Efficient planning and allocation of research tasks
- **Strategic Delegation**: Delegates complex research to specialized sub-agents
- **Synthesis & Coordination**: Compiles sub-agent findings into comprehensive reports
- **File System Integration**: Manages virtual file system for persistent context
- **Quality Control**: May delegate to critique sub-agent for validation

### Research Sub-Agent

Specialized for systematic investigation with:
- **Structured Planning**: Develops focused, systematic approaches to research
- **Methodical Execution**: Follows planned workflow to ensure thoroughness
- **Research Tools**: Expert use of deep_web_search and strategic thinking
- **Focused Expertise**: Concentrates on specific research mandates

### Critique Sub-Agent

Specialized for research validation with:
- **Systematic Assessment**: Planned approach to comprehensive evaluation
- **Quality Assessment**: Evaluates research findings for completeness and accuracy
- **Independent Perspective**: Provides unbiased validation and improvement recommendations
- **Validation Planning**: Structures critique process for thorough review

### Virtual File System

Automatic storage of:
- Search results with AI summaries
- Full webpage content as markdown
- Organized file naming for easy retrieval
- Persistent context across research sessions

## Testing

Run the test suite:

```bash
# Unit tests
make unit sample-deep-agent

# Integration tests (requires API keys)
make integration sample-deep-agent

# All tests
make test
```

## Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking included in lint
```

### Project Structure

```
src/sample_deep_agent/
├── __init__.py
├── context.py          # Runtime configuration schema
├── graph.py            # Main deep agent creation and export
├── prompts.py          # Agent instructions and prompts
├── subagents.py        # Critique sub-agent configuration
└── tools/              # Custom tools (currently empty)
    └── __init__.py
```

## API Reference

### Context Configuration

```python
class DeepAgentContext(BaseModel):
    model_name: str = "openrouter:anthropic/claude-sonnet-4"
    recursion_limit: int = 1000
    summarization_model: str = "openrouter:google/gemini-2.5-flash-lite"
    max_todos: int  # Configurable task limit per session
```

### Sub-Agent Configuration

```python
# Research Sub-Agent - Primary research delegation target
research_sub_agent = {
    "name": "research-agent",
    "description": "Specialized for systematic research investigation",
    "prompt": SUB_RESEARCH_PROMPT
}

# Critique Sub-Agent - Research validation and quality assessment
critique_sub_agent = {
    "name": "critique-agent",
    "description": "Specialized for research validation and quality assessment",
    "prompt": SUB_CRITIQUE_PROMPT
}

# Available agent configurations
RESEARCH_AGENTS = [research_sub_agent]  # Primary research workflow
ALL_SUBAGENTS = [research_sub_agent, critique_sub_agent]  # Full capabilities
```

## Comparison with Sample-Agent

| Feature | Sample-Agent | Sample-Deep-Agent |
|---------|--------------|-------------------|
| Framework | langgraph-supervisor | deepagents |
| Architecture | Supervisor pattern | Deep agent with specialized sub-agents |
| Memory | Message history only | Virtual file system + task state |
| Tools | Basic tools + handoff | deep_web_search + think_tool |
| Delegation | Handoff between agents | Structured task delegation |
| Research | Limited web search | AI-enhanced search with summarization |
| Planning | None | Intelligent task management |
| Workflow | Ad-hoc conversation | Systematic, planned execution |
| Sub-Agent Behavior | Independent operation | Structured, methodical workflows |

## Contributing

1. Follow the existing code patterns
2. Add tests for new functionality
3. Update documentation as needed
4. Run `make lint` before committing

## License

MIT