# Sample Agent

A sample LangGraph agent demonstrating the supervisor pattern with custom handoff tools and message forwarding.

## Overview

This sample agent showcases:
- **Supervisor Pattern**: Central coordination between specialized sub-agents
- **Custom Handoff Tools**: Seamless agent-to-agent communication
- **Message Forwarding**: Efficient information flow between agents
- **Specialized Sub-agents**: Math expert and research expert agents

## Architecture

```
┌─────────────────┐
│   Supervisor    │
│     Agent       │
└─────────┬───────┘
          │
    ┌─────┴─────┐
    │           │
┌───▼───┐   ┌───▼───┐
│ Math  │   │Research│
│Expert │   │Expert │
└───────┘   └───────┘
```

## Sub-agents

- **Math Expert**: Handles mathematical calculations using `add` and `multiply` tools
- **Research Expert**: Performs web searches and research tasks

## Quick Start

### Development Server

```bash
# Start development server
make dev sample-agent

# Start without opening browser
make dev sample-agent -- --no-browser

# With custom host/port
make dev sample-agent -- --host 0.0.0.0 --port 3000
```

### Testing

```bash
# Run unit tests
make unit sample-agent

# Run integration tests
make integration sample-agent

# Run all tests
make test_apps
```

### Linting & Formatting

```bash
# Lint code
make lint_apps

# Format code
make format_apps
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and set:

```bash
# OpenRouter API Key
OPENROUTER_API_KEY=your_key_here

# LangSmith (optional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=your_project_name
LANGCHAIN_API_KEY=your_langsmith_key
```

### Agent Configuration

The agent uses:
- **Model**: Anthropic Claude Sonnet 4 via OpenRouter
- **Dependencies**: Shared libs (`shared`, `common`)
- **Graph Entry Point**: `src/sample_agent/graph.py:make_graph`

## Project Structure

```
apps/sample-agent/
├── README.md                   # This file
├── pyproject.toml             # Package configuration
├── langgraph.json             # LangGraph configuration
├── .env                       # Environment variables
├── .env.example               # Environment template
├── src/sample_agent/          # Main package
│   ├── __init__.py
│   ├── graph.py              # Main graph definition
│   ├── state.py              # Agent state schema
│   ├── prompts.py            # Agent prompts
│   ├── context.py            # Context utilities
│   ├── subagents/            # Sub-agent definitions
│   │   ├── math.py           # Math expert agent
│   │   └── research.py       # Research expert agent
│   └── tools/                # Tool definitions
│       ├── basic.py          # Basic tools (add, multiply, web_search)
│       └── handoff.py        # Custom handoff tools
└── tests/                    # Test suite
    ├── conftest.py           # Test configuration
    ├── unit/                 # Unit tests
    └── integration/          # Integration tests
```

## Usage Examples

### Using the API

```python
from sample_agent.graph import make_graph

# Create the graph
graph = make_graph()

# Use the agent
result = await graph.ainvoke({
    "messages": [{"role": "user", "content": "Calculate 15 * 24"}]
})
```

### Development

```python
# Import components
from sample_agent.state import AgentState
from sample_agent.subagents import create_math_agent, create_research_agent
from sample_agent.tools.handoff import create_custom_handoff_tool
```

## Extending the Agent

1. **Add new sub-agents** in `src/sample_agent/subagents/`
2. **Add new tools** in `src/sample_agent/tools/`
3. **Update prompts** in `src/sample_agent/prompts.py`
4. **Modify the graph** in `src/sample_agent/graph.py`

## API Endpoints

When running the dev server:

- **API**: http://127.0.0.1:2024
- **Studio UI**: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- **API Docs**: http://127.0.0.1:2024/docs

## Dependencies

- `langgraph-supervisor>=0.0.28` - Supervisor pattern implementation
- `langgraph-up-devkits` - Development utilities (workspace dependency)
- Shared libraries from `libs/shared` and `libs/common`

## License

MIT