# 🚀 LangGraph-UP Monorepo

**LangGraph-UP Monorepo** is a batteries-included monorepo framework for building sophisticated LangGraph applications. Think "Next.js for LangGraph" - opinionated, enterprise-ready, and designed for developer productivity.

[![CI](https://github.com/webup/langgraph-up-monorepo/workflows/CI/badge.svg)](https://github.com/your-org/langgraph-up-monorepo/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/uv-package%20manager-green.svg)](https://github.com/astral-sh/uv)

## ✨ Key Features

- 🌐 **Universal Model Loading** - OpenRouter, Qwen, QwQ, SiliconFlow with automatic registration
- 🤖 **Multi-Agent Orchestration** - Supervisor patterns, handoffs, and collaboration workflows
- 🛠 **Production Middleware** - Summarization, context management, error handling
- 🧪 **Developer Experience** - Hot reload, comprehensive testing, strict linting
- 🚀 **Deployment Ready** - LangGraph Cloud configurations included
- 🌍 **Global Ready** - Region-based provider configuration (PRC/International)

## 🚀 Quick Start

### Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/your-org/langgraph-up-monorepo.git
cd langgraph-up-monorepo
uv sync
```

### 30-Second Demo

```python
from langgraph_up_devkits import load_chat_model

# Zero-setup model loading across providers
model = load_chat_model("openrouter:anthropic/claude-sonnet-4")
# model = load_chat_model("qwen:qwen-flash")
# model = load_chat_model("siliconflow:Qwen/Qwen3-8B")

# Start building your agent
from sample_agent import make_graph

app = make_graph()
result = await app.ainvoke({"messages": [{"role": "user", "content": "What's 25 * 4?"}]})
```

### Run Sample Agent

```bash
# Start development server
make dev sample-agent

# Or without browser
make dev sample-agent -- --no-browser
```

## 🏗 Architecture

### Monorepo Structure

```
langgraph-up-monorepo/
├── libs/
│   ├── shared/                    # Shared utilities
│   ├── common/                    # Common helper functions
│   └── langgraph-up-devkits/      # 🎯 Core framework
│       ├── utils/providers.py     #   → Multi-provider model loading
│       ├── middleware/            #   → Production middleware
│       ├── tools/                 #   → Web search, fetch, MCP
│       └── context.py             #   → Context-aware prompts
├── apps/
│   └── sample-agent/              # 🤖 Multi-agent supervisor example
│       ├── src/sample_agent/
│       │   ├── graph.py           #   → Main supervisor graph
│       │   ├── subagents/         #   → Math & research experts
│       │   └── tools/             #   → Agent-specific tools
│       └── langgraph.json         #   → Deployment config
├── pyproject.toml                 # Root dependencies
├── Makefile                       # Development commands
└── .github/workflows/             # CI/CD pipeline
```

### Core Components

#### 🌐 Universal Model Loading

Automatic provider registration with fallback support:

```python
from langgraph_up_devkits import load_chat_model

# Anthropic via OpenRouter (preferred)
model = load_chat_model("openrouter:anthropic/claude-sonnet-4")

# Qwen models (PRC/International regions)
model = load_chat_model("qwen:qwen-flash")

# SiliconFlow models
model = load_chat_model("siliconflow:Qwen/Qwen3-8B")

# With configuration
model = load_chat_model(
    "openrouter:anthropic/claude-sonnet-4",
    temperature=0.7,
    max_tokens=1000
)
```

#### 🤖 Multi-Agent Patterns

```python
from sample_agent.subagents import math, research
from sample_agent.tools import create_handoff_tool

# Create specialized agents
math_agent = math.make_graph()
research_agent = research.make_graph()

# Enable handoffs between agents
math_to_research = create_handoff_tool("research_expert")
research_to_math = create_handoff_tool("math_expert")
```

#### 🔧 Custom Middleware (LangChain v1)

Built-in middleware for dynamic model switching and behavior modification:

```python
from langchain.agents import create_agent
from langgraph_up_devkits import ModelProviderMiddleware, load_chat_model

# Model provider middleware for automatic switching
middleware = ModelProviderMiddleware()

agent = create_agent(
    model=load_chat_model("openrouter:gpt-4o"),  # Fallback model
    tools=[web_search],
    middleware=[middleware]  # Enables context-based model switching
)

# Context specifies different model - middleware switches automatically
context = {"model": "siliconflow:Qwen/Qwen3-8B"}
result = await agent.ainvoke(messages, context=context)
```

For detailed documentation on additional features like middleware, tools, and utilities, see:

- **Framework Documentation**: [`libs/langgraph-up-devkits/README.md`](libs/langgraph-up-devkits/README.md)
- **Agent Examples**: [`apps/sample-agent/README.md`](apps/sample-agent/README.md)

## 🛠 Development

### Commands

See the [Makefile](Makefile) for complete command reference.

```bash
# Testing
make test                    # Run all tests
make test_libs              # Test libraries only
make test_apps              # Test applications only
make unit sample-agent      # Test specific app

# Code Quality
make lint                   # Run linters (ruff + mypy)
make format                 # Format code

# Development
make dev sample-agent                    # Start dev server with browser
make dev sample-agent -- --no-browser   # Start without browser
make dev sample-agent -- --host 0.0.0.0 --port 3000  # Custom host/port
```

### Project Structure Guidelines

- **`libs/`** - Reusable packages shared across agents
- **`apps/`** - Individual agent implementations
- **Shared dependencies** - Managed in root `pyproject.toml`
- **Agent-specific deps** - In app-level `pyproject.toml`

### Creating New Agents

```bash
# Copy sample agent structure
cp -r apps/sample-agent apps/my-agent

# Update configuration
# Edit apps/my-agent/langgraph.json
# Edit apps/my-agent/pyproject.toml
# Implement apps/my-agent/src/my_agent/graph.py
```

## 🧪 Testing

```bash
# Run all tests (126+ tests in libs, 10+ in apps)
make test

# Run tests for specific components
make test_libs              # Test libraries only
make test_apps              # Test applications only
make unit sample-agent      # Test specific app
```

## 🔧 Troubleshooting

Common issues and detailed troubleshooting guides are available in:

- **Setup Issues**: [`libs/langgraph-up-devkits/README.md#troubleshooting`](libs/langgraph-up-devkits/README.md#troubleshooting)
- **Agent Issues**: [`apps/sample-agent/README.md#troubleshooting`](apps/sample-agent/README.md#troubleshooting)

## 🤝 Contributing

### Development Setup

```bash
git clone https://github.com/your-org/langgraph-up-monorepo.git
cd langgraph-up-monorepo
uv sync
make lint  # Ensure code quality
make test  # Run test suite
```

### Code Standards

- **Type Safety** - Strict mypy checking enabled
- **Code Style** - Ruff formatting and linting
- **Testing** - High test coverage required
- **Documentation** - Comprehensive docstrings

### Submission Process

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Ensure tests pass (`make test`)
4. Ensure linting passes (`make lint`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### Core Frameworks
- **[LangGraph](https://github.com/langchain-ai/langgraph)** - Agent orchestration framework
- **[LangChain](https://github.com/langchain-ai/langchain)** - Foundation for agent components

### Development Tools
- **[UV](https://github.com/astral-sh/uv)** - Fast Python package management
- **[langchain-dev-utils](https://github.com/TBice123123/langchain-dev-utils)** - Development utilities for LangChain

### Model Providers
- **[OpenRouter](https://openrouter.ai/)** - Multi-provider model access

---

**Built with ❤️ for the LangGraph community**

Ready to build production-grade agents? [Get started →](#-quick-start)