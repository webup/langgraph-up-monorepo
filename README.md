# 🚀 LangGraph-UP Monorepo

**LangGraph-UP Monorepo** showcases how to build production-ready LangGraph agents using the latest **LangChain & LangGraph** V1 ecosystem, organized in a clean monorepo structure with shared libraries and multiple agent applications.

[![Version](https://img.shields.io/badge/version-v0.2.0-blue.svg)](https://github.com/webup/langgraph-up-monorepo/releases/tag/v0.2.0)
[![LangChain](https://img.shields.io/badge/LangChain-v1alpha-blue.svg)](https://github.com/langchain-ai/langchain)
[![LangGraph](https://img.shields.io/badge/LangGraph-v1alpha-blue.svg)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/badge/PyPI-langgraph--up--devkits-blue.svg)](https://test.pypi.org/project/langgraph-up-devkits/)
[![Twitter](https://img.shields.io/twitter/follow/zhanghaili0610?style=social)](https://twitter.com/zhanghaili0610)

## ✨ Key Features

- 🌐 **Universal Model Loading** - OpenRouter, Qwen, QwQ, SiliconFlow with automatic registration
- 🤖 **Multi-Agent Orchestration** - Supervisor & deep research patterns with specialized sub-agents
- 🛠 **Custom Middleware** - Model switching, file masking, summarization, and state management
- 🧪 **Developer Experience** - Hot reload, comprehensive testing, strict linting, PyPI publishing
- 🚀 **Deployment Ready** - LangGraph Cloud configurations included
- 🌍 **Global Ready** - Region-based provider configuration (PRC/International)

## 🚀 Quick Start

### Installation

```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/webup/langgraph-up-monorepo.git
cd langgraph-up-monorepo
uv sync --dev
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

### Sample Agents

This monorepo includes two complete agent examples demonstrating different patterns:

#### 🤖 sample-agent: Supervisor Pattern
Multi-agent system with a coordinator that delegates to specialized sub-agents.

```bash
make dev sample-agent
```

**Features:**
- Supervisor-based coordination
- Math expert (add, multiply operations)
- Research expert (web search capabilities)
- Cross-agent handoffs

#### 🔬 sample-deep-agent: Deep Research Pattern
Advanced research workflow with virtual file system and structured planning.

```bash
make dev sample-deep-agent
```

**Features:**
- Deep web search with content extraction
- Virtual file system for document management
- Think tool for strategic TODO planning
- Research & critique sub-agents
- FileSystemMaskMiddleware to optimize token usage

## 🏗 Architecture

### Monorepo Structure

```
langgraph-up-monorepo/
├── libs/
│   ├── shared/                    # Shared utilities
│   ├── common/                    # Common helper functions
│   └── langgraph-up-devkits/      # 🎯 Core framework (published to PyPI)
│       ├── utils/providers.py     #   → Multi-provider model loading
│       ├── middleware/            #   → Custom middleware (model, file, summary)
│       ├── tools/                 #   → Web search, deep search, MCP integration
│       └── context/               #   → Context schemas & aware prompts
├── apps/
│   ├── sample-agent/              # 🤖 Supervisor pattern (math + research agents)
│   │   ├── src/sample_agent/
│   │   │   ├── graph.py           #   → Main supervisor graph
│   │   │   ├── subagents/         #   → Math & research experts
│   │   │   └── tools/             #   → Agent-specific tools
│   │   └── langgraph.json         #   → Deployment config
│   └── sample-deep-agent/         # 🔬 Deep research pattern (VFS + think tool)
│       ├── src/sample_deep_agent/
│       │   ├── graph.py           #   → Deep agent with research workflow
│       │   ├── subagents.py       #   → Research & critique experts
│       │   └── prompts.py         #   → Structured TODO planning prompts
│       └── langgraph.json         #   → Deployment config
├── pyproject.toml                 # Root dependencies
├── Makefile                       # Development commands
├── PUBLISHING.md                  # PyPI publishing guide
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

Built-in middleware for dynamic model switching, state management, and behavior modification:

```python
from langchain.agents import create_agent
from langgraph_up_devkits import (
    ModelProviderMiddleware,
    FileSystemMaskMiddleware,
    load_chat_model
)

# Model provider middleware for automatic switching
model_middleware = ModelProviderMiddleware()

# File system middleware to mask large file content from LLM context
fs_middleware = FileSystemMaskMiddleware()

agent = create_agent(
    model=load_chat_model("openrouter:gpt-4o"),  # Fallback model
    tools=[web_search, deep_web_search],
    middleware=[model_middleware, fs_middleware]
)

# Context specifies different model - middleware switches automatically
context = {"model": "siliconflow:Qwen/Qwen3-8B"}
result = await agent.ainvoke(messages, context=context)
```

**Available Middleware:**
- `ModelProviderMiddleware` - Dynamic model switching based on context
- `FileSystemMaskMiddleware` - Masks virtual file systems from LLM to save tokens
- `SummarizationMiddleware` - Automatic message summarization for long conversations

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

# Publishing (langgraph-up-devkits)
make build_devkits                       # Build distribution packages
make check_devkits                       # Validate package
make release_test_devkits               # Build and publish to Test PyPI
make release_devkits                     # Build and publish to PyPI
```

See [PUBLISHING.md](PUBLISHING.md) for detailed publishing guide.

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