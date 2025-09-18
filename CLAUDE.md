# LangGraph Development Principles

If you are coding with LangGraph, follow these principles and patterns.

## Monorepo Structure and Shared Dependencies

This project follows a monorepo structure with shared packages and dependencies for LangGraph agents:

### Project Structure
```
celhive-agents/
├── libs/
│   ├── shared/                 # Real package with pyproject.toml
│   │   ├── pyproject.toml
│   │   └── src/shared/
│   │       ├── __init__.py
│   │       └── utils.py        # get_dummy_message(), get_shared_timestamp()
│   ├── common/                 # Faux package (no pyproject.toml)
│   │   ├── __init__.py
│   │   └── helpers.py          # get_common_prefix()
│   └── langgraph-up-devkits/   # Development utilities package
│       ├── pyproject.toml
│       └── src/langgraph_up_devkits/
├── apps/
│   └── sample-agent/           # Example multi-agent supervisor pattern
│       ├── langgraph.json      # deps: ["../../libs/shared", "../../libs/common", "."]
│       ├── pyproject.toml
│       ├── README.md
│       ├── .env.example
│       ├── src/sample_agent/
│       │   ├── __init__.py
│       │   ├── state.py        # Agent state definitions
│       │   ├── graph.py        # Main supervisor graph with make_graph()
│       │   ├── prompts.py      # Agent prompts
│       │   ├── context.py      # Context utilities
│       │   ├── subagents/      # Sub-agent implementations
│       │   │   ├── __init__.py
│       │   │   ├── math.py     # Math expert with make_graph()
│       │   │   └── research.py # Research expert with make_graph()
│       │   └── tools/          # Tool definitions
│       │       ├── __init__.py
│       │       ├── basic.py    # Basic tools (add, multiply, web_search)
│       │       └── handoff.py  # Custom handoff tools
│       └── tests/              # Comprehensive test suite
│           ├── conftest.py
│           ├── unit/
│           └── integration/
├── pyproject.toml              # Root project with shared dependencies
├── Makefile                    # Centralized testing and development commands
└── README.md
```

### Shared Package Management
- **`libs/shared/`**: Real Python package with `pyproject.toml`, containing utility functions
- **`libs/common/`**: Faux package (directory with Python files, no `pyproject.toml`)
- **`libs/langgraph-up-devkits/`**: Development utilities including multi-provider `load_chat_model`, tools, and middleware
- **Root `pyproject.toml`**: Contains shared dependencies for all agents
- **Agent-specific `pyproject.toml`**: Contains agent-specific dependencies

### Agent Dependencies Configuration
Each agent's `langgraph.json` should include:
```json
{
  "dependencies": ["../../libs/shared", "../../libs/common", "."]
}
```

**Note**: The `langgraph-up-devkits` package is automatically available as a workspace dependency and doesn't need explicit inclusion in `langgraph.json`.

### Build and Deployment
```bash
# Sample Agent (supervisor pattern with math & research experts)
langgraph build -t python-monorepo-sample-agent -c apps/sample-agent/langgraph.json
langgraph up -c apps/sample-agent/langgraph.json --image IMAGE_ID_GENERATED_ABOVE

# Development server (during development)
make dev sample-agent                    # Start with browser
make dev sample-agent -- --no-browser   # Start without browser
```

## Critical Structure Requirements

### MANDATORY FIRST STEP
Before creating any files, **always search the codebase** for existing LangGraph-related files:
- Files with names like: `graph.py`, `main.py`, `app.py`, `agent.py`, `workflow.py`
- Files containing: `.compile()`, `StateGraph`, `create_react_agent`, `app =`, graph exports
- Any existing LangGraph imports or patterns

**For this monorepo structure**: Follow the existing pattern with agents in `apps/` directory, each with:
- `src/{agent_name}/graph.py` for the main LangGraph implementation (use meaningful names like `sample_agent`)
- `langgraph.json` with proper dependencies configuration including shared libs
- Agent-specific `pyproject.toml`
- Comprehensive test suite with unit and integration tests
- README.md with usage examples and API documentation

**If any LangGraph files exist**: Follow the existing structure exactly. Do not create new agent.py files.

**Only create agent.py when**: Building from completely empty directory with zero existing LangGraph files.

- When starting from scratch OR adding new agents to this monorepo:
  1. For monorepo: Create new agent in `apps/{meaningful-name}/` following existing `sample-agent` structure
  2. For standalone: `agent.py` at project root with compiled graph exported as `app`
  3. `langgraph.json` configuration file in the same directory as the graph
  4. Proper state management defined with `TypedDict` or Pydantic `BaseModel`
  5. Include shared dependencies: `["../../libs/shared", "../../libs/common", "."]` for monorepo agents
  6. Use `make_graph(config)` pattern for consistent configuration handling
  7. Test small components before building complex graphs

## Deployment-First Principles

**CRITICAL**: All LangGraph agents should be written for DEPLOYMENT unless otherwise specified.

### Core Requirements:
- **NEVER ADD A CHECKPOINTER** unless explicitly requested by user
- Always export compiled graph as `app`
- Use prebuilt components when possible
- Use `load_chat_model` from `langgraph-up-devkits` for consistent model loading
- Follow model preference hierarchy: Anthropic > OpenAI > Google (via OpenRouter)
- Keep state minimal (MessagesState usually sufficient)

#### AVOID unless user specifically requests
```python
# Don't do this unless asked!
from langgraph.checkpoint.memory import MemorySaver
graph = create_react_agent(model, tools, checkpointer=MemorySaver())
```

#### For existing codebases
- Always search for existing graph export patterns first
- Work within the established structure rather than imposing new patterns
- Do not create `agent.py` if graphs are already exported elsewhere

### Standard Structure for New Projects:

#### Standalone Projects:
```
./agent.py          # Main agent file, exports: app
./langgraph.json    # LangGraph configuration
```

#### Monorepo Projects (this codebase):
```
./apps/{meaningful-name}/
├── langgraph.json              # Agent configuration with shared dependencies
├── pyproject.toml              # Agent-specific dependencies
└── src/{agent_name}/
    ├── __init__.py
    ├── state.py                # State definitions
    └── graph.py                # Main LangGraph implementation, exports: app
```

### Export Pattern:
```python
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph_up_devkits import load_chat_model

def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Make graph based on runtime configuration - rebuilds on each call."""
    if config is None:
        config = {}

    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "openrouter:anthropic/claude-sonnet-4")

    # Load model based on configuration
    model = load_chat_model(model_name)

    # Build your graph with configured model
    # ... graph construction logic ...

    return graph.compile()

# Export as 'app' for LangGraph deployment
app = make_graph()  # Required for LangGraph agents
```

### Key Architectural Patterns:

#### 1. `make_graph(config)` Pattern
All agents should implement a `make_graph(config: RunnableConfig | None = None)` function that:
- Accepts optional runtime configuration
- Loads models based on config (with sensible defaults)
- Returns a compiled graph ready for deployment
- Enables dynamic reconfiguration at runtime

#### 2. Subagent Structure
For multi-agent systems, each subagent should have its own `make_graph()` function:
```python
# apps/sample-agent/src/sample_agent/subagents/math.py
def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    # Math agent implementation
    return create_agent(model, tools=[add, multiply], name="math_expert", ...)

# apps/sample-agent/src/sample_agent/subagents/research.py
def make_graph(config: RunnableConfig | None = None) -> CompiledStateGraph[Any, Any, Any, Any]:
    # Research agent implementation
    return create_agent(model, tools=[web_search], name="research_expert", ...)
```

## Prefer Prebuilt Components

**Always use prebuilt components when possible** - they are deployment-ready and well-tested.

### Basic Agents - Use create_agent from LangChain v1:
```python
from langchain.agents import create_agent

# Simple, deployment-ready ReAct agent
graph = create_agent(
    model=model,
    tools=tools,
    prompt="Your agent instructions here"
)
app = graph
```

### Multi-Agent Systems:

#### Supervisor Pattern (central coordination):
```python
from langgraph_supervisor import create_supervisor

supervisor = create_supervisor(
    agents=[agent1, agent2],
    model=model,
    prompt="You coordinate between agents..."
)
app = supervisor.compile()
```
Documentation: https://langchain-ai.github.io/langgraph/reference/supervisor/

#### Swarm Pattern (dynamic handoffs):
```python
from langgraph_swarm import create_swarm, create_handoff_tool

alice = create_react_agent(
    model,
    [tools, create_handoff_tool(agent_name="Bob")],
    prompt="You are Alice.",
    name="Alice",
)

workflow = create_swarm([alice, bob], default_active_agent="Alice")
app = workflow.compile()
```
Documentation: https://langchain-ai.github.io/langgraph/reference/swarm/

### Only Build Custom StateGraph When:
- Prebuilt components don't fit the specific use case
- User explicitly asks for custom workflow
- Complex branching logic required
- Advanced streaming patterns needed

## Model Preferences

**ALWAYS use load_chat_model from langgraph-up-devkits** for model initialization with OpenRouter priority:

```python
from langgraph_up_devkits import load_chat_model

# 1. PREFER: Anthropic via OpenRouter (default in sample-agent)
model = load_chat_model("openrouter:anthropic/claude-sonnet-4")

# 2. SECOND CHOICE: OpenAI via OpenRouter
model = load_chat_model("openrouter:openai/gpt-4o")

# 3. THIRD CHOICE: Google via OpenRouter
model = load_chat_model("openrouter:google/gemini-2.5-pro")

# Additional provider support:
model = load_chat_model("qwen:qwen-flash")                    # Qwen models
model = load_chat_model("qwq:qwq-32b-preview")                # QwQ models
model = load_chat_model("siliconflow:Qwen/Qwen3-8B")          # SiliconFlow models
model = load_chat_model("siliconflow:THUDM/GLM-Z1-9B-0414")   # SiliconFlow GLM models

# Additional configuration options:
model = load_chat_model(
    "openrouter:anthropic/claude-sonnet-4",
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

# The load_chat_model function automatically handles:
# - Provider registration and setup
# - Fallback to init_chat_model when dev-utils unavailable
# - Consistent model naming across the monorepo
# - Seamless support for multiple providers with simple prefixes
```

**Benefits of langgraph-up-devkits load_chat_model:**
- ✅ Automatic provider registration
- ✅ Consistent model loading across all agents
- ✅ Fallback support when langchain-dev-utils unavailable
- ✅ Integrated with monorepo shared dependencies
- ✅ **Multi-provider support**: OpenRouter, Qwen, QwQ, SiliconFlow with simple prefixes
- ✅ **Seamless switching**: Change providers by just updating the model name prefix

**Required API Keys** (set in environment or `.env` file):
- **OpenRouter**: `OPENROUTER_API_KEY` for `openrouter:` models
- **Qwen/QwQ**: `DASHSCOPE_API_KEY` for `qwen:` and `qwq:` models
- **SiliconFlow**: `SILICONFLOW_API_KEY` for `siliconflow:` models

**NOTE**: During development, ignore missing key errors when testing without API access.

## Async-First Development

**ALWAYS prefer async methods** for better performance and concurrent execution:

```python
# PREFER: Async invocation
result = await agent.ainvoke({"messages": state["messages"]})

# PREFER: Async streaming
async for event in agent.astream({"messages": state["messages"]}):
    # Process streaming events
    pass

# PREFER: Async streaming with specific mode
async for event in agent.astream(
    {"messages": state["messages"]},
    stream_mode="updates"
):
    if "__interrupt__" in event:
        # Handle interrupts
        pass
    for node_name, node_data in event.items():
        # Process node outputs
        pass

# Example async agent node
async def my_agent_node(state: State) -> Dict[str, Any]:
    result = await model.ainvoke(state["messages"])
    return {"messages": [result]}
```

**Use sync methods only when**:
- Working with legacy code that doesn't support async
- Simple prototyping or testing
- Integrating with sync-only libraries

## Message and State Handling

### CRITICAL: Extract Message Content Properly
```python
# CORRECT: Extract message content properly (async preferred)
result = await agent.ainvoke({"messages": state["messages"]})
if result.get("messages"):
    final_message = result["messages"][-1]  # This is a message object
    content = final_message.content         # This is the string content

# WRONG: Treating message objects as strings
content = result["messages"][-1]  # This is an object, not a string!
if content.startswith("Error"):   # Will fail - objects don't have startswith()
```

### State Updates Must Be Dictionaries:
```python
# PREFER: Async node functions
async def my_node(state: State) -> Dict[str, Any]:
    # Do work...
    return {
        "field_name": extracted_string,    # Always return dict updates
        "messages": updated_message_list   # Not the raw messages
    }

# Sync version (use only when necessary)
def my_node(state: State) -> Dict[str, Any]:
    # Do work...
    return {
        "field_name": extracted_string,    # Always return dict updates
        "messages": updated_message_list   # Not the raw messages
    }
```

## Streaming and Interrupts

### Streaming Patterns (Async-First):
```python
# PREFER: Async streaming
async for event in agent.astream(
    {"messages": state["messages"]},
    stream_mode="updates"  # Required for interrupts
):
    if "__interrupt__" in event:
        # Handle interrupts
        interrupt_obj = event["__interrupt__"][0]
        interrupt_data = interrupt_obj.value
        # Process interrupt data
        pass

    for node_name, node_data in event.items():
        # Process individual node outputs
        pass
```

**Key Points**:
- Interrupts only work with `stream_mode="updates"`, not `stream_mode="values"`
- In "updates" mode, events are structured as `{node_name: node_data, ...}`
- Check for `"__interrupt__"` key directly in the event object
- Iterate through `event.items()` to access individual node outputs
- Interrupts appear as `event["__interrupt__"]` containing a tuple of `Interrupt` objects
- Access interrupt data via `interrupt_obj.value` where `interrupt_obj = event["__interrupt__"][0]`

Documentation:
- LangGraph Streaming: https://langchain-ai.github.io/langgraph/how-tos/stream-updates/
- SDK Streaming: https://langchain-ai.github.io/langgraph/cloud/reference/sdk/python_sdk_ref/#stream
- Concurrent Interrupts: https://docs.langchain.com/langgraph-platform/interrupt-concurrent

### When to Use Interrupts:
Use `interrupt()` when you need:
- User approval for generated plans or proposed changes
- Human confirmation before executing potentially risky operations
- Additional clarification when the task is ambiguous
- User input data entry or for decision points that require human judgment
- Feedback on partially completed work before proceeding

### Correct Interrupt Usage:
```python
# CORRECT: interrupt() pauses execution for human input
interrupt("Please confirm action")
# Execution resumes after human provides input through platform

# AVOID: Treating interrupt() as synchronous
result = interrupt("Please confirm action")  # Wrong - doesn't return values
if result == "yes":  # This won't work
    proceed()
```

## Common LangGraph Errors to Avoid

- Incorrect `interrupt()` usage: It pauses execution, doesn't return values
- Refer to documentation for best interrupt handling practices, including waiting for user input and proper handling of it
- Wrong state update patterns: Return updates, not full state
- Missing state type annotations
- Missing state fields (current_field, user_input)
- Invalid edge conditions: Ensure all paths have valid transitions
- Not handling error states properly
- Not exporting graph as 'app' when creating new LangGraph agents from scratch
- Forgetting `langgraph.json` configuration
- **Type assumption errors**: Assuming message objects are strings, or that state fields are certain types
- **Chain operations without type checking**: Like `state.get("field", "")[-1].method()` without verifying types

## Framework Integration Patterns

### Integration Debugging
When building integrations, always start with debugging:

```python
# Temporary debugging for new integrations
def my_integration_function(input_data, config):
    print(f"=== DEBUG START ===")
    print(f"Input type: {type(input_data)}")
    print(f"Input data: {input_data}")
    print(f"Config type: {type(config)}")
    print(f"Config data: {config}")
    
    # Process...
    result = process(input_data, config)
    
    print(f"Result type: {type(result)}")
    print(f"Result data: {result}")
    print(f"=== DEBUG END ===")
    
    return result
```

### Config Propagation Verification
Always verify the receiving end actually uses configuration:

```python
# WRONG: Assuming config is used
def my_node(state: State) -> Dict[str, Any]:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# CORRECT: Actually using config
def my_node(state: State, config: RunnableConfig) -> Dict[str, Any]:
    # Extract configuration
    configurable = config.get("configurable", {})
    system_prompt = configurable.get("system_prompt", "Default prompt")
    
    # Use configuration in messages
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}
```
## Patterns to Avoid

### Don't Mix Responsibilities in Single Nodes:
```python
# AVOID: LLM call + tool execution in same node
def bad_node(state):
    ai_response = model.invoke(state["messages"])  # LLM call
    tool_result = tool_node.invoke({"messages": [ai_response]})  # Tool execution
    return {"messages": [...]}  # Mixed concerns!

# PREFER: Separate nodes for separate concerns
def llm_node(state):
    return {"messages": [model.invoke(state["messages"])]}

def tool_node(state):
    return ToolNode(tools).invoke(state)

# Connect with edges
workflow.add_edge("llm", "tools")
```

### Overly Complex Agents When Simple Ones Suffice
```python
# AVOID: Unnecessary complexity
workflow = StateGraph(ComplexState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
# ... 20 lines of manual setup when create_react_agent would work
```

### Avoid Overly Complex State:
```python
# AVOID: Too many state fields
class State(TypedDict):
    messages: List[BaseMessage]
    user_input: str
    current_step: int
    metadata: Dict[str, Any]
    history: List[Dict]
    # ... many more fields

# PREFER: Use MessagesState when sufficient
from langgraph.graph import MessagesState
```

### Wrong Export Patterns
```python
# AVOID: Wrong variable names or missing export
compiled_graph = workflow.compile()  # Wrong name
# Missing: app = compiled_graph
```

### Incorrect interrupt() usage
```python
# AVOID: Treating interrupt() as synchronous
result = interrupt("Please confirm action")  # Wrong - doesn't return values
if result == "yes":  # This won't work
    proceed()

# CORRECT: interrupt() pauses execution for human input
interrupt("Please confirm action")
# Execution resumes after human provides input through platform
```
Reference: https://langchain-ai.github.io/langgraph/concepts/streaming/#whats-possible-with-langgraph-streaming

## LangGraph-Specific Coding Standards

### Structured LLM Calls and Validation
When working with LangGraph nodes that involve LLM calls, always use structured output with Pydantic dataclasses:

- Use `with_structured_output()` method for LLM calls that need specific response formats
- Define Pydantic BaseModel classes for all structured data (state schemas, LLM responses, tool inputs/outputs)
- Validate and parse LLM responses using Pydantic models
- For conditional nodes relying on LLM decisions, use structured output

Example: `llm.with_structured_output(MyPydanticModel).invoke(messages)` instead of raw string parsing

### General Guidelines:
- Test small components before building complex graphs
- **Avoid unnecessary complexity**: Consider if simpler approaches with prebuilt components would achieve the same goals
- Write concise and clear code without overly verbose implementations
- Only install trusted, well-maintained packages

## Code Quality Requirements

### MANDATORY: Always run tests and lint with Makefile

**CRITICAL**: This monorepo uses a centralized root-level Makefile for ALL testing and code quality operations. You MUST ALWAYS use Makefile commands - NEVER run pytest, ruff, or mypy directly.

**ALWAYS use Makefile commands:**
```bash
# Testing (REQUIRED)
make test           # Run all tests across all packages
make test_libs      # Run tests on libs/ packages only
make test_apps      # Run tests on apps/ packages only

# Integration Testing
make test_integration        # Run all integration tests
make test_integration_libs   # Run integration tests on libs/ only
make test_integration_apps   # Run integration tests on apps/ only

# Parameterized Testing (for specific apps)
make unit <app_name>         # Run unit tests for specific app
make integration <app_name>  # Run integration tests for specific app
# Examples:
make unit sample-agent
make integration sample-agent

# Linting (REQUIRED after any code changes)
make lint           # Run linters on all packages (ruff + mypy on src/)
make lint_libs      # Lint libs/ packages only
make lint_apps      # Lint apps/ packages only

# Formatting (RECOMMENDED before linting)
make format         # Run code formatters on all packages (ruff format + import sorting)
make format_libs    # Format libs/ packages only
make format_apps    # Format apps/ packages only

# Development Server
make dev <app_name> [-- args...]  # Start LangGraph dev server with optional args
# Examples:
make dev sample-agent
make dev sample-agent -- --no-browser
make dev sample-agent -- --host 0.0.0.0 --port 3000
```

**NEVER run these directly:**
```bash
# ❌ DON'T DO THIS:
pytest tests/
uv run pytest tests/
ruff check src/
mypy src/
```

**ALWAYS do this instead:**
```bash
# ✅ CORRECT:
make test
make lint
make format
```

**Requirements:**
- ALL testing MUST use `make test` commands
- ALL linting MUST use `make lint` commands
- All lint errors MUST be fixed before considering a task complete
- Use Makefile for consistent environment and dependency management
- Follow ruff and mypy strict typing standards
- Import sorting with ruff is mandatory

**Workflow:**
1. Make code changes
2. Run `make format` (optional but recommended)
3. Run `make test` to ensure tests pass (or `make unit <app_name>` for specific app)
4. Run `make lint` to check code quality
5. Fix any issues and repeat until all checks pass
6. For development testing: `make dev <app_name> -- --no-browser`
7. Only then is the task considered complete

**Development Commands:**
- `make dev <app_name>` - Start LangGraph dev server with browser
- `make dev <app_name> -- --no-browser` - Start without browser
- `make dev <app_name> -- --host 0.0.0.0 --port 3000` - Custom host/port
- `make unit <app_name>` - Run unit tests for specific app
- `make integration <app_name>` - Run integration tests for specific app

## Documentation Guidelines

### When to Consult Documentation:
Always use documentation tools before implementing LangGraph code (the API evolves rapidly):
- Before creating new graph nodes or modifying existing ones
- When implementing state schemas or message passing patterns
- Before using LangGraph-specific decorators, annotations, or utilities
- When working with conditional edges, dynamic routing, or subgraphs
- Before implementing tool calling patterns within graph nodes
 - When building applications that integrate multiple frameworks (e.g., LangGraph + Streamlit, LangGraph + Next.js/React), also consult the framework docs to ensure correct syntax and patterns

### Key Documentation Resources:
- LangGraph Streaming: https://langchain-ai.github.io/langgraph/how-tos/stream-updates/
- LangGraph Config: https://langchain-ai.github.io/langgraph/how-tos/pass-config-to-tools/
- Supervisor Pattern: https://langchain-ai.github.io/langgraph/reference/supervisor/
- Swarm Pattern: https://langchain-ai.github.io/langgraph/reference/swarm/
- Agentic Concepts: https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/

### Documentation Navigation
- Determine the base URL from the current documentation page
- For `../`, go one level up in the URL hierarchy
- For `../../`, go two levels up, then append the relative path
- Example: From `https://langchain-ai.github.io/langgraph/tutorials/get-started/langgraph-platform/setup/` with link `../../langgraph-platform/local-server`
  - Go up two levels: `https://langchain-ai.github.io/langgraph/tutorials/get-started/`
  - Append path: `https://langchain-ai.github.io/langgraph/tutorials/get-started/langgraph-platform/local-server`
- If you encounter an HTTP 404 error, the constructed URL is likely incorrect—rebuild it carefully