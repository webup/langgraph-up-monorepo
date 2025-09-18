# LangGraph Monorepo Example

A Python monorepo example demonstrating LangGraph agents with shared packages.

## Structure

```
python-monorepo-example/
├── libs/
│   ├── shared/                 # Real package with pyproject.toml
│   │   ├── pyproject.toml
│   │   └── src/shared/
│   │       ├── __init__.py
│   │       └── utils.py        # get_dummy_message(), get_shared_timestamp()
│   └── common/                 # Faux package (no pyproject.toml)
│       ├── __init__.py
│       └── helpers.py          # get_common_prefix()
├── apps/
│   ├── agent1/
│   │   ├── langgraph.json      # deps: ["../../libs/shared", "../../libs/common"]
│   │   ├── pyproject.toml
│   │   └── src/agent1/
│   │       ├── __init__.py
│   │       ├── state.py
│   │       └── graph.py        # Simple single-node LangGraph
│   └── agent2/
│       ├── langgraph.json      # deps: ["../../libs/shared", "../../libs/common"]
│       ├── pyproject.toml
│       └── src/agent2/
│           ├── __init__.py
│           ├── state.py
│           └── graph.py        # Simple single-node LangGraph
├── pyproject.toml              # Root project with shared dependencies
├── .gitignore
└── README.md
```

## Key Features

### Shared Packages
- **`libs/shared/`**: Real Python package with `pyproject.toml`, containing utility functions
- **`libs/common/`**: Faux package (directory with Python files, no `pyproject.toml`)

### Shared dependencies
- There is a root `pyproject.toml` that contains shared dependencies for all your agents

### Agent Configuration
- Each agent has `langgraph.json` with `"dependencies": ["../../libs/shared", "../../libs/common", "."]`
- Each agent has its own `pyproject.toml` for agent-specific dependencies

## Usage

Each agent can be run independently using the LangGraph CLI, and both will have access to the shared utility functions via the dependency configuration. For monorepo support, you can run your build command from the root, getting the benefits of the shared dependencies.

### Agent 1

```bash
langgraph build -t python-monorepo-agent-1 -c apps/agent1/langgraph.json
langgraph up -c apps/agent1/langgraph.json --image IMAGE_ID_GENERATED_ABOVE
```

### Agent 2

```bash
langgraph build -t python-monorepo-agent-2 -c apps/agent2/langgraph.json
langgraph up -c apps/agent2/langgraph.json --image IMAGE_ID_GENERATED_ABOVE
```
