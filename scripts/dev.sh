#!/bin/bash

# LangGraph dev server launcher with argument support
# Usage: ./scripts/dev.sh <app_name> [args...]

set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./scripts/dev.sh <app_name> [args...]"
    echo "Example: ./scripts/dev.sh agent1"
    echo "Example: ./scripts/dev.sh agent1 --port 8080"
    echo "Example: ./scripts/dev.sh agent1 --host 0.0.0.0 --port 3000 --no-browser"
    exit 1
fi

app_name="$1"
shift  # Remove app_name from arguments, leaving the rest as langgraph dev args

if [ ! -d "apps/$app_name" ]; then
    echo "❌ App 'apps/$app_name' not found"
    exit 1
fi

echo "Stopping any existing LangGraph dev servers..."
pkill -f "langgraph dev" || true
sleep 1

if [ $# -gt 0 ]; then
    echo "Starting LangGraph dev server for apps/$app_name with args: $*"
    cd "apps/$app_name" && uv run langgraph dev "$@"
else
    echo "Starting LangGraph dev server for apps/$app_name..."
    cd "apps/$app_name" && uv run langgraph dev
fi