"""Shared utility functions."""

from datetime import datetime


def get_dummy_message() -> str:
    """Get a dummy message for testing."""
    return "Hello from shared library!"


def get_shared_timestamp() -> str:
    """Get a timestamp message for logging/debugging."""
    return f"Shared lib called at: {datetime.now().isoformat()}"