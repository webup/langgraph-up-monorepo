"""FileSystem Mask middleware for shadowing files field in state."""

from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime


class FileSystemMaskMiddleware(AgentMiddleware[Any]):
    """Middleware that shadows the 'files' field before model calls and restores it after.

    This middleware removes the 'files' field from state before passing to the model
    (via before_model hook) and restores it after the model has run (via after_model hook).
    This is useful when you want to prevent the model from seeing or being influenced by
    file system data in the state.

    The shadowed files are stored internally and restored automatically, ensuring no
    data loss while keeping the model's context clean.

    Note: This middleware requires another middleware to define the state schema with
    a 'files' field. Use it in combination with a state-defining middleware.

    Example:
        >>> from langchain.agents import create_agent
        >>> from langgraph_up_devkits.middleware import FileSystemMaskMiddleware
        >>>
        >>> # Define a middleware that adds files field to state
        >>> class FilesStateMiddleware(AgentMiddleware):
        ...     class FilesState(AgentState):
        ...         files: dict
        ...     state_schema = FilesState
        >>>
        >>> # Use both middlewares together
        >>> agent = create_agent(
        ...     model=model,
        ...     tools=tools,
        ...     middleware=[FilesStateMiddleware(), FileSystemMaskMiddleware()]
        ... )
        >>>
        >>> # Files will be masked from model but preserved in state
        >>> result = agent.invoke({"messages": [...], "files": {...}})
    """

    _NO_FILES_SENTINEL = object()  # Sentinel to distinguish None from no files

    def __init__(self) -> None:
        """Initialize the FileSystemMask middleware."""
        super().__init__()
        self._shadowed_files: Any = self._NO_FILES_SENTINEL

    def before_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Shadow the 'files' field before model execution.

        Args:
            state: The current agent state, expected to have a "files" field.
            runtime: The runtime context (unused but required by interface).

        Returns:
            State update with "files" field removed, or None if no files to shadow.
        """
        # Check if state has 'files' field
        if isinstance(state, dict) and "files" in state:
            # Store the files for later restoration (even if None)
            self._shadowed_files = state["files"]

            # Return state update without files
            new_state = {k: v for k, v in state.items() if k != "files"}
            return new_state

        return None

    def after_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Restore the 'files' field after model execution.

        Args:
            state: The current agent state after model execution.
            runtime: The runtime context (unused but required by interface).

        Returns:
            State update with "files" field restored, or None if no files to restore.
        """
        # Check if we have shadowed files
        if self._shadowed_files is not self._NO_FILES_SENTINEL:
            # Restore the files (even if None)
            files = self._shadowed_files
            self._shadowed_files = self._NO_FILES_SENTINEL  # Clean up
            return {"files": files}

        return None
