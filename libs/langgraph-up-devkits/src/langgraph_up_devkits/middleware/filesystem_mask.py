"""FileSystem Mask middleware for shadowing files field in state."""

from typing import Any

from langgraph.runtime import Runtime

from .base import BaseMiddleware


class FileSystemMaskMiddleware(BaseMiddleware[Any, Any]):
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

    def __init__(self, debug: bool = False) -> None:
        """Initialize the FileSystemMask middleware.

        Args:
            debug: Enable debug logging for file masking operations.
        """
        super().__init__(debug=debug)
        self._shadowed_files: Any = self._NO_FILES_SENTINEL

    def before_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Shadow the 'files' field before model execution.

        Args:
            state: The current agent state, expected to have a "files" field.
            runtime: The runtime context.

        Returns:
            State update with "files" field removed, or None if no files to shadow.
        """
        # Check if state has 'files' field
        if isinstance(state, dict) and "files" in state:
            # Store the files for later restoration (even if None)
            self._shadowed_files = state["files"]
            self._log("Shadowing files field", runtime, files_count=len(state["files"]) if state["files"] else 0)

            # Return state update without files
            new_state = {k: v for k, v in state.items() if k != "files"}
            return new_state

        self._log("No files field to shadow", runtime)
        return None

    async def abefore_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Shadow the 'files' field before model execution (async version).

        Args:
            state: The current agent state, expected to have a "files" field.
            runtime: The runtime context.

        Returns:
            State update with "files" field removed, or None if no files to shadow.
        """
        # Async version delegates to sync implementation
        return self.before_model(state, runtime)

    def after_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Restore the 'files' field after model execution.

        Args:
            state: The current agent state after model execution.
            runtime: The runtime context.

        Returns:
            State update with "files" field restored, or None if no files to restore.
        """
        # Check if we have shadowed files
        if self._shadowed_files is not self._NO_FILES_SENTINEL:
            # Restore the files (even if None)
            files = self._shadowed_files
            self._shadowed_files = self._NO_FILES_SENTINEL  # Clean up
            self._log("Restoring files field", runtime, files_count=len(files) if files else 0)
            return {"files": files}

        self._log("No files to restore", runtime)
        return None

    async def aafter_model(self, state: Any, runtime: Runtime[None]) -> dict[str, Any] | None:
        """Restore the 'files' field after model execution (async version).

        Args:
            state: The current agent state after model execution.
            runtime: The runtime context.

        Returns:
            State update with "files" field restored, or None if no files to restore.
        """
        # Async version delegates to sync implementation
        return self.after_model(state, runtime)
