"""Base middleware with structured logging support."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import structlog
from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

StateT = TypeVar("StateT", bound=AgentState[Any])
ContextT = TypeVar("ContextT")


class BaseMiddleware(AgentMiddleware[StateT, ContextT], Generic[StateT, ContextT]):  # type: ignore[type-var]
    """Base middleware class with debug logging capabilities.

    This base class provides:
    - Structured logging using structlog
    - Debug mode toggle from runtime context or constructor
    - Private _log method for consistent logging across middleware

    The debug flag can be set in two ways (in order of precedence):
    1. Via runtime.context.debug (dynamic, per-request)
    2. Via constructor debug parameter (static, per-middleware instance)

    Example:
        ```python
        class MyMiddleware(BaseMiddleware[MyState, MyContext]):
            def __init__(self, debug: bool = False) -> None:
                super().__init__(debug=debug)

            async def abefore_model(self, state: MyState, runtime: Runtime[MyContext]) -> dict[str, Any] | None:
                self._log("Processing before model", runtime, state_keys=list(state.keys()))
                # ... middleware logic ...
                return None
        ```
    """

    def __init__(self, debug: bool = False) -> None:
        """Initialize the base middleware.

        Args:
            debug: Default debug logging flag. Can be overridden by runtime.context.debug.
        """
        super().__init__()
        self.debug = debug
        self._logger = structlog.get_logger(self.__class__.__name__)

    def _should_log(self, runtime: Runtime[ContextT] | None = None) -> bool:  # type: ignore[type-var]
        """Determine if logging should be enabled.

        Args:
            runtime: Optional runtime context to check for debug flag.

        Returns:
            True if logging should be enabled, False otherwise.
        """
        # Priority 1: Check runtime context for debug flag
        if runtime and hasattr(runtime, "context") and runtime.context:
            context_debug = getattr(runtime.context, "debug", None)
            if context_debug is not None:
                return bool(context_debug)

        # Priority 2: Fall back to instance debug flag
        return self.debug

    def _log(self, message: str, runtime: Runtime[ContextT] | None = None, **kwargs: Any) -> None:  # type: ignore[type-var]
        """Log debug messages if debug mode is enabled.

        This method uses structlog for structured logging with context.
        Checks runtime.context.debug first, then falls back to self.debug.

        Args:
            message: The log message to output.
            runtime: Optional runtime context to check for debug flag.
            **kwargs: Additional structured context to include in the log entry.
                     These will be output as key-value pairs.

        Example:
            ```python
            self._log("Processing state", runtime, state_size=len(state), step="validation")
            # Output: Processing state state_size=5 step=validation
            ```
        """
        if self._should_log(runtime):
            self._logger.debug(message, **kwargs)


__all__ = ["BaseMiddleware"]
