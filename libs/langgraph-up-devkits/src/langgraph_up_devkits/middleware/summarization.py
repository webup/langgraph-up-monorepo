"""Summarization middleware for token management in long conversations."""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware as LCSummarizationMiddleware
from langchain.chat_models import BaseChatModel

from ..utils.providers import load_chat_model


class SummarizationMiddleware(LCSummarizationMiddleware):
    """Middleware that summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.

    This is a shallow wrapper around LangChain's SummarizationMiddleware that uses
    our custom load_chat_model utility for automatic provider registration.
    """

    def __init__(
        self,
        model: str | BaseChatModel,
        **kwargs: Any,
    ) -> None:
        """Initialize the summarization middleware.

        Args:
            model: The language model to use for generating summaries. Can be a string
                (e.g., "qwen:qwen-flash", "siliconflow:qwen-plus") or an already
                initialized BaseChatModel.
            **kwargs: Additional arguments passed to LangChain's SummarizationMiddleware.
                Common options include:
                - max_tokens_before_summary: Token threshold to trigger summarization
                - messages_to_keep: Number of recent messages to preserve
                - token_counter: Function to count tokens in messages
                - summary_prompt: Prompt template for generating summaries
                - summary_prefix: Prefix added to system message when including summary
        """
        # Initialize the model if it's a string
        if isinstance(model, str):
            model = load_chat_model(model)

        # Call parent constructor with initialized model and all other kwargs
        super().__init__(model=model, **kwargs)
