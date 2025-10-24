"""Unit tests for human-in-the-loop (HITL) functionality in deep agent."""

from unittest.mock import Mock, patch

from sample_deep_agent.context import DeepAgentContext


class TestDeepAgentHITLConfiguration:
    """Unit tests for deep agent HITL configuration."""

    def test_default_context_no_interrupt_fields(self):
        """Test that context no longer has interrupt fields (moved to make_graph)."""
        context = DeepAgentContext()

        # Interrupt fields should not exist in context anymore
        assert not hasattr(context, "enable_interrupts")
        assert not hasattr(context, "interrupt_on")
        assert not hasattr(context, "subagent_interrupts")

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_checkpointer_added_when_interrupt_on_provided(
        self, mock_load_model, mock_create_deep_agent
    ):
        """Test that checkpointer is automatically added when interrupt_on is provided."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        # Create graph with interrupt_on
        interrupt_on = {"deep_web_search": True}
        make_graph(interrupt_on=interrupt_on)

        # Verify create_deep_agent was called with checkpointer
        call_args = mock_create_deep_agent.call_args
        assert call_args[1]["checkpointer"] is not None

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_no_checkpointer_when_no_interrupt_on(
        self, mock_load_model, mock_create_deep_agent
    ):
        """Test that checkpointer is not added when interrupt_on is None."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        # Create graph without interrupt_on (default)
        make_graph()

        # Verify create_deep_agent was called without checkpointer
        call_args = mock_create_deep_agent.call_args
        assert call_args[1]["checkpointer"] is None

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_interrupt_on_passed_to_create_deep_agent(
        self, mock_load_model, mock_create_deep_agent
    ):
        """Test that interrupt_on configuration is passed to create_deep_agent."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        # Create graph with interrupt_on config
        interrupt_config = {"deep_web_search": True}
        make_graph(interrupt_on=interrupt_config)

        # Verify create_deep_agent was called with interrupt_on
        call_args = mock_create_deep_agent.call_args
        assert call_args[1]["interrupt_on"] == interrupt_config

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_subagent_interrupts_applied_to_subagents(
        self, mock_load_model, mock_create_deep_agent
    ):
        """Test that subagent interrupt overrides are applied correctly."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        # Create graph with subagent interrupt overrides
        subagent_config = {
            "research-agent": {
                "deep_web_search": True,
            }
        }
        make_graph(subagent_interrupts=subagent_config)

        # Verify create_deep_agent was called with modified subagents
        call_args = mock_create_deep_agent.call_args
        subagents = call_args[1]["subagents"]

        # Find the research-agent subagent
        research_agent = next(
            (s for s in subagents if s["name"] == "research-agent"),
            None,
        )
        assert research_agent is not None
        assert research_agent["interrupt_on"] == {"deep_web_search": True}

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_interrupt_config_with_allowed_decisions(
        self, mock_load_model, mock_create_deep_agent
    ):
        """Test interrupt configuration with custom allowed_decisions."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        interrupt_config = {
            "deep_web_search": {
                "allowed_decisions": ["approve", "reject"]  # No editing allowed
            }
        }

        make_graph(interrupt_on=interrupt_config)

        # Verify create_deep_agent was called with interrupt_on
        call_args = mock_create_deep_agent.call_args
        assert call_args[1]["interrupt_on"]["deep_web_search"]["allowed_decisions"] == [
            "approve",
            "reject",
        ]

    @patch("sample_deep_agent.graph.create_deep_agent")
    @patch("sample_deep_agent.graph.load_chat_model")
    def test_mixed_interrupt_configurations(self, mock_load_model, mock_create_deep_agent):
        """Test mixed interrupt configurations (bool and dict)."""
        from sample_deep_agent.graph import make_graph

        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.with_config.return_value = mock_agent
        mock_create_deep_agent.return_value = mock_agent

        interrupt_config = {
            "deep_web_search": True,  # Default behavior
            "think_tool": {"allowed_decisions": ["approve"]},  # Must approve
        }

        make_graph(interrupt_on=interrupt_config)

        # Verify create_deep_agent was called with interrupt_on
        call_args = mock_create_deep_agent.call_args
        assert call_args[1]["interrupt_on"]["deep_web_search"] is True
        assert call_args[1]["interrupt_on"]["think_tool"]["allowed_decisions"] == ["approve"]
