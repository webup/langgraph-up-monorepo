"""Unit tests for basic sample_agent workflow components."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from sample_agent.state import AgentState
from sample_agent.tools.basic import add, multiply, web_search
from sample_agent.tools.handoff import create_custom_handoff_tool


class TestBasicWorkflow:
    """Unit tests for basic workflow components."""

    def test_basic_math_tools(self):
        """Test basic math tools work correctly."""
        assert add(2.5, 3.5) == 6.0
        assert multiply(4.0, 5.0) == 20.0
        assert add(67317, 164000) == 231317
        
    def test_web_search_tool(self):
        """Test web search tool returns expected FAANG data."""
        result = web_search("FAANG headcount")
        
        # Should contain all FAANG companies
        assert "Facebook" in result and "Meta" in result
        assert "Apple" in result
        assert "Amazon" in result 
        assert "Netflix" in result
        assert "Google" in result and "Alphabet" in result
        
        # Should contain expected numbers
        assert "67,317" in result
        assert "164,000" in result
        assert "1,551,000" in result
        assert "14,000" in result
        assert "181,269" in result

    def test_faang_calculation(self):
        """Test manual FAANG calculation using basic tools."""
        # Test the calculation we expect from the workflow
        meta = 67317
        apple = 164000
        amazon = 1551000
        netflix = 14000
        google = 181269
        
        # Step by step calculation
        step1 = add(meta, apple)  # 231317
        step2 = add(step1, amazon)  # 1782317
        step3 = add(step2, netflix)  # 1796317
        total = add(step3, google)  # 1977586
        
        assert total == 1977586

    def test_handoff_tool_creation(self):
        """Test handoff tool creation and basic properties."""
        tool = create_custom_handoff_tool(
            agent_name="test_agent",
            name="handoff_to_test", 
            description="Test handoff tool for test_agent"
        )
        
        assert tool is not None
        assert tool.name == "handoff_to_test"
        assert "test_agent" in tool.description
        assert hasattr(tool, 'metadata')
        assert tool.metadata.get("__handoff_destination") == "test_agent"

    def test_agent_state_structure(self):
        """Test AgentState structure and typing."""
        state = AgentState(
            messages=[HumanMessage(content="test")],
            remaining_steps=10,
            task_description="Test task",
            active_agent="test_agent"
        )
        
        assert len(state["messages"]) == 1
        assert state["remaining_steps"] == 10
        assert state["task_description"] == "Test task"
        assert state["active_agent"] == "test_agent"

    @patch('sample_agent.subagents.math.create_agent')
    @patch('sample_agent.subagents.math.load_chat_model')
    def test_math_agent_creation(self, mock_load_model, mock_create_agent):
        """Test math agent creation with mocked dependencies."""
        from sample_agent.subagents.math import make_graph

        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.compile.return_value = "compiled_agent"
        mock_create_agent.return_value = mock_agent

        config = {"configurable": {"model_name": "test_model"}}
        result = make_graph(config)

        mock_load_model.assert_called_once_with("test_model")
        mock_create_agent.assert_called_once()
        call_args = mock_create_agent.call_args
        assert call_args[1]['model'] == mock_model
        assert call_args[1]['name'] == "math_expert"
        assert len(call_args[1]['tools']) == 2  # add and multiply
        # Result should be compiled agent since agent has compile method
        assert result == "compiled_agent"

    @patch('sample_agent.subagents.research.create_agent')
    @patch('sample_agent.subagents.research.load_chat_model')
    def test_research_agent_creation(self, mock_load_model, mock_create_agent):
        """Test research agent creation with mocked dependencies."""
        from sample_agent.subagents.research import make_graph

        mock_model = Mock()
        mock_load_model.return_value = mock_model
        mock_agent = Mock()
        mock_agent.compile.return_value = "compiled_agent"
        mock_create_agent.return_value = mock_agent

        config = {"configurable": {"model_name": "test_model"}}
        result = make_graph(config)

        mock_load_model.assert_called_once_with("test_model")
        mock_create_agent.assert_called_once()
        call_args = mock_create_agent.call_args
        assert call_args[1]['model'] == mock_model
        assert call_args[1]['name'] == "research_expert"
        assert len(call_args[1]['tools']) == 1  # web_search
        # Result should be compiled agent since agent has compile method
        assert result == "compiled_agent"

    @patch('sample_agent.graph.load_chat_model')
    @patch('sample_agent.graph.make_math_graph')
    @patch('sample_agent.graph.make_research_graph')
    @patch('sample_agent.graph.create_supervisor')
    def test_graph_creation_components(
        self, mock_create_supervisor, mock_research_graph, mock_math_graph, mock_load_model
    ):
        """Test that graph creation calls all expected components."""
        # Setup mocks
        mock_supervisor_model = Mock()
        mock_load_model.return_value = mock_supervisor_model
        mock_math_agent = Mock()
        mock_research_agent = Mock()
        mock_math_graph.return_value = mock_math_agent
        mock_research_graph.return_value = mock_research_agent
        mock_workflow = Mock()
        mock_app = Mock()
        mock_workflow.compile.return_value = mock_app
        mock_create_supervisor.return_value = mock_workflow
        
        # Import and use make_graph function
        from sample_agent.graph import make_graph
        make_graph()
        
        # Verify calls - now config is passed to make_graph functions
        mock_math_graph.assert_called_once_with({})
        mock_research_graph.assert_called_once_with({})
        mock_load_model.assert_called_once_with("siliconflow:zai-org/GLM-4.5-Air")
        mock_create_supervisor.assert_called_once()
        
        # Verify supervisor was called with correct agents
        supervisor_call = mock_create_supervisor.call_args
        agents = supervisor_call[0][0]  # First positional argument
        assert mock_research_agent in agents
        assert mock_math_agent in agents
        
        # Verify supervisor model and tools were passed
        assert supervisor_call[1]['model'] == mock_supervisor_model
        assert 'tools' in supervisor_call[1]
        assert len(supervisor_call[1]['tools']) == 3  # 2 handoff + 1 forwarding
        
        # Verify workflow.compile was called with name
        mock_workflow.compile.assert_called_once_with(name="supervisor")

    def test_workflow_state_transitions(self):
        """Test state transitions in workflow."""
        # Test initial state
        initial_state = AgentState(
            messages=[HumanMessage(content="What's the FAANG total?")],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )
        
        # Simulate state after research handoff
        research_state = AgentState(
            messages=initial_state["messages"] + [AIMessage(content="Research complete")],
            remaining_steps=8,
            task_description="Research FAANG headcounts",
            active_agent="research_expert"
        )
        
        # Simulate state after math handoff
        math_state = AgentState(
            messages=research_state["messages"] + [AIMessage(content="Calculation: 1,977,586")],
            remaining_steps=6,
            task_description="Calculate total headcount",
            active_agent="math_expert"
        )
        
        # Verify state progression
        assert len(math_state["messages"]) == 3
        assert math_state["remaining_steps"] < initial_state["remaining_steps"]
        assert math_state["active_agent"] == "math_expert"

    @pytest.mark.asyncio
    async def test_mock_workflow_execution(self):
        """Test mock workflow execution without real agents."""
        
        class MockWorkflow:
            async def ainvoke(self, state):
                # Simulate processing
                final_state = state.copy()
                final_state["messages"] = state["messages"] + [
                    AIMessage(content="FAANG total headcount is 1,977,586 employees")
                ]
                final_state["remaining_steps"] = state["remaining_steps"] - 5
                final_state["active_agent"] = "supervisor"
                return final_state
        
        mock_app = MockWorkflow()
        
        initial_state = AgentState(
            messages=[HumanMessage(content="what's the combined headcount of the FAANG companies in 2024?")],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )
        
        result = await mock_app.ainvoke(initial_state)
        
        assert len(result["messages"]) == 2
        assert "1,977,586" in result["messages"][-1].content
        assert result["remaining_steps"] == 5
        assert result["active_agent"] == "supervisor"
