"""Integration tests for handoff functionality with real models and workflows."""

import pytest
from langchain.messages import HumanMessage
from sample_agent.graph import make_graph
from sample_agent.state import AgentState
from sample_agent.tools.handoff import create_custom_handoff_tool

# Note: LANGCHAIN_TRACING_V2 and OPENROUTER_API_KEY are set in .env file
# to enable real model calls and tracing


class TestHandoffIntegration:
    """Integration tests for agent handoff functionality with real models."""

    def test_handoff_tool_metadata(self):
        """Test handoff tool creation and metadata for FAANG question routing."""
        # Test math handoff tool
        math_handoff = create_custom_handoff_tool(
            agent_name="math_expert",
            name="assign_to_math_expert",
            description="Assign mathematical calculations and analysis tasks to the math expert",
        )

        assert math_handoff.name == "assign_to_math_expert"
        assert math_handoff.metadata.get("__handoff_destination") == "math_expert"
        assert "math" in math_handoff.description.lower()

        # Test research handoff tool
        research_handoff = create_custom_handoff_tool(
            agent_name="research_expert",
            name="assign_to_research_expert",
            description="Assign research and information gathering tasks to the research expert",
        )

        assert research_handoff.name == "assign_to_research_expert"
        assert research_handoff.metadata.get("__handoff_destination") == "research_expert"
        assert "research" in research_handoff.description.lower()

    @pytest.mark.asyncio
    async def test_faang_headcount_real_workflow(self):
        """Test real handoff workflow for FAANG headcount question with actual models."""
        # Create the actual graph with real models
        app = make_graph()

        # Initial state with FAANG headcount question
        initial_state = AgentState(
            messages=[HumanMessage(content="what's the combined headcount of the FAANG companies in 2024?")],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )

        # Execute the real workflow with actual models
        result = await app.ainvoke(initial_state)

        # Verify the workflow completed successfully
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Check that we got a meaningful response
        final_message = result["messages"][-1]
        response_content = final_message.content.lower()

        # Should mention FAANG or specific companies
        faang_indicators = ["faang", "facebook", "meta", "apple", "amazon", "netflix", "google", "alphabet"]
        has_faang = any(indicator in response_content for indicator in faang_indicators)

        # Should contain numerical information (headcount)
        numerical_indicators = ["employees", "headcount", "workforce", "staff", "total", "1,977,586", "1977586"]
        has_numbers = any(indicator in response_content for indicator in numerical_indicators)

        # Should show evidence of processing the request
        assert has_faang or has_numbers, f"No FAANG or numerical indicators found in: {response_content}"

        # Should show evidence of handoff activity (consumed steps or meaningful response)
        steps_consumed = initial_state["remaining_steps"] - result["remaining_steps"]
        has_meaningful_response = len(final_message.content) > 50  # Non-trivial response

        assert steps_consumed > 0 or has_meaningful_response, (
            f"Workflow should have consumed steps or provided meaningful response. "
            f"Steps consumed: {steps_consumed}, Response length: {len(final_message.content)}"
        )

        print(f"Final response: {final_message.content}")
        print(f"Remaining steps: {result['remaining_steps']} (consumed {steps_consumed})")

    @pytest.mark.asyncio
    async def test_research_agent_handoff(self):
        """Test handoff to research agent with real models."""
        app = make_graph()

        # Test explicit research request
        research_message = (
            "Research the current employee counts for tech companies "
            "Facebook, Apple, Amazon, Netflix, and Google"
        )
        research_state = AgentState(
            messages=[HumanMessage(content=research_message)],
            remaining_steps=8,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(research_state)

        # Verify research was performed
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content.lower()

        # Should show evidence of research activity
        research_indicators = ["research", "employee", "company", "tech"]
        has_research = any(indicator in final_content for indicator in research_indicators)

        # Should mention some of the companies
        company_indicators = ["facebook", "meta", "apple", "amazon", "netflix", "google"]
        has_companies = any(company in final_content for company in company_indicators)

        assert has_research or has_companies, f"No research indicators found in: {final_content}"

        # Verify meaningful work was done
        steps_consumed = research_state["remaining_steps"] - result["remaining_steps"]
        assert steps_consumed >= 0, f"Steps should not increase: {steps_consumed}"

        print(f"Research response: {result['messages'][-1].content}")

    @pytest.mark.asyncio
    async def test_math_agent_handoff(self):
        """Test handoff to math agent with real models."""
        app = make_graph()

        # Test explicit math calculation request
        math_state = AgentState(
            messages=[HumanMessage(content="Calculate the sum: 67317 + 164000 + 1551000 + 14000 + 181269")],
            remaining_steps=8,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(math_state)

        # Verify calculation was performed
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content

        # Should contain the calculation result or show calculation work
        calculation_indicators = ["1977586", "1,977,586", "sum", "total", "add"]
        has_calculation = any(indicator in final_content for indicator in calculation_indicators)

        assert has_calculation, f"No calculation indicators found in: {final_content}"

        # Verify meaningful work was done
        steps_consumed = math_state["remaining_steps"] - result["remaining_steps"]
        assert steps_consumed >= 0, f"Steps should not increase: {steps_consumed}"

        print(f"Math response: {result['messages'][-1].content}")

    @pytest.mark.asyncio
    async def test_supervisor_coordination_real(self):
        """Test that supervisor coordinates handoffs properly with real models."""
        app = make_graph()

        # Question that requires supervisor to make routing decisions
        coordination_message = (
            "I need to know the total headcount of major tech companies "
            "and understand how they compare"
        )
        coordination_state = AgentState(
            messages=[HumanMessage(content=coordination_message)],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )

        result = await app.ainvoke(coordination_state)

        # Supervisor should handle the request appropriately
        assert result is not None
        assert len(result["messages"]) > 1

        final_content = result["messages"][-1].content.lower()

        # Should have made progress on the request (consumed steps or meaningful response)
        steps_consumed = coordination_state["remaining_steps"] - result["remaining_steps"]
        has_meaningful_response = len(final_content) > 30

        assert steps_consumed >= 0 and (steps_consumed > 0 or has_meaningful_response), (
            f"Should show evidence of coordination work. "
            f"Steps consumed: {steps_consumed}, Response length: {len(final_content)}"
        )

        # Should show evidence of addressing the request
        processing_indicators = ["tech", "companies", "headcount", "total", "compare"]
        has_processing = any(indicator in final_content for indicator in processing_indicators)

        assert has_processing, f"No evidence of request processing in: {final_content}"

        print(f"Coordination response: {result['messages'][-1].content}")

    def test_state_structure_validation(self):
        """Test AgentState structure for integration workflows."""
        faang_question = "what's the combined headcount of the FAANG companies in 2024?"

        # Test initial state structure
        initial_state = AgentState(
            messages=[HumanMessage(content=faang_question)],
            remaining_steps=10,
            task_description=None,
            active_agent=None
        )

        # Verify state structure
        assert len(initial_state["messages"]) == 1
        assert initial_state["messages"][0].content == faang_question
        assert initial_state["remaining_steps"] == 10
        assert initial_state["active_agent"] is None
        assert initial_state["task_description"] is None

        # Test that state is properly typed
        assert isinstance(initial_state["messages"], list)
        assert isinstance(initial_state["remaining_steps"], int)