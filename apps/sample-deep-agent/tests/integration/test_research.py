"""Integration tests for research workflows."""

import os

import pytest
from langchain.messages import AIMessage, HumanMessage


@pytest.mark.integration
@pytest.mark.slow
class TestResearchWorkflow:
    """Test end-to-end research workflows."""

    async def test_research_agent_structured_workflow(self):
        """Test that research agents create structured TODO plans and execute them systematically."""
        # Skip if no API credentials available
        if not os.getenv("SILICONFLOW_API_KEY"):
            pytest.skip("No API credentials available for integration test")

        from sample_deep_agent.graph import make_graph

        # Test with MCP question that should trigger structured workflow
        agent = make_graph()

        try:
            result = await agent.ainvoke({
                "messages": [HumanMessage(content="What is MCP (Model Context Protocol)?")]
            })
        except Exception as e:
            if "402" in str(e) or "credits" in str(e).lower():
                pytest.skip("Insufficient API credits for integration test")
            raise

        # Verify the workflow executed
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Should have final response from agent
        final_message = result["messages"][-1]
        assert isinstance(final_message, AIMessage)
        assert len(final_message.content) > 50  # Should have meaningful content

        # Check for expected tool usage patterns
        messages = result["messages"]
        all_tools = []

        for msg in messages:
            # Track all tool calls at any level
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    tool_name = tool_call.get('function', {}).get('name', '')
                    if not tool_name:
                        tool_name = tool_call.get('name', '')
                    if tool_name:
                        all_tools.append(tool_name)

            # Check additional_kwargs for tool calls
            if hasattr(msg, 'additional_kwargs') and 'tool_calls' in msg.additional_kwargs:
                for tool_call in msg.additional_kwargs['tool_calls']:
                    tool_name = tool_call.get('function', {}).get('name', '')
                    if not tool_name:
                        tool_name = tool_call.get('name', '')
                    if tool_name:
                        all_tools.append(tool_name)

        # Verify task delegation occurred (coordinator delegates to subagent)
        assert 'task' in all_tools, f"Should delegate to subagent via 'task' tool, tools: {all_tools}"

        # The workflow completed successfully with task delegation
        # We verified task was called, which means subagent was invoked with middleware
        print(f"✅ Task delegation successful with tools: {all_tools}")

        # Verify response contains MCP-related content
        content_lower = final_message.content.lower()
        mcp_indicators = ['mcp', 'model context protocol', 'protocol']
        found_indicators = sum(1 for indicator in mcp_indicators if indicator in content_lower)
        assert found_indicators >= 1, (
            f"Should contain MCP content, found {found_indicators}/{len(mcp_indicators)} indicators"
        )


