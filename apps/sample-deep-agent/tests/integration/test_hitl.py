"""Integration tests for human-in-the-loop (HITL) functionality."""

import os
import uuid

import pytest
from langchain.messages import HumanMessage
from langgraph.types import Command


@pytest.mark.integration
@pytest.mark.slow
class TestHITLWorkflow:
    """Test end-to-end HITL workflows with interrupts."""

    async def test_comprehensive_hitl_workflow(self):
        """Test comprehensive HITL workflow.

        - Top-level task: interrupt with approve/reject only, approve it
        - think_tool calls: allow all (no interrupts)
        - write_todos: allow but limit to max 1 todo
        - All deep_web_search calls: reject all at all levels
        - Verify no deep_web_search in final message list
        """
        # Skip if no API credentials available
        if not os.getenv("SILICONFLOW_API_KEY") or not os.getenv("TAVILY_API_KEY"):
            pytest.skip("No API credentials available for integration test")

        from sample_deep_agent.context import DeepAgentContext
        from sample_deep_agent.graph import make_graph

        # Create context with max_todos limit
        context = DeepAgentContext(
            max_todos=1,  # Limit to 1 todo to prevent excessive planning
        )

        # Create graph with HITL configuration
        from dataclasses import asdict

        config = {"configurable": asdict(context)}

        # Define interrupt configuration
        interrupt_on = {
            "task": {"allowed_decisions": ["approve", "reject"]},  # Only approve/reject
            "write_todos": False,  # Don't interrupt write_todos
            "think_tool": False,  # Don't interrupt think_tool
            "deep_web_search": True,  # Interrupt at top level
        }

        subagent_interrupts = {
            "research-agent": {
                "deep_web_search": True,  # Interrupt in subagent too
                "think_tool": False,  # Don't interrupt think_tool in subagent
            }
        }

        agent = make_graph(config, interrupt_on=interrupt_on, subagent_interrupts=subagent_interrupts)

        # Use thread_id for state persistence (required for HITL)
        thread_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": thread_id}}

        try:
            # Invoke the agent with a research task that requires web search
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="What are the core features of LangChain v1?")]},
                config=thread_config,
            )
        except Exception as e:
            if "402" in str(e) or "credits" in str(e).lower():
                pytest.skip("Insufficient API credits for integration test")
            raise

        # Track statistics
        task_approved = False
        deep_web_search_rejected_count = 0

        max_iterations = 20  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            if result.get("__interrupt__"):
                interrupts = result["__interrupt__"][0].value
                action_requests = interrupts["action_requests"]

                # Check what tools are being interrupted
                tool_names = [action["name"] for action in action_requests]
                print(f"Iteration {iteration}: Interrupted for tools: {tool_names}")

                # Process each action request
                decisions = []
                for action in action_requests:
                    tool_name = action["name"]

                    if tool_name == "task":
                        if not task_approved:
                            print("✅ Approving task (only approve/reject allowed)")
                            decisions.append({"type": "approve"})
                            task_approved = True
                        else:
                            print("❌ Rejecting subsequent task call")
                            decisions.append({"type": "reject"})

                    elif tool_name == "deep_web_search":
                        print(f"❌ Rejecting deep_web_search call #{deep_web_search_rejected_count + 1}")
                        decisions.append({"type": "reject"})
                        deep_web_search_rejected_count += 1

                    else:
                        # For other tools, approve
                        print(f"✅ Approving other tool: {tool_name}")
                        decisions.append({"type": "approve"})

                # Resume execution with decisions
                result = await agent.ainvoke(Command(resume={"decisions": decisions}), config=thread_config)
            else:
                # No more interrupts - workflow completed
                print("Workflow completed without further interrupts")
                break

            iteration += 1

        # Verify we got a result
        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 1

        # Verify task was approved
        assert task_approved, "Task should have been approved"

        # Verify at least one deep_web_search was rejected
        assert deep_web_search_rejected_count > 0, "Should have rejected at least one deep_web_search call"

        # Verify no deep_web_search was executed (check for ToolMessage responses)
        # Note: AIMessages may contain rejected tool_calls, but we check for actual execution
        tool_messages = [msg for msg in result["messages"] if msg.__class__.__name__ == "ToolMessage"]
        for tool_msg in tool_messages:
            # ToolMessage.name contains the tool that was executed
            if hasattr(tool_msg, "name"):
                assert tool_msg.name != "deep_web_search", (
                    f"Found executed deep_web_search in ToolMessage - should have been rejected. "
                    f"Content: {tool_msg.content[:200]}"
                )

        # Report summary
        print("\n📊 Summary:")
        print("  - task: approved")
        print(f"  - deep_web_search calls rejected: {deep_web_search_rejected_count}")
        print("  - max_todos limit: 1")
        print(f"  - Total messages in result: {len(result['messages'])}")

        # The agent should have responded
        final_message = result["messages"][-1]
        assert len(final_message.content) > 0

        print("\n✅ Successfully completed comprehensive HITL workflow")
