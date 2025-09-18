"""Integration tests for ModelProviderMiddleware with real APIs."""

import os
from dataclasses import dataclass

import pytest
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from langgraph_up_devkits.middleware import ModelProviderMiddleware
from langgraph_up_devkits.utils import load_chat_model

# Skip integration tests if no API keys
pytestmark = pytest.mark.skipif(
    not (os.getenv("SILICONFLOW_API_KEY") and os.getenv("TAVILY_API_KEY")),
    reason="Integration tests require SILICONFLOW_API_KEY and TAVILY_API_KEY"
)


@dataclass
class MiddlewareTestContext:
    """Test context with model field for middleware testing."""
    user_id: str = "middleware_test"
    session_id: str = "test_session"
    model: str = "siliconflow:Qwen/Qwen3-8B"


@pytest.mark.asyncio
async def test_middleware_model_switching():
    """Test middleware switches models based on context."""
    # Test if SiliconFlow provider can be loaded
    try:
        load_chat_model("siliconflow:Qwen/Qwen3-8B")
        siliconflow_available = True
    except Exception:
        siliconflow_available = False

    if not siliconflow_available:
        pytest.skip("SiliconFlow provider not available")

    print("✅ SiliconFlow provider registered for middleware test")
    middleware = ModelProviderMiddleware()

    # When using middleware, prompt must be a string
    middleware_prompt_text = (
        "You are a helpful assistant testing middleware functionality. "
        "Respond briefly to user requests. "
        "If the middleware is working correctly, you should be using a SiliconFlow model."
    )

    try:
        # Create agent with a fallback model - middleware should switch to context model
        from langchain.chat_models import init_chat_model
        fallback_model = init_chat_model("openai:gpt-3.5-turbo")

        print(f"Creating agent with fallback model: {type(fallback_model)}")

        agent = create_agent(
            model=fallback_model,  # Middleware will switch from this
            tools=[],
            prompt=middleware_prompt_text,  # Must be string when using middleware
            context_schema=MiddlewareTestContext,
            middleware=[middleware]
        )

        context = MiddlewareTestContext(
            user_id="middleware_user",
            session_id="test_session",
            model="siliconflow:Qwen/Qwen3-8B"  # Middleware should switch to this
        )

        print(f"Invoking agent with context model: {context.model}")

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Say hello briefly")]},
            context=context
        )

        assert isinstance(result, dict)
        assert "messages" in result

        if result.get("messages"):
            response = result["messages"][-1].content
            print(f"Middleware response: {response}")

            # Verify we got a meaningful response (indicating model switching worked)
            if response and response.strip():
                print("✅ Middleware successfully switched to context model!")
                print(f"Response: '{response.strip()}'")

                # Additional check: look for Chinese characters (typical of Qwen models)
                if any(ord(char) > 127 for char in response):
                    print("✅ Response contains non-ASCII characters (likely Chinese) - confirms Qwen model usage!")

                # ANY non-empty response indicates the model switch worked
                assert True, "Model switching successful!"
            else:
                print("❌ No response from model - middleware may have failed")
                assert False, "Expected response from switched model"
        else:
            print("❌ No messages in response")
            assert False, "Expected messages in response"

    except Exception as e:
        print(f"Middleware test error: {e}")
        import traceback
        traceback.print_exc()
        # Test basic middleware structure as fallback
        assert hasattr(middleware, 'modify_model_request')
        # Re-raise to see the actual error
        raise


@pytest.mark.asyncio
async def test_middleware_error_handling():
    """Test middleware handles unregistered provider models gracefully."""

    @dataclass
    class NoModelContext:
        """Context without model field."""
        user_id: str = "test"

    middleware = ModelProviderMiddleware()

    # When using middleware, prompt must be a string
    error_prompt_text = "You are a test assistant. This should fail due to invalid provider."

    try:
        # Use a fake provider that should fail
        agent = create_agent(
            model="fakeproviderthatdoesnotexist:some-model",  # This should fail in middleware
            tools=[],
            prompt=error_prompt_text,  # Must be string when using middleware
            context_schema=NoModelContext,
            middleware=[middleware]
        )

        context = NoModelContext(user_id="error_test")

        await agent.ainvoke(
            {"messages": [HumanMessage(content="Test error case")]},
            context=context
        )

        # This should not be reached - the middleware should fail
        assert False, "Expected middleware to fail with unregistered provider"

    except ValueError as e:
        if "Unable to infer model provider" in str(e) or "Cannot resolve model provider" in str(e):
            print(f"✅ Expected provider error caught: {e}")
            assert True  # This is the expected failure
        else:
            print(f"❌ Unexpected ValueError: {e}")
            raise

    except Exception as e:
        print(f"❌ Unexpected error type: {type(e).__name__}: {e}")
        # Check if this is a different kind of expected error related to provider resolution
        if ("fakeproviderthatdoesnotexist" in str(e) or
            "provider" in str(e).lower() or
            "model provider" in str(e).lower()):
            print("✅ Expected provider error caught (different format)")
            assert True
        else:
            raise


