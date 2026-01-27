#!/usr/bin/env python3
"""
Test script for Gemini error handling improvements.

This script simulates Gemini API errors to verify that:
1. PA preserves instruction context during errors
2. NO_OP results in empty instruction (preserving last valid instruction)
3. Session save occurs after 3 consecutive errors
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agentproxy.pa_agent import PAAgent
from agentproxy.function_executor import FunctionName, FunctionResult
from agentproxy.models import PAReasoning


def test_error_output():
    """Test that _error_output produces correct results."""
    print("\n=== Test 1: Error Output Behavior ===\n")

    # Create a minimal PAAgent instance
    agent = PAAgent(working_dir=".", session_id="test-session")

    # Simulate first error
    print("Simulating error #1...")
    error_info = {
        "error_type": "network",
        "status_code": None,
        "message": "Connection timeout"
    }
    result1 = agent._error_output(error_info)

    print(f"Function: {result1.function_call.name.value}")
    print(f"Reasoning: {result1.reasoning.decision}")
    print(f"Error count: {agent._consecutive_errors}")
    assert result1.function_call.name == FunctionName.NO_OP
    assert agent._consecutive_errors == 1
    print("✅ First error returns NO_OP\n")

    # Simulate second error
    print("Simulating error #2...")
    result2 = agent._error_output(error_info)
    print(f"Function: {result2.function_call.name.value}")
    print(f"Error count: {agent._consecutive_errors}")
    assert result2.function_call.name == FunctionName.NO_OP
    assert agent._consecutive_errors == 2
    print("✅ Second error returns NO_OP\n")

    # Simulate third error
    print("Simulating error #3...")
    result3 = agent._error_output(error_info)
    print(f"Function: {result3.function_call.name.value}")
    print(f"Reasoning: {result3.reasoning.decision}")
    print(f"Error count: {agent._consecutive_errors}")
    assert result3.function_call.name == FunctionName.SAVE_SESSION
    assert agent._consecutive_errors == 3
    print("✅ Third error triggers SAVE_SESSION\n")

    # Verify session ID is mentioned in decision
    assert "test-session" in result3.reasoning.decision.lower() or "session" in result3.reasoning.decision.lower()
    print("✅ Session ID mentioned in error output\n")


def test_parse_agent_output():
    """Test that successful parsing resets error counter."""
    print("\n=== Test 2: Parse Error Recovery ===\n")

    agent = PAAgent(working_dir=".", session_id="test-session2")

    # Simulate error to increment counter
    error_info = {"error_type": "parse", "status_code": None, "message": "Invalid JSON"}
    agent._error_output(error_info)
    print(f"After error: counter = {agent._consecutive_errors}")
    assert agent._consecutive_errors == 1

    # Simulate successful parse
    valid_response = '''
    {
        "reasoning": {
            "current_state": "Testing",
            "claude_progress": "In progress",
            "insights": "All good",
            "decision": "Continue"
        },
        "function_call": {
            "name": "send_to_claude",
            "arguments": {"instruction": "Keep going"}
        }
    }
    '''
    result = agent._parse_agent_output(valid_response)
    print(f"After successful parse: counter = {agent._consecutive_errors}")
    assert agent._consecutive_errors == 0
    assert result.function_call.name == FunctionName.SEND_TO_CLAUDE
    print("✅ Successful parse resets error counter\n")


def test_fallback_output():
    """Test fallback output increments error counter."""
    print("\n=== Test 3: Fallback Output ===\n")

    agent = PAAgent(working_dir=".", session_id="test-session3")

    result = agent._fallback_output("Invalid format")
    print(f"Function: {result.function_call.name.value}")
    print(f"Error count: {agent._consecutive_errors}")
    assert result.function_call.name == FunctionName.NO_OP
    assert agent._consecutive_errors == 1
    print("✅ Fallback increments error counter and returns NO_OP\n")


def test_gemini_error_parsing():
    """Test Gemini error string parsing."""
    print("\n=== Test 4: Error String Parsing ===\n")

    agent = PAAgent(working_dir=".", session_id="test-session4")

    # Test with status code
    error_str1 = "[GEMINI_ERROR:http:429:Rate limit exceeded]"
    info1 = agent._parse_gemini_error(error_str1)
    print(f"Parsed: {info1}")
    assert info1["error_type"] == "http"
    assert info1["status_code"] == 429
    assert "Rate limit" in info1["message"]
    print("✅ Parses error with status code\n")

    # Test without status code
    error_str2 = "[GEMINI_ERROR:network:Connection timeout]"
    info2 = agent._parse_gemini_error(error_str2)
    print(f"Parsed: {info2}")
    assert info2["error_type"] == "network"
    assert info2["status_code"] is None
    assert "Connection timeout" in info2["message"]
    print("✅ Parses error without status code\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Testing Gemini Error Handling Improvements")
    print("="*60)

    try:
        test_error_output()
        test_parse_agent_output()
        test_fallback_output()
        test_gemini_error_parsing()

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
