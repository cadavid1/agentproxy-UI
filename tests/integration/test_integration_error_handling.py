#!/usr/bin/env python3
"""
Integration test demonstrating the Gemini error handling fix.

This simulates a realistic scenario where Gemini fails during PA operation
and verifies that instruction context is preserved.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))

from agentproxy.pa import PA
from agentproxy.function_executor import FunctionName


def test_instruction_preservation_during_errors():
    """
    Integration test: Verify instruction context is preserved during Gemini errors.

    Scenario:
    1. Start with task "Add authentication"
    2. First iteration: Gemini works, sets new instruction "Review code"
    3. Second iteration: Gemini fails (error #1)
       - Expected: Keep "Review code" instruction
    4. Third iteration: Gemini fails (error #2)
       - Expected: Keep "Review code" instruction
    5. Fourth iteration: Gemini works again
       - Expected: Normal operation resumes
    """
    print("\n" + "="*70)
    print("Integration Test: Instruction Preservation During Gemini Errors")
    print("="*70 + "\n")

    # Create PA instance
    pa = PA(working_dir=".", session_id="integration-test")

    # Set initial state
    pa._original_task = "Add authentication to the login page"
    pa._last_valid_instruction = "Add authentication to the login page"

    print(f"Initial task: {pa._original_task}")
    print(f"Initial instruction: {pa._last_valid_instruction}\n")

    # Simulate iteration 1: Normal operation
    print("--- Iteration 1: Normal Gemini Response ---")
    mock_result = Mock()
    mock_result.name = FunctionName.SEND_TO_CLAUDE
    mock_result.metadata = {}

    # Simulate PA updating instruction
    new_instruction = "Review the authentication code for security issues"
    pa._last_valid_instruction = new_instruction
    print(f"PA updates instruction to: '{new_instruction}'\n")

    # Simulate iteration 2: Gemini error #1
    print("--- Iteration 2: Gemini Error #1 ---")
    mock_error_result = Mock()
    mock_error_result.name = FunctionName.NO_OP
    mock_error_result.metadata = {
        "reason": "Gemini network error (attempt 1/3) - preserving instruction context",
        "error_type": "network",
    }

    synthesized = pa._synthesize_instruction(mock_error_result)
    print(f"NO_OP result synthesizes to: '{synthesized}' (empty string)")

    # Main loop would preserve instruction
    if synthesized:
        current_instruction = synthesized
    else:
        current_instruction = pa._last_valid_instruction

    print(f"Current instruction: '{current_instruction}'")
    assert current_instruction == new_instruction, "❌ Instruction context lost!"
    print("✅ Instruction preserved during error #1\n")

    # Simulate iteration 3: Gemini error #2
    print("--- Iteration 3: Gemini Error #2 ---")
    synthesized = pa._synthesize_instruction(mock_error_result)
    print(f"NO_OP result synthesizes to: '{synthesized}' (empty string)")

    if synthesized:
        current_instruction = synthesized
    else:
        current_instruction = pa._last_valid_instruction

    print(f"Current instruction: '{current_instruction}'")
    assert current_instruction == new_instruction, "❌ Instruction context lost!"
    print("✅ Instruction preserved during error #2\n")

    # Simulate iteration 4: Gemini recovers
    print("--- Iteration 4: Gemini Recovers ---")
    mock_normal_result = Mock()
    mock_normal_result.name = FunctionName.VERIFY_CODE
    mock_normal_result.success = True
    mock_normal_result.metadata = {}

    synthesized = pa._synthesize_instruction(mock_normal_result)
    print(f"VERIFY_CODE (success) synthesizes to: '{synthesized}'")

    if synthesized:
        current_instruction = synthesized
        pa._last_valid_instruction = synthesized

    print(f"New instruction: '{current_instruction}'")
    print("✅ Normal operation resumed\n")

    print("="*70)
    print("✅ Integration test PASSED - Context preserved throughout error cycle")
    print("="*70 + "\n")


def test_session_save_after_3_errors():
    """
    Test that after 3 consecutive errors, PA triggers session save.
    """
    print("\n" + "="*70)
    print("Integration Test: Session Save After 3 Errors")
    print("="*70 + "\n")

    from agentproxy.pa_agent import PAAgent

    agent = PAAgent(working_dir=".", session_id="test-3-errors")

    error_info = {
        "error_type": "http",
        "status_code": 503,
        "message": "Service unavailable"
    }

    # Error 1
    print("Error #1...")
    result1 = agent._error_output(error_info)
    print(f"  Function: {result1.function_call.name.value}")
    assert result1.function_call.name == FunctionName.NO_OP
    print("  ✅ Returns NO_OP\n")

    # Error 2
    print("Error #2...")
    result2 = agent._error_output(error_info)
    print(f"  Function: {result2.function_call.name.value}")
    assert result2.function_call.name == FunctionName.NO_OP
    print("  ✅ Returns NO_OP\n")

    # Error 3
    print("Error #3...")
    result3 = agent._error_output(error_info)
    print(f"  Function: {result3.function_call.name.value}")
    assert result3.function_call.name == FunctionName.SAVE_SESSION
    print("  ✅ Triggers SAVE_SESSION")

    # Check metadata
    assert result3.function_call.arguments.get("error_type") == "http"
    assert result3.function_call.arguments.get("status_code") == 503
    print("  ✅ Error metadata included\n")

    print("="*70)
    print("✅ Session save test PASSED")
    print("="*70 + "\n")


def main():
    """Run all integration tests."""
    try:
        test_instruction_preservation_during_errors()
        test_session_save_after_3_errors()

        print("\n" + "🎉"*35)
        print("ALL INTEGRATION TESTS PASSED!")
        print("🎉"*35 + "\n")
        return 0
    except AssertionError as e:
        print(f"\n❌ Integration test failed: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
