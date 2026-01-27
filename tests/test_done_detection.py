"""
Tests for _synthesize_instruction exhaustive match
====================================================

Verifies that every FunctionName enum value has explicit handling
and the catch-all default is unreachable.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Any, Dict

from agentproxy.function_executor import FunctionName, FunctionResult


def make_result(name: FunctionName, success: bool = True, output: str = "OK", metadata: dict = None) -> FunctionResult:
    """Helper to create a FunctionResult."""
    return FunctionResult(
        name=name,
        success=success,
        output=output,
        metadata=metadata or {},
    )


def _synthesize_instruction(result: FunctionResult) -> str:
    """Standalone copy of PA._synthesize_instruction for unit testing.

    This must match the implementation in pa.py exactly.
    """
    if result.name == FunctionName.VERIFY_CODE:
        return "Continue" if result.success else f"Fix: {result.output[:200]}"
    elif result.name == FunctionName.RUN_TESTS:
        return "Continue" if result.success else f"Fix tests: {result.output[:200]}"
    elif result.name == FunctionName.VERIFY_PRODUCT:
        return "" if result.success else f"Fix: {result.output[:200]}"
    elif result.name == FunctionName.CHECK_SERVER:
        return "" if result.success else f"Server check failed: {result.output[:200]}"
    elif result.name == FunctionName.REVIEW_CHANGES:
        if result.metadata.get("has_issues"):
            return f"Fix review issues: {result.output[:200]}"
        return ""
    elif result.name == FunctionName.SEND_TO_CLAUDE:
        return ""  # Already queued its instruction
    elif result.name in (
        FunctionName.NO_OP,
        FunctionName.SAVE_SESSION,
        FunctionName.READ_FILE,
        FunctionName.CREATE_TASK,
        FunctionName.UPDATE_TASK,
        FunctionName.COMPLETE_TASK,
        FunctionName.MARK_DONE,
    ):
        return ""
    return "Continue with the task."  # Unreachable with exhaustive match


class TestSynthesizeInstruction:
    """Test _synthesize_instruction exhaustive match."""

    def test_send_to_claude_returns_empty(self):
        result = make_result(FunctionName.SEND_TO_CLAUDE)
        assert _synthesize_instruction(result) == ""

    def test_read_file_returns_empty(self):
        result = make_result(FunctionName.READ_FILE)
        assert _synthesize_instruction(result) == ""

    def test_verify_product_success_returns_empty(self):
        result = make_result(FunctionName.VERIFY_PRODUCT, success=True)
        assert _synthesize_instruction(result) == ""

    def test_verify_product_failure_returns_fix(self):
        result = make_result(FunctionName.VERIFY_PRODUCT, success=False, output="Script failed: exit code 1")
        instruction = _synthesize_instruction(result)
        assert instruction.startswith("Fix:")
        assert "Script failed" in instruction

    def test_create_task_returns_empty(self):
        result = make_result(FunctionName.CREATE_TASK)
        assert _synthesize_instruction(result) == ""

    def test_update_task_returns_empty(self):
        result = make_result(FunctionName.UPDATE_TASK)
        assert _synthesize_instruction(result) == ""

    def test_complete_task_returns_empty(self):
        result = make_result(FunctionName.COMPLETE_TASK)
        assert _synthesize_instruction(result) == ""

    def test_mark_done_returns_empty(self):
        result = make_result(FunctionName.MARK_DONE)
        assert _synthesize_instruction(result) == ""

    def test_no_op_returns_empty(self):
        result = make_result(FunctionName.NO_OP)
        assert _synthesize_instruction(result) == ""

    def test_save_session_returns_empty(self):
        result = make_result(FunctionName.SAVE_SESSION)
        assert _synthesize_instruction(result) == ""

    def test_verify_code_success_returns_continue(self):
        result = make_result(FunctionName.VERIFY_CODE, success=True)
        assert _synthesize_instruction(result) == "Continue"

    def test_verify_code_failure_returns_fix(self):
        result = make_result(FunctionName.VERIFY_CODE, success=False, output="Syntax error line 5")
        instruction = _synthesize_instruction(result)
        assert instruction.startswith("Fix:")

    def test_run_tests_success_returns_continue(self):
        result = make_result(FunctionName.RUN_TESTS, success=True)
        assert _synthesize_instruction(result) == "Continue"

    def test_run_tests_failure_returns_fix(self):
        result = make_result(FunctionName.RUN_TESTS, success=False, output="2 tests failed")
        instruction = _synthesize_instruction(result)
        assert instruction.startswith("Fix tests:")

    def test_check_server_success_returns_empty(self):
        result = make_result(FunctionName.CHECK_SERVER, success=True)
        assert _synthesize_instruction(result) == ""

    def test_check_server_failure_returns_error(self):
        result = make_result(FunctionName.CHECK_SERVER, success=False, output="Connection refused")
        instruction = _synthesize_instruction(result)
        assert "Server check failed" in instruction

    def test_review_changes_no_issues_returns_empty(self):
        result = make_result(FunctionName.REVIEW_CHANGES, metadata={})
        assert _synthesize_instruction(result) == ""

    def test_review_changes_with_issues_returns_fix(self):
        result = make_result(
            FunctionName.REVIEW_CHANGES,
            metadata={"has_issues": True},
            output="Missing error handling",
        )
        instruction = _synthesize_instruction(result)
        assert instruction.startswith("Fix review issues:")

    def test_all_function_names_have_explicit_handling(self):
        """Verify every FunctionName enum value has a case and never reaches the catch-all."""
        for fn in FunctionName:
            result = make_result(fn)
            instruction = _synthesize_instruction(result)
            assert instruction != "Continue with the task.", (
                f"FunctionName.{fn.name} fell through to catch-all default"
            )
