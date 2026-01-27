"""
Integration test: Gemini DONE Classifier Gate e2e
==================================================

Exercises the full _run_task_single_worker loop with the classifier gate,
mocking only external I/O (Claude subprocess, Gemini API, git).  The gate
logic, delta tracking, and state transitions run for real.

Scenarios tested:
  A. Gate skipped (early rounds with code changes)
  B. Gate triggers → DONE high confidence → state=DONE
  C. Gate triggers → STOP high confidence → state=STOPPED
  D. Gate triggers → ERROR high confidence → state=ERROR
  E. Gate triggers → DONE low confidence → hint injected, loop continues
  F. Gate triggers → CONTINUE with stalling → warning injected, loop continues
  G. Round deltas accumulate correctly
  H. Verification PASS triggers gate immediately
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch, call

import pytest

from agentproxy.pa import PA, _jaccard_similarity
from agentproxy.models import (
    ControllerState,
    EventType,
    OutputEvent,
    PAReasoning,
)
from agentproxy.function_executor import FunctionName, FunctionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pa() -> PA:
    """Create a PA instance with Gemini client stubbed out."""
    with patch("agentproxy.pa_agent.GeminiClient"):
        with patch("agentproxy.pa_agent.load_dotenv"):
            pa = PA(working_dir=".", session_id="gate-e2e", display_mode="none")
    return pa


def _noop_gen(*_a, **_kw):
    """Generator that yields nothing."""
    return
    yield  # noqa: make it a generator


def _fake_stream_claude(output_text: str):
    """Return a generator factory that yields one TEXT event."""
    def gen(self, instruction, iteration=0):
        yield OutputEvent(
            event_type=EventType.TEXT,
            content=output_text,
            metadata={"source": "claude"},
        )
    return gen


def _fake_verification(result_text: str):
    """Return a generator factory that yields one TOOL_RESULT event."""
    def gen(self, task, changed_files):
        yield OutputEvent(
            event_type=EventType.TOOL_RESULT,
            content=result_text,
            metadata={"source": "pa"},
        )
    return gen


def _make_run_iteration_result(func_name=FunctionName.SEND_TO_CLAUDE, done=False):
    """Build a (reasoning, result) pair for agent.run_iteration."""
    reasoning = PAReasoning(
        current_state="testing",
        claude_progress="progress",
        insights="none",
        decision="continue",
    )
    result = FunctionResult(
        name=func_name,
        success=True,
        output="ok",
        metadata={"done": done},
    )
    return reasoning, result


def _collect_events(pa: PA, task: str, max_iter: int = 10) -> list[OutputEvent]:
    """Drain the run_task generator and return all events."""
    return list(pa.run_task(task, max_iterations=max_iter))


def _seed_zero_code_deltas(pa: PA, n: int = 3) -> None:
    """Pre-populate _round_deltas with n zero-code rounds.

    This lets no_code_rounds >= 3 from the very first iteration
    so the classifier gate fires immediately.
    """
    for i in range(n):
        pa._round_deltas.append({
            "round": -(n - i),  # negative rounds = seed
            "lines_added": 0,
            "lines_removed": 0,
            "output_similarity": 0.0,
            "instruction_sent": "seed",
        })


def _setup_common_mocks(pa: PA) -> None:
    """Wire up the standard mock set shared by most scenarios."""
    pa.agent.self_check = MagicMock(return_value=(True, "OK"))
    pa.agent.generate_session_summary = MagicMock(return_value="summary")
    pa.agent.save_session_summary = MagicMock()
    pa.agent.load_session_summary = MagicMock(return_value=None)
    pa.agent.smart_update_task_status = MagicMock(return_value=None)
    pa.agent.get_claude_instruction = MagicMock(return_value=None)


# ---------------------------------------------------------------------------
# Scenario A: Gate skipped when code is progressing
# ---------------------------------------------------------------------------

class TestGateSkippedWhileProgressing:
    """
    When every round has code changes, no_code_rounds == 0,
    verification is vacuous, and Claude hasn't signaled stream-done,
    the classifier should NEVER be invoked.
    """

    def test_classifier_not_called_during_progress(self):
        pa = _make_pa()
        _setup_common_mocks(pa)

        call_count = {"classify": 0, "iteration": 0}

        def classify_done_spy(*args, **kwargs):
            call_count["classify"] += 1
            return "CONTINUE", 0.5, "still working"

        pa.agent.classify_done = classify_done_spy

        def run_iteration_side_effect(ctx):
            call_count["iteration"] += 1
            if call_count["iteration"] >= 3:
                # Simulate the real run_iteration setting _is_done
                pa.agent._is_done = True
                return _make_run_iteration_result(FunctionName.MARK_DONE, done=True)
            return _make_run_iteration_result()

        pa.agent.run_iteration = run_iteration_side_effect

        # File tracker: always returns code changes
        pa._file_tracker.get_code_changes = MagicMock(return_value=(20, 5))
        pa._file_tracker.get_changed_files = MagicMock(return_value=["app.py"])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        with patch.object(PA, '_stream_claude', _fake_stream_claude("I wrote code")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("No scripts found — skipped")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=5)

        # Classifier should never have been called
        assert call_count["classify"] == 0, (
            f"Classifier was called {call_count['classify']} times "
            "during rounds with code changes"
        )
        assert pa._state == ControllerState.DONE
        # Deltas should have been recorded
        assert len(pa._round_deltas) >= 2
        # Every delta should show code changes
        for d in pa._round_deltas:
            assert d["lines_added"] == 20


# ---------------------------------------------------------------------------
# Scenario B: DONE high confidence → state=DONE
# ---------------------------------------------------------------------------

class TestDoneHighConfidence:
    """
    When the classifier returns DONE with confidence >= 0.8
    the loop should break and set state=DONE.

    Pre-seed 3 zero-code deltas so the gate fires on round 1.
    """

    def test_done_breaks_loop(self):
        pa = _make_pa()
        _setup_common_mocks(pa)
        _seed_zero_code_deltas(pa, 3)

        pa.agent.run_iteration = MagicMock(
            return_value=_make_run_iteration_result()
        )

        # Classifier returns DONE high confidence
        pa.agent.classify_done = MagicMock(
            return_value=("DONE", 0.95, "All requirements satisfied")
        )

        # File tracker: no code changes
        pa._file_tracker.get_code_changes = MagicMock(return_value=(0, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=[])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("I finished the task")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("No scripts found — skipped")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        assert pa._state == ControllerState.DONE
        assert pa.agent._is_done is True

        # Should see the DONE message in events
        done_events = [e for e in events if "DONE" in (e.content or "")]
        assert len(done_events) >= 1
        assert "confidence=0.95" in done_events[0].content

        # run_iteration should NOT have been called (we broke before it)
        pa.agent.run_iteration.assert_not_called()

        # Classifier was called immediately (seeded no_code_rounds >= 3)
        assert pa.agent.classify_done.call_count == 1


# ---------------------------------------------------------------------------
# Scenario C: STOP high confidence → state=STOPPED
# ---------------------------------------------------------------------------

class TestStopHighConfidence:
    """
    When the classifier returns STOP with confidence >= 0.8
    the loop should break and set state=STOPPED.
    """

    def test_stop_breaks_loop(self):
        pa = _make_pa()
        _setup_common_mocks(pa)
        _seed_zero_code_deltas(pa, 3)

        pa.agent.run_iteration = MagicMock(
            return_value=_make_run_iteration_result()
        )

        pa.agent.classify_done = MagicMock(
            return_value=("STOP", 0.90, "Claude is looping without progress")
        )

        pa._file_tracker.get_code_changes = MagicMock(return_value=(0, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=[])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("Repeating myself again")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("FAIL: tests broken")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        assert pa._state == ControllerState.STOPPED

        stop_events = [e for e in events if "STOPPED" in (e.content or "")]
        assert len(stop_events) >= 1
        assert "confidence=0.90" in stop_events[0].content

        # Should have broken on the first iteration
        pa.agent.run_iteration.assert_not_called()


# ---------------------------------------------------------------------------
# Scenario D: ERROR high confidence → state=ERROR
# ---------------------------------------------------------------------------

class TestErrorHighConfidence:
    """
    When the classifier returns ERROR with confidence >= 0.8
    the loop should break and set state=ERROR.
    """

    def test_error_breaks_loop(self):
        pa = _make_pa()
        _setup_common_mocks(pa)

        pa.agent.run_iteration = MagicMock(
            return_value=_make_run_iteration_result()
        )

        # Classifier returns ERROR on any gate trigger
        pa.agent.classify_done = MagicMock(
            return_value=("ERROR", 0.88, "Missing required dependency numpy")
        )

        pa._file_tracker.get_code_changes = MagicMock(return_value=(0, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=[])
        # claude_stream_done=True triggers gate immediately
        type(pa._file_tracker).is_done = PropertyMock(return_value=True)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("ImportError: no module numpy")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("FAIL: ImportError")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        assert pa._state == ControllerState.ERROR

        error_events = [e for e in events
                        if e.event_type == EventType.ERROR and "ERROR" in (e.content or "")]
        assert len(error_events) >= 1
        assert "confidence=0.88" in error_events[0].content


# ---------------------------------------------------------------------------
# Scenario E: DONE low confidence → hint injected, loop continues
# ---------------------------------------------------------------------------

class TestDoneLowConfidence:
    """
    When the classifier returns DONE with confidence < 0.8
    the loop should NOT break — instead it injects a hint into the
    reasoning context and continues to the agent.run_iteration call.
    """

    def test_low_confidence_injects_hint(self):
        pa = _make_pa()
        _setup_common_mocks(pa)

        # Track what context the reasoning loop receives
        captured_contexts = []

        def run_iteration_capture(ctx):
            captured_contexts.append(ctx)
            # Mark done on second call to stop the loop
            if len(captured_contexts) >= 2:
                pa.agent._is_done = True
                return _make_run_iteration_result(FunctionName.MARK_DONE, done=True)
            return _make_run_iteration_result()

        pa.agent.run_iteration = run_iteration_capture

        # First call: low-confidence DONE. Second call: high-confidence DONE.
        call_seq = iter([
            ("DONE", 0.55, "Maybe done"),
            ("DONE", 0.95, "Definitely done"),
        ])
        pa.agent.classify_done = MagicMock(side_effect=lambda *a, **kw: next(call_seq))

        pa._file_tracker.get_code_changes = MagicMock(return_value=(0, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=[])
        # claude_stream_done=True triggers gate immediately
        type(pa._file_tracker).is_done = PropertyMock(return_value=True)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("Probably done")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("No scripts found — skipped")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        # First iteration: low confidence → hint injected → run_iteration called
        assert len(captured_contexts) >= 1
        assert "[DONE CLASSIFIER] confidence=0.55" in captured_contexts[0]
        assert "MARK_DONE" in captured_contexts[0]


# ---------------------------------------------------------------------------
# Scenario F: CONTINUE with stalling → warning injected
# ---------------------------------------------------------------------------

class TestContinueWithStalling:
    """
    When the classifier returns CONTINUE but no_code_rounds >= 3,
    a PROGRESS WARNING should be injected into the reasoning context.

    Pre-seed 3 zero-code deltas so the gate fires from round 1.
    """

    def test_stall_warning_injected(self):
        pa = _make_pa()
        _setup_common_mocks(pa)
        _seed_zero_code_deltas(pa, 3)

        captured_contexts = []

        def run_iteration_capture(ctx):
            captured_contexts.append(ctx)
            if len(captured_contexts) >= 2:
                pa.agent._is_done = True
                return _make_run_iteration_result(FunctionName.MARK_DONE, done=True)
            return _make_run_iteration_result()

        pa.agent.run_iteration = run_iteration_capture

        # Always return CONTINUE
        pa.agent.classify_done = MagicMock(
            return_value=("CONTINUE", 0.7, "Still working")
        )

        pa._file_tracker.get_code_changes = MagicMock(return_value=(0, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=[])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("Not sure what to do")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("No scripts found — skipped")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        # Iteration 1: no_code_rounds >= 3 (seeded) + CONTINUE → warning injected
        stall_contexts = [c for c in captured_contexts if "[PROGRESS WARNING]" in c]
        assert len(stall_contexts) >= 1, (
            "Expected PROGRESS WARNING to be injected into context "
            f"but got contexts: {[c[-200:] for c in captured_contexts]}"
        )
        assert "No code changes for" in stall_contexts[0]


# ---------------------------------------------------------------------------
# Scenario G: Round deltas accumulate correctly through the loop
# ---------------------------------------------------------------------------

class TestDeltaAccumulationE2E:
    """
    Verify that _round_deltas grows with each iteration and records
    the correct lines_added/removed and output_similarity values.
    """

    def test_deltas_recorded_per_round(self):
        pa = _make_pa()
        _setup_common_mocks(pa)

        iteration_count = {"n": 0}

        def run_iteration_side_effect(ctx):
            iteration_count["n"] += 1
            if iteration_count["n"] >= 3:
                pa.agent._is_done = True
                return _make_run_iteration_result(FunctionName.MARK_DONE, done=True)
            return _make_run_iteration_result()

        pa.agent.run_iteration = run_iteration_side_effect

        # Varying code changes per iteration
        code_changes = iter([(30, 0), (10, 5), (0, 0)])
        pa._file_tracker.get_code_changes = MagicMock(
            side_effect=lambda: next(code_changes, (0, 0))
        )
        pa._file_tracker.get_changed_files = MagicMock(return_value=["app.py"])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        # Classifier for the round(s) where it triggers
        pa.agent.classify_done = MagicMock(
            return_value=("CONTINUE", 0.5, "working")
        )

        # Outputs that share some words so similarity > 0 between rounds
        outputs = iter([
            "Creating new app files and setting up project",
            "Editing app files and fixing the project config",
            "Creating new app files and setting up project",
        ])

        def stream_claude_varying(self, instruction, iteration=0):
            yield OutputEvent(
                event_type=EventType.TEXT,
                content=next(outputs, "done"),
                metadata={"source": "claude"},
            )

        with patch.object(PA, '_stream_claude', stream_claude_varying), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("No scripts found — skipped")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build something", max_iter=5)

        assert len(pa._round_deltas) >= 2

        # Round 1: 30 added, 0 removed, similarity=0 (first round)
        assert pa._round_deltas[0]["lines_added"] == 30
        assert pa._round_deltas[0]["lines_removed"] == 0
        assert pa._round_deltas[0]["output_similarity"] == 0.0
        assert pa._round_deltas[0]["round"] == 1

        # Round 2: 10 added, 5 removed
        assert pa._round_deltas[1]["lines_added"] == 10
        assert pa._round_deltas[1]["lines_removed"] == 5
        assert pa._round_deltas[1]["round"] == 2
        # Shared words: "app", "files", "and", "project", "the" → similarity > 0
        assert pa._round_deltas[1]["output_similarity"] > 0.0


# ---------------------------------------------------------------------------
# Scenario H: Verification PASS triggers the gate even on round 1
# ---------------------------------------------------------------------------

class TestVerificationTriggersGate:
    """
    When verification_passed=True, the gate fires regardless of
    no_code_rounds count (even on the first round).
    """

    def test_verification_pass_invokes_classifier_immediately(self):
        pa = _make_pa()
        _setup_common_mocks(pa)

        pa.agent.run_iteration = MagicMock(
            return_value=_make_run_iteration_result()
        )

        pa.agent.classify_done = MagicMock(
            return_value=("DONE", 0.92, "Verification passed, task complete")
        )

        # Code IS changing — but verification PASSES
        pa._file_tracker.get_code_changes = MagicMock(return_value=(50, 0))
        pa._file_tracker.get_changed_files = MagicMock(return_value=["main.py"])
        type(pa._file_tracker).is_done = PropertyMock(return_value=False)

        with patch.object(PA, '_stream_claude',
                          _fake_stream_claude("Tests pass!")), \
             patch.object(PA, '_run_auto_verification',
                          _fake_verification("PASS: All 12 tests passed")), \
             patch.object(PA, '_ensure_git_repo'), \
             patch.object(PA, '_setup_task_breakdown', _noop_gen):

            events = _collect_events(pa, "Build a widget", max_iter=10)

        # Classifier was called on iteration 1 because verification_passed=True
        assert pa.agent.classify_done.call_count >= 1
        assert pa._state == ControllerState.DONE
        # Only 1 round of deltas (broke immediately)
        assert len(pa._round_deltas) == 1
        assert pa._round_deltas[0]["lines_added"] == 50
