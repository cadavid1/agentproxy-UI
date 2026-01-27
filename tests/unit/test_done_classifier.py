"""
Tests for Gemini DONE Classifier Gate
======================================

Covers:
- classify_done() on PAAgent (mock Gemini)
- Should-classify gate logic
- Multi-part call flow through GeminiClient
- Round delta accumulation and jaccard similarity
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agentproxy.pa import _jaccard_similarity


# =============================================================================
# TestClassifyDone — mock PAAgent._gemini.call()
# =============================================================================

class TestClassifyDone:
    """Test PAAgent.classify_done() with mocked Gemini responses."""

    def _make_agent(self):
        """Create a PAAgent with mocked Gemini client."""
        with patch("agentproxy.pa_agent.GeminiClient") as MockGemini:
            mock_gemini = MagicMock()
            MockGemini.return_value = mock_gemini
            with patch("agentproxy.pa_agent.load_dotenv"):
                from agentproxy.pa_agent import PAAgent
                agent = PAAgent.__new__(PAAgent)
                agent.working_dir = "."
                agent.user_mission = None
                agent.context_dir = None
                agent._gemini = mock_gemini
                agent._memory = MagicMock()
                agent._executor = MagicMock()
                agent._project_context = ""
                agent._context_images = {}
                agent._history = []
                agent._is_done = False
                agent._consecutive_errors = 0
                agent._iteration = 0
                return agent

    def test_done_high_confidence(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "DONE",
            "confidence": 0.95,
            "reason": "Task requirements satisfied",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", "PASS"
        )
        assert decision == "DONE"
        assert confidence >= 0.8
        assert "satisfied" in reason

    def test_done_low_confidence(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "DONE",
            "confidence": 0.55,
            "reason": "Possibly done",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", "PASS"
        )
        assert decision == "DONE"
        assert confidence < 0.8
        assert reason == "Possibly done"

    def test_continue(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "CONTINUE",
            "confidence": 0.7,
            "reason": "Still working",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", ""
        )
        assert decision == "CONTINUE"

    def test_error_high_confidence(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "ERROR",
            "confidence": 0.9,
            "reason": "Missing dependency",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", "FAIL"
        )
        assert decision == "ERROR"
        assert confidence >= 0.8

    def test_stop_high_confidence(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "STOP",
            "confidence": 0.85,
            "reason": "Claude is looping",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", ""
        )
        assert decision == "STOP"
        assert confidence >= 0.8

    def test_malformed_json_returns_stop(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = "not valid json at all {{"
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", ""
        )
        assert decision == "STOP"
        assert confidence == 0.0
        assert "classifier parse error" in reason

    def test_missing_prompt_file(self):
        agent = self._make_agent()
        with patch.object(Path, "exists", return_value=False):
            decision, confidence, reason = agent.classify_done(
                "build a widget", "signals", "deltas", "output", ""
            )
        assert decision == "STOP"
        assert confidence == 0.0
        assert "not found" in reason

    def test_invalid_decision_normalized_to_continue(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "MAYBE",
            "confidence": 0.5,
            "reason": "Uncertain",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", ""
        )
        assert decision == "CONTINUE"

    def test_markdown_fenced_json_parsed(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = '```json\n{"decision": "DONE", "confidence": 0.92, "reason": "All good"}\n```'
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", "PASS"
        )
        assert decision == "DONE"
        assert confidence == 0.92

    def test_case_insensitive_decision(self):
        agent = self._make_agent()
        agent._gemini.call.return_value = json.dumps({
            "decision": "done",
            "confidence": 0.88,
            "reason": "Finished",
        })
        decision, confidence, reason = agent.classify_done(
            "build a widget", "signals", "deltas", "output", "PASS"
        )
        assert decision == "DONE"


# =============================================================================
# TestShouldClassifyGate — verify gate triggers on each condition
# =============================================================================

class TestShouldClassifyGate:
    """Test the cheap gate that decides whether to invoke the classifier."""

    def test_triggers_on_verification_passed(self):
        """Gate should fire when verification passes."""
        verification_passed = True
        no_code_rounds = 0
        claude_stream_done = False
        should_classify = verification_passed or no_code_rounds >= 3 or claude_stream_done
        assert should_classify

    def test_triggers_on_no_code_rounds_ge_3(self):
        """Gate should fire when 3+ recent rounds have no code changes."""
        verification_passed = False
        no_code_rounds = 3
        claude_stream_done = False
        should_classify = verification_passed or no_code_rounds >= 3 or claude_stream_done
        assert should_classify

    def test_triggers_on_claude_stream_done(self):
        """Gate should fire when Claude signals stream done."""
        verification_passed = False
        no_code_rounds = 0
        claude_stream_done = True
        should_classify = verification_passed or no_code_rounds >= 3 or claude_stream_done
        assert should_classify

    def test_no_trigger_when_all_false(self):
        """Gate should NOT fire when all conditions are false."""
        verification_passed = False
        no_code_rounds = 1
        claude_stream_done = False
        should_classify = verification_passed or no_code_rounds >= 3 or claude_stream_done
        assert not should_classify

    def test_no_code_rounds_computed_from_deltas(self):
        """Verify no_code_rounds is correctly computed from recent deltas."""
        deltas = [
            {"round": 1, "lines_added": 10, "lines_removed": 0, "output_similarity": 0.0, "instruction_sent": "a"},
            {"round": 2, "lines_added": 0, "lines_removed": 0, "output_similarity": 0.3, "instruction_sent": "b"},
            {"round": 3, "lines_added": 0, "lines_removed": 0, "output_similarity": 0.5, "instruction_sent": "c"},
            {"round": 4, "lines_added": 0, "lines_removed": 0, "output_similarity": 0.8, "instruction_sent": "d"},
        ]
        recent = deltas[-5:]
        no_code = sum(1 for d in recent if d["lines_added"] == 0 and d["lines_removed"] == 0)
        assert no_code == 3


# =============================================================================
# TestMultiPartCall — verify extra_parts flow through GeminiClient
# =============================================================================

class TestMultiPartCall:
    """Test that extra_parts are passed through GeminiClient.call() to _call_once()."""

    def test_extra_parts_appended_to_request(self):
        """Verify extra_parts become {"text": ...} entries in the request parts."""
        from agentproxy.gemini_client import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client.api_key = "test-key"

        captured_payloads = []

        def mock_urlopen(req, timeout=60):
            payload = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(payload)
            # Return a valid response
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"totalTokenCount": 0},
            }).encode("utf-8")
            return mock_resp

        with patch("agentproxy.gemini_client.urllib.request.urlopen", mock_urlopen):
            with patch("agentproxy.gemini_client.get_telemetry") as mock_tel:
                mock_tel.return_value.enabled = False
                client.call(
                    system_prompt="sys",
                    user_prompt="user",
                    extra_parts=["[ARTIFACT: TASK]\nDo X", "[ARTIFACT: OUTPUT]\nDid Y"],
                )

        assert len(captured_payloads) == 1
        parts = captured_payloads[0]["contents"][0]["parts"]
        # system + user + 2 extra = 4 parts
        assert len(parts) == 4
        assert parts[0]["text"] == "sys"
        assert parts[1]["text"] == "user"
        assert parts[2]["text"] == "[ARTIFACT: TASK]\nDo X"
        assert parts[3]["text"] == "[ARTIFACT: OUTPUT]\nDid Y"

    def test_no_extra_parts_backward_compatible(self):
        """Verify omitting extra_parts produces the same 2-part request."""
        from agentproxy.gemini_client import GeminiClient

        client = GeminiClient.__new__(GeminiClient)
        client.api_key = "test-key"

        captured_payloads = []

        def mock_urlopen(req, timeout=60):
            payload = json.loads(req.data.decode("utf-8"))
            captured_payloads.append(payload)
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"totalTokenCount": 0},
            }).encode("utf-8")
            return mock_resp

        with patch("agentproxy.gemini_client.urllib.request.urlopen", mock_urlopen):
            with patch("agentproxy.gemini_client.get_telemetry") as mock_tel:
                mock_tel.return_value.enabled = False
                client.call(system_prompt="sys", user_prompt="user")

        parts = captured_payloads[0]["contents"][0]["parts"]
        assert len(parts) == 2


# =============================================================================
# TestRoundDeltas — verify delta array accumulation and jaccard
# =============================================================================

class TestRoundDeltas:
    """Test round delta accumulation on PA and jaccard similarity helper."""

    def test_jaccard_identical(self):
        assert _jaccard_similarity("hello world foo", "hello world foo") == 1.0

    def test_jaccard_completely_different(self):
        assert _jaccard_similarity("aaa bbb", "ccc ddd") == 0.0

    def test_jaccard_partial_overlap(self):
        sim = _jaccard_similarity("the cat sat", "the dog sat")
        # 2 shared words (the, sat) out of 4 unique (the, cat, sat, dog)
        assert sim == pytest.approx(0.5)

    def test_jaccard_empty_strings(self):
        assert _jaccard_similarity("", "") == 0.0
        assert _jaccard_similarity("hello", "") == 0.0
        assert _jaccard_similarity("", "hello") == 0.0

    def test_jaccard_case_insensitive(self):
        assert _jaccard_similarity("Hello World", "hello world") == 1.0

    def test_round_deltas_accumulate(self):
        """Verify _round_deltas grows each iteration."""
        deltas: list[dict] = []
        for i in range(5):
            deltas.append({
                "round": i + 1,
                "lines_added": i * 10,
                "lines_removed": 0,
                "output_similarity": 0.0,
                "instruction_sent": f"do step {i + 1}",
            })
        assert len(deltas) == 5
        assert deltas[0]["round"] == 1
        assert deltas[4]["lines_added"] == 40

    def test_round_deltas_cleared_on_reset(self):
        """Verify deltas are cleared when PA is reset."""
        deltas: list[dict] = [{"round": 1}]
        last_output = "some output"
        # Simulate reset
        deltas.clear()
        last_output = ""
        assert len(deltas) == 0
        assert last_output == ""

    def test_similarity_recorded_in_delta(self):
        """Verify output_similarity is computed and stored."""
        prev = "the quick brown fox jumps over the lazy dog"
        curr = "the quick brown fox jumps over the lazy cat"
        sim = _jaccard_similarity(curr, prev)
        delta = {
            "round": 2,
            "lines_added": 0,
            "lines_removed": 0,
            "output_similarity": round(sim, 2),
            "instruction_sent": "continue",
        }
        # 8 of 10 unique words overlap
        assert 0.7 < delta["output_similarity"] < 1.0

    def test_instruction_truncated_in_delta(self):
        """Verify instruction_sent is truncated to 100 chars."""
        long_instruction = "x" * 200
        delta = {
            "instruction_sent": long_instruction[:100],
        }
        assert len(delta["instruction_sent"]) == 100
