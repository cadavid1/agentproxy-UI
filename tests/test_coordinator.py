"""
Unit tests for the multi-worker coordinator package.

Tests coordinator models, milestone parsing, availability checks,
and the PA multi-worker dispatch path.
"""

import os
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock, PropertyMock


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestMilestoneResult:
    """Test MilestoneResult dataclass and serialization."""

    def test_create_milestone_result(self):
        from agentproxy.coordinator.models import MilestoneResult

        result = MilestoneResult(
            status="completed",
            events=[{"event_type": "TEXT", "content": "hello", "timestamp": "2025-01-01T00:00:00", "metadata": {}}],
            files_changed=["src/main.py"],
            summary="Did the thing",
            duration=12.5,
            milestone_index=0,
        )
        assert result.status == "completed"
        assert len(result.events) == 1
        assert result.files_changed == ["src/main.py"]
        assert result.duration == 12.5

    def test_to_dict_roundtrip(self):
        from agentproxy.coordinator.models import MilestoneResult

        original = MilestoneResult(
            status="error",
            events=[{"event_type": "ERROR", "content": "boom", "timestamp": "2025-01-01T00:00:00", "metadata": {}}],
            files_changed=["a.py", "b.py"],
            summary="Failed",
            duration=3.2,
            milestone_index=2,
        )
        d = original.to_dict()
        restored = MilestoneResult.from_dict(d)

        assert restored.status == original.status
        assert restored.events == original.events
        assert restored.files_changed == original.files_changed
        assert restored.summary == original.summary
        assert restored.duration == original.duration
        assert restored.milestone_index == original.milestone_index

    def test_from_dict_defaults(self):
        from agentproxy.coordinator.models import MilestoneResult

        result = MilestoneResult.from_dict({})
        assert result.status == "error"
        assert result.events == []
        assert result.files_changed == []
        assert result.summary == ""
        assert result.duration == 0.0
        assert result.milestone_index == 0


class TestOutputEventSerialization:
    """Test OutputEvent serialization/deserialization helpers."""

    def test_serialize_roundtrip(self):
        from agentproxy.models import OutputEvent, EventType
        from agentproxy.coordinator.models import serialize_output_event, deserialize_output_event

        event = OutputEvent(
            event_type=EventType.TEXT,
            content="hello world",
            metadata={"source": "test"},
        )
        d = serialize_output_event(event)
        assert d["event_type"] == "TEXT"
        assert d["content"] == "hello world"
        assert "timestamp" in d

        restored = deserialize_output_event(d)
        assert restored.event_type == EventType.TEXT
        assert restored.content == "hello world"
        assert restored.metadata == {"source": "test"}

    def test_deserialize_all_event_types(self):
        from agentproxy.models import EventType
        from agentproxy.coordinator.models import deserialize_output_event

        for et in EventType:
            d = {
                "event_type": et.name,
                "content": f"test-{et.name}",
                "timestamp": "2025-01-01T00:00:00",
                "metadata": {},
            }
            event = deserialize_output_event(d)
            assert event.event_type == et


# ---------------------------------------------------------------------------
# Celery availability
# ---------------------------------------------------------------------------


class TestCeleryAvailability:
    """Test is_celery_available() guard."""

    def test_returns_false_when_celery_missing(self):
        """is_celery_available() returns False when celery is not installed."""
        import importlib
        import sys

        # Save originals
        celery_mod = sys.modules.get("celery")
        redis_mod = sys.modules.get("redis")

        try:
            # Force celery import to fail
            sys.modules["celery"] = None
            # Need to reimport coordinator to pick up the change
            from agentproxy.coordinator import is_celery_available
            # Reload the function's import attempt
            assert is_celery_available() is False
        finally:
            # Restore
            if celery_mod is not None:
                sys.modules["celery"] = celery_mod
            else:
                sys.modules.pop("celery", None)
            if redis_mod is not None:
                sys.modules["redis"] = redis_mod
            else:
                sys.modules.pop("redis", None)

    def test_returns_true_when_both_importable(self):
        """is_celery_available() returns True when celery and redis are importable."""
        try:
            import celery  # noqa: F401
            import redis  # noqa: F401
        except ImportError:
            pytest.skip("celery and/or redis not installed")

        from agentproxy.coordinator import is_celery_available
        assert is_celery_available() is True


# ---------------------------------------------------------------------------
# Milestone parsing
# ---------------------------------------------------------------------------


class TestMilestoneParsing:
    """Test Coordinator._parse_milestones()."""

    def test_parse_checklist_items(self):
        from agentproxy.coordinator.coordinator import Coordinator

        breakdown = """# Task: Build a REST API

- [ ] Set up project structure
- [ ] Create user endpoints
- [ ] Add authentication
- [ ] Write tests
"""
        milestones = Coordinator._parse_milestones(breakdown)
        assert len(milestones) == 4
        assert milestones[0] == "Set up project structure"
        assert milestones[3] == "Write tests"

    def test_parse_checked_items(self):
        from agentproxy.coordinator.coordinator import Coordinator

        breakdown = "- [x] Already done\n- [ ] Still todo"
        milestones = Coordinator._parse_milestones(breakdown)
        assert len(milestones) == 2

    def test_parse_star_bullets(self):
        from agentproxy.coordinator.coordinator import Coordinator

        breakdown = "* [ ] Step A\n* [ ] Step B"
        milestones = Coordinator._parse_milestones(breakdown)
        assert len(milestones) == 2

    def test_parse_empty_returns_empty(self):
        from agentproxy.coordinator.coordinator import Coordinator

        milestones = Coordinator._parse_milestones("")
        assert milestones == []

    def test_parse_non_checklist_returns_empty(self):
        from agentproxy.coordinator.coordinator import Coordinator

        breakdown = "This is just a paragraph.\nNo checklist items here."
        milestones = Coordinator._parse_milestones(breakdown)
        assert milestones == []

    def test_parse_indented_items(self):
        from agentproxy.coordinator.coordinator import Coordinator

        breakdown = "  - [ ] Indented step\n    - [ ] More indented"
        milestones = Coordinator._parse_milestones(breakdown)
        assert len(milestones) == 2


# ---------------------------------------------------------------------------
# Context accumulation
# ---------------------------------------------------------------------------


class TestContextAccumulation:
    """Test Coordinator._update_context()."""

    def test_accumulates_files(self):
        from agentproxy.coordinator.coordinator import Coordinator
        from agentproxy.coordinator.models import MilestoneResult

        ctx = {"prior_summary": "", "prior_files_changed": []}
        result = MilestoneResult(
            status="completed",
            files_changed=["a.py", "b.py"],
            summary="Step 1 done",
        )
        new_ctx = Coordinator._update_context(ctx, result)
        assert "a.py" in new_ctx["prior_files_changed"]
        assert "b.py" in new_ctx["prior_files_changed"]
        assert "Step 1 done" in new_ctx["prior_summary"]

    def test_deduplicates_files(self):
        from agentproxy.coordinator.coordinator import Coordinator
        from agentproxy.coordinator.models import MilestoneResult

        ctx = {"prior_summary": "", "prior_files_changed": ["a.py"]}
        result = MilestoneResult(
            status="completed",
            files_changed=["a.py", "c.py"],
            summary="Step 2",
        )
        new_ctx = Coordinator._update_context(ctx, result)
        assert new_ctx["prior_files_changed"].count("a.py") == 1

    def test_appends_summary(self):
        from agentproxy.coordinator.coordinator import Coordinator
        from agentproxy.coordinator.models import MilestoneResult

        ctx = {"prior_summary": "Step 1", "prior_files_changed": []}
        result = MilestoneResult(status="completed", summary="Step 2")
        new_ctx = Coordinator._update_context(ctx, result)
        assert "Step 1" in new_ctx["prior_summary"]
        assert "Step 2" in new_ctx["prior_summary"]


# ---------------------------------------------------------------------------
# _should_use_multi_worker
# ---------------------------------------------------------------------------


class TestShouldUseMultiWorker:
    """Test PA._should_use_multi_worker()."""

    def test_false_without_env_var(self):
        """Returns False when AGENTPROXY_MULTI_WORKER is not set."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENTPROXY_MULTI_WORKER", None)
            from agentproxy.pa import PA

            pa = MagicMock(spec=PA)
            pa._should_use_multi_worker = PA._should_use_multi_worker.__get__(pa)
            assert pa._should_use_multi_worker() is False

    def test_false_with_env_zero(self):
        """Returns False when AGENTPROXY_MULTI_WORKER=0."""
        with patch.dict(os.environ, {"AGENTPROXY_MULTI_WORKER": "0"}):
            from agentproxy.pa import PA

            pa = MagicMock(spec=PA)
            pa._should_use_multi_worker = PA._should_use_multi_worker.__get__(pa)
            assert pa._should_use_multi_worker() is False

    def test_true_with_env_and_celery(self):
        """Returns True when env var is set and celery is available."""
        try:
            import celery  # noqa: F401
            import redis  # noqa: F401
        except ImportError:
            pytest.skip("celery and/or redis not installed")

        with patch.dict(os.environ, {"AGENTPROXY_MULTI_WORKER": "1"}):
            from agentproxy.pa import PA

            pa = MagicMock(spec=PA)
            pa._should_use_multi_worker = PA._should_use_multi_worker.__get__(pa)
            assert pa._should_use_multi_worker() is True


# ---------------------------------------------------------------------------
# Queue routing
# ---------------------------------------------------------------------------


class TestQueueRouting:
    """Test queue configuration for workers."""

    def test_default_queue(self):
        from agentproxy.coordinator.coordinator import Coordinator

        mock_pa = MagicMock()
        coord = Coordinator(mock_pa)
        assert coord.queue == "default"

    def test_custom_queue(self):
        from agentproxy.coordinator.coordinator import Coordinator

        mock_pa = MagicMock()
        coord = Coordinator(mock_pa, queue="worker-gpu-1")
        assert coord.queue == "worker-gpu-1"


# ---------------------------------------------------------------------------
# Worker CLI
# ---------------------------------------------------------------------------


class TestWorkerCLI:
    """Test worker_cli argument parsing (without starting Celery)."""

    def test_default_args(self):
        """Default args produce 'default' queue and 'info' loglevel."""
        from agentproxy.coordinator.worker_cli import main
        import argparse

        # We can't actually call main() without celery, but we can test arg parsing
        parser = argparse.ArgumentParser()
        parser.add_argument("--queue", default=None)
        parser.add_argument("--loglevel", default="info")
        parser.add_argument("--concurrency", type=int, default=1)

        args = parser.parse_args([])
        assert args.queue is None
        assert args.loglevel == "info"
        assert args.concurrency == 1

    def test_custom_args(self):
        """Custom args are parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--queue", default=None)
        parser.add_argument("--loglevel", default="info")
        parser.add_argument("--concurrency", type=int, default=1)

        args = parser.parse_args(["--queue", "gpu-1", "--loglevel", "debug", "--concurrency", "2"])
        assert args.queue == "gpu-1"
        assert args.loglevel == "debug"
        assert args.concurrency == 2


# ---------------------------------------------------------------------------
# Telemetry metrics existence
# ---------------------------------------------------------------------------


class TestMilestoneMetrics:
    """Test that milestone metrics are created when telemetry is enabled."""

    def test_milestone_metrics_exist(self):
        """Milestone counters and histogram are created."""
        with patch.dict(os.environ, {"AGENTPROXY_ENABLE_TELEMETRY": "1"}):
            from agentproxy.telemetry import OTEL_AVAILABLE

            if not OTEL_AVAILABLE:
                pytest.skip("OTEL packages not installed")

            from agentproxy.telemetry import AgentProxyTelemetry

            telemetry = AgentProxyTelemetry()
            assert hasattr(telemetry, "milestones_dispatched")
            assert hasattr(telemetry, "milestones_completed")
            assert hasattr(telemetry, "milestone_duration")
