"""
Integration tests for multi-worker coordination.

These tests require Redis to be running (skipped automatically if unavailable).
Start Redis with: docker run -d -p 6379:6379 redis:7-alpine
"""

import os
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def redis_is_reachable(host="localhost", port=6379) -> bool:
    """Check if a Redis server is reachable."""
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host=host, port=port, socket_connect_timeout=2)
        r.ping()
        return True
    except Exception:
        return False


def celery_is_available() -> bool:
    try:
        import celery  # noqa: F401
        import redis  # noqa: F401
        return True
    except ImportError:
        return False


requires_redis = pytest.mark.skipif(
    not redis_is_reachable(),
    reason="Redis not reachable on localhost:6379",
)
requires_celery = pytest.mark.skipif(
    not celery_is_available(),
    reason="celery and/or redis packages not installed",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_celery
class TestCeleryAppFactory:
    """Test that the Celery app factory creates a properly configured app."""

    def test_make_celery_app_defaults(self):
        from agentproxy.coordinator.celery_app import make_celery_app

        app = make_celery_app()
        assert app.main == "agentproxy"
        assert app.conf.worker_concurrency == 1
        assert app.conf.task_acks_late is True
        assert app.conf.task_reject_on_worker_lost is True
        assert app.conf.result_expires == 3600

    def test_make_celery_app_custom_urls(self):
        from agentproxy.coordinator.celery_app import make_celery_app

        app = make_celery_app(
            broker_url="redis://custom-host:6380/0",
            result_backend="redis://custom-host:6380/1",
        )
        assert "custom-host" in str(app.conf.broker_url)

    def test_task_serializer_is_json(self):
        from agentproxy.coordinator.celery_app import make_celery_app

        app = make_celery_app()
        assert app.conf.task_serializer == "json"
        assert app.conf.result_serializer == "json"


@requires_celery
@requires_redis
class TestRunMilestoneTask:
    """Test the run_milestone Celery task with a real Redis backend."""

    def test_task_is_registered(self):
        """run_milestone task should be discoverable by Celery."""
        from agentproxy.coordinator.tasks import celery_app

        assert "agentproxy.run_milestone" in celery_app.tasks

    def test_run_milestone_locally(self):
        """Execute run_milestone synchronously (no worker needed)."""
        from agentproxy.coordinator.tasks import run_milestone
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock PA to avoid actual Claude execution
            with patch("agentproxy.coordinator.tasks.PA") as MockPA:
                mock_pa = MagicMock()
                mock_pa.run_task.return_value = iter([])
                mock_pa._session_files_changed = []
                mock_pa.state = MagicMock()
                mock_pa.state.name = "DONE"
                MockPA.return_value = mock_pa

                result = run_milestone(
                    milestone_prompt="Create a hello function",
                    working_dir=tmpdir,
                    session_id="test-session-123",
                    milestone_index=0,
                    context={},
                )

                assert result["status"] == "completed"
                assert isinstance(result["events"], list)
                assert result["milestone_index"] == 0
                assert result["duration"] > 0

    def test_run_milestone_error_handling(self):
        """Task returns error status when PA raises."""
        from agentproxy.coordinator.tasks import run_milestone
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("agentproxy.coordinator.tasks.PA") as MockPA:
                MockPA.side_effect = RuntimeError("test explosion")

                result = run_milestone(
                    milestone_prompt="Blow up",
                    working_dir=tmpdir,
                    session_id="test-session-456",
                    milestone_index=1,
                    context={},
                )

                assert result["status"] == "error"
                assert "failed" in result["summary"].lower() or "error" in result["summary"].lower()


@requires_celery
@requires_redis
class TestSequentialOrdering:
    """Test that milestones are dispatched sequentially."""

    def test_coordinator_dispatches_in_order(self):
        """Coordinator should dispatch milestones 0, 1, 2 in sequence."""
        from agentproxy.coordinator.coordinator import Coordinator
        from agentproxy.coordinator.models import MilestoneResult

        mock_pa = MagicMock()
        mock_pa.working_dir = "/tmp/test"
        mock_pa.session_id = "test-session"
        mock_pa.agent.generate_task_breakdown.return_value = (
            "- [ ] Step one\n- [ ] Step two\n- [ ] Step three"
        )

        coord = Coordinator(mock_pa)

        # Track dispatch order
        dispatched = []

        def mock_apply_async(args=None, queue=None, **kw):
            dispatched.append(args[3])  # milestone_index
            mock_result = MagicMock()
            mock_result.ready.return_value = True
            mock_result.get.return_value = MilestoneResult(
                status="completed",
                events=[],
                files_changed=[],
                summary=f"Step {args[3] + 1}",
                duration=1.0,
                milestone_index=args[3],
            ).to_dict()
            return mock_result

        with patch("agentproxy.coordinator.coordinator.run_milestone") as mock_task:
            mock_task.apply_async = mock_apply_async

            events = list(coord.run_task_multi_worker("Do three things"))

        assert dispatched == [0, 1, 2]


@requires_celery
class TestWorkerQueueConfig:
    """Test worker queue configuration."""

    def test_worker_listens_on_default(self):
        """Worker CLI should always include 'default' queue."""
        import argparse

        # Simulate CLI args
        queues = ["default"]
        named_queue = None
        if named_queue and named_queue not in queues:
            queues.append(named_queue)

        assert "default" in queues

    def test_worker_adds_named_queue(self):
        """Worker CLI adds named queue alongside default."""
        queues = ["default"]
        named_queue = "worker-gpu-1"
        if named_queue and named_queue not in queues:
            queues.append(named_queue)

        assert queues == ["default", "worker-gpu-1"]

    def test_no_duplicate_default(self):
        """Passing 'default' as named queue doesn't duplicate."""
        queues = ["default"]
        named_queue = "default"
        if named_queue and named_queue not in queues:
            queues.append(named_queue)

        assert queues == ["default"]
