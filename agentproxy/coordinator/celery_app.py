"""
Celery App Factory
==================

Creates and configures a Celery application for agentproxy multi-worker
coordination. Uses Redis as both broker and result backend.

Usage:
    from agentproxy.coordinator.celery_app import make_celery_app
    app = make_celery_app()
"""

import os


def make_celery_app(
    broker_url: str = None,
    result_backend: str = None,
):
    """Create a configured Celery app for agentproxy task dispatch.

    Args:
        broker_url: Redis URL for the Celery broker.
            Defaults to AGENTPROXY_BROKER_URL env var or redis://localhost:6379/0.
        result_backend: Redis URL for storing task results.
            Defaults to AGENTPROXY_RESULT_BACKEND env var or redis://localhost:6379/1.

    Returns:
        A configured Celery application instance.
    """
    from celery import Celery

    broker_url = broker_url or os.getenv(
        "AGENTPROXY_BROKER_URL", "redis://localhost:6379/0"
    )
    result_backend = result_backend or os.getenv(
        "AGENTPROXY_RESULT_BACKEND", "redis://localhost:6379/1"
    )

    app = Celery(
        "agentproxy",
        broker=broker_url,
        backend=result_backend,
    )

    app.conf.update(
        # One task at a time per worker (sequential milestone execution)
        worker_concurrency=1,
        # Acknowledge after task completes (not on receive) for reliability
        task_acks_late=True,
        # Re-queue task if worker dies mid-execution
        task_reject_on_worker_lost=True,
        # Results expire after 1 hour
        result_expires=3600,
        # Serialize as JSON for cross-language compatibility
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Route all agentproxy tasks to configurable queue
        task_routes={
            "agentproxy.run_milestone": {
                "queue": os.getenv("AGENTPROXY_TASK_QUEUE", "default"),
            },
        },
    )

    # Auto-discover tasks in this package
    app.autodiscover_tasks(["agentproxy.coordinator"])

    return app
