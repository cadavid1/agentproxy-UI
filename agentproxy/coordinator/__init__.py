"""
Coordinator Package
===================

Multi-worker coordination for agentproxy using Celery + Redis.
This is an optional dependency -- if celery is not installed,
agentproxy runs in single-worker mode as before.

Exports:
    Coordinator: Orchestrates task decomposition and milestone dispatch.
    is_celery_available: Returns True if celery and redis are importable.
"""

from .coordinator import Coordinator


def is_celery_available() -> bool:
    """Check whether Celery and Redis packages are importable.

    Returns True only if both ``celery`` and ``redis`` can be imported.
    Does NOT check whether a Redis server is actually reachable -- that
    is deferred to connection time.
    """
    try:
        import celery  # noqa: F401
        import redis  # noqa: F401
        return True
    except ImportError:
        return False


__all__ = ["Coordinator", "is_celery_available"]
