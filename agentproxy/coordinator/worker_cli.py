"""
Worker CLI
==========

Entry point for the ``pa-worker`` command.
Starts a Celery worker that listens for agentproxy milestone tasks.

Usage:
    pa-worker                     # listen on 'default' queue
    pa-worker --queue gpu-1       # listen on 'default' AND 'gpu-1' queues
    pa-worker --loglevel debug    # verbose logging
"""

import argparse
import sys


def main(argv=None):
    """Start a Celery worker for agentproxy tasks."""
    parser = argparse.ArgumentParser(
        description="Start an agentproxy Celery worker",
    )
    parser.add_argument(
        "--queue",
        default=None,
        help=(
            "Additional named queue to listen on (worker always listens on "
            "'default'). Example: --queue worker-gpu-1"
        ),
    )
    parser.add_argument(
        "--loglevel",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Celery worker log level (default: info)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent worker processes (default: 1)",
    )

    args = parser.parse_args(argv)

    # Build queue list: always include 'default', optionally add named queue
    queues = ["default"]
    if args.queue and args.queue not in queues:
        queues.append(args.queue)
    queue_str = ",".join(queues)

    try:
        from .celery_app import make_celery_app
    except ImportError:
        print(
            "Error: celery and redis packages are required.\n"
            "Install with: pip install 'agentproxy[worker]'",
            file=sys.stderr,
        )
        sys.exit(1)

    app = make_celery_app()

    # Import tasks so they are registered with the app
    from . import tasks  # noqa: F401

    print(f"Starting pa-worker on queue(s): {queue_str}")
    app.worker_main([
        "worker",
        f"--loglevel={args.loglevel}",
        "-Q", queue_str,
        "-c", str(args.concurrency),
    ])


if __name__ == "__main__":
    main()
