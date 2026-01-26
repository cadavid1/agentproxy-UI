"""
Celery Tasks
=============

Defines the ``run_milestone`` Celery task that executes a single milestone
via PA in single-worker mode.  Results are returned as serialized dicts
through the Redis result backend.
"""

import time
from typing import Any, Dict, List

from .celery_app import make_celery_app
from .models import MilestoneResult, serialize_output_event

celery_app = make_celery_app()


@celery_app.task(bind=True, name="agentproxy.run_milestone")
def run_milestone(
    self,
    milestone_prompt: str,
    working_dir: str,
    session_id: str,
    milestone_index: int,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a single milestone via PA single-worker mode.

    This task is picked up by a ``pa-worker`` process.  It creates a
    fresh PA instance pointed at *working_dir*, runs the milestone
    prompt, collects all OutputEvents, and returns a serialized
    :class:`MilestoneResult`.

    Args:
        milestone_prompt: The instruction for this milestone.
        working_dir: Filesystem path the worker should operate in.
        session_id: Parent session ID for telemetry linking.
        milestone_index: Zero-based position in the milestone sequence.
        context: Accumulated context from prior milestones (files_changed,
            summaries, etc.).

    Returns:
        A dict representation of :class:`MilestoneResult`.
    """
    start = time.time()
    events: List[Dict[str, Any]] = []
    files_changed: List[str] = []
    status = "completed"
    summary = ""

    try:
        # Import PA here to avoid circular imports at module level
        from ..pa import PA

        # Build enriched prompt with prior-milestone context
        enriched_prompt = milestone_prompt
        prior_summary = context.get("prior_summary", "")
        prior_files = context.get("prior_files_changed", [])
        if prior_summary:
            enriched_prompt = (
                f"Previous milestones completed:\n{prior_summary}\n\n"
                f"Files changed so far: {', '.join(prior_files) if prior_files else 'none'}\n\n"
                f"Current milestone:\n{milestone_prompt}"
            )

        pa = PA(working_dir=working_dir, session_id=session_id)

        for event in pa.run_task(enriched_prompt):
            events.append(serialize_output_event(event))
            if event.content:
                # Collect file-change hints from events
                pass

        # Gather files changed from the PA file tracker
        files_changed = list(set(pa._session_files_changed))
        summary = f"Milestone {milestone_index + 1} completed"

        if pa.state.name == "ERROR":
            status = "error"
            summary = f"Milestone {milestone_index + 1} finished with errors"

    except Exception as exc:
        status = "error"
        summary = f"Milestone {milestone_index + 1} failed: {exc}"

    duration = time.time() - start
    result = MilestoneResult(
        status=status,
        events=events,
        files_changed=files_changed,
        summary=summary,
        duration=duration,
        milestone_index=milestone_index,
    )
    return result.to_dict()
