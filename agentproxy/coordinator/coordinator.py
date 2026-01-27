"""
Coordinator
===========

Orchestrates multi-worker task execution by decomposing a task into
milestones and dispatching each one sequentially to a Celery worker.

The generator-based interface is preserved: ``run_task_multi_worker()``
yields ``OutputEvent`` objects exactly like the single-worker path.
"""

import re
import time
from typing import TYPE_CHECKING, Any, Dict, Generator, List

from ..models import EventType, OutputEvent
from ..telemetry import get_telemetry
from .models import MilestoneResult, deserialize_output_event

if TYPE_CHECKING:
    from ..pa import PA


# Default poll interval (seconds) when waiting for a Celery result
_POLL_INTERVAL = 2.0

# Maximum time to wait for a single milestone (seconds)
_MILESTONE_TIMEOUT = 1800  # 30 minutes


class Coordinator:
    """Orchestrate task decomposition and sequential milestone dispatch.

    Args:
        pa: The parent PA instance (used for task breakdown and context).
        queue: Celery queue name for milestone dispatch.
    """

    def __init__(self, pa: "PA", queue: str = "default"):
        self.pa = pa
        self.queue = queue

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_task_multi_worker(
        self, task: str, max_iterations: int = 100
    ) -> Generator[OutputEvent, None, None]:
        """Decompose *task* into milestones, dispatch each, yield events.

        This is the multi-worker counterpart to ``PA._run_task_single_worker``.
        Milestones are dispatched sequentially (one at a time) to avoid
        conflict resolution complexity.
        """
        telemetry = get_telemetry()

        # 1. Generate task breakdown via PA agent
        yield self._emit(
            "[Coordinator] Decomposing task into milestones...",
            EventType.THINKING,
        )
        breakdown_text = self.pa.agent.generate_task_breakdown(task)
        milestones = self._parse_milestones(breakdown_text)

        if not milestones:
            yield self._emit(
                "[Coordinator] No milestones extracted -- falling back to single milestone",
                EventType.THINKING,
            )
            milestones = [task]

        yield self._emit(
            f"[Coordinator] {len(milestones)} milestone(s) planned",
            EventType.TEXT,
        )

        # 2. Sequential dispatch
        context: Dict[str, Any] = {
            "prior_summary": "",
            "prior_files_changed": [],
        }

        for i, milestone in enumerate(milestones):
            yield self._emit(
                f"[Coordinator] Dispatching milestone {i + 1}/{len(milestones)}: {milestone[:120]}",
                EventType.TEXT,
            )

            # Record telemetry
            if telemetry.enabled:
                telemetry.milestones_dispatched.add(1)

            # Dispatch to Celery
            from .tasks import run_milestone

            async_result = run_milestone.apply_async(
                args=[
                    milestone,
                    self.pa.working_dir,
                    self.pa.session_id,
                    i,
                    context,
                ],
                queue=self.queue,
            )

            # Poll and yield progress events
            yield from self._poll_result(async_result, i, len(milestones))

            # Collect result and update context
            raw = async_result.get(timeout=_MILESTONE_TIMEOUT)
            result = MilestoneResult.from_dict(raw)

            # Record telemetry
            if telemetry.enabled:
                telemetry.milestones_completed.add(1, {"status": result.status})
                telemetry.milestone_duration.record(result.duration)

            # Replay worker events into the parent generator
            for evt_dict in result.events:
                yield deserialize_output_event(evt_dict)

            yield self._emit(
                f"[Coordinator] Milestone {i + 1} {result.status} "
                f"({result.duration:.1f}s, {len(result.files_changed)} files changed)",
                EventType.TEXT,
            )

            if result.status == "error":
                yield self._emit(
                    f"[Coordinator] Milestone {i + 1} errored: {result.summary}",
                    EventType.ERROR,
                )
                # Continue to next milestone rather than aborting the whole task
                # The worker already attempted error recovery internally

            context = self._update_context(context, result)

        # 3. Final summary
        yield self._emit(
            f"[Coordinator] All {len(milestones)} milestones dispatched",
            EventType.COMPLETED,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_milestones(breakdown_text: str) -> List[str]:
        """Extract milestone strings from a markdown task breakdown.

        The breakdown returned by ``PAAgent.generate_task_breakdown`` is a
        markdown string with ``- [ ]`` checklist items.  We extract the
        text from each checklist item as a milestone prompt.
        """
        milestones: List[str] = []
        for line in breakdown_text.splitlines():
            # Match markdown checklist items: - [ ] description
            match = re.match(r"^\s*[-*]\s*\[[ x]?\]\s*(.+)$", line, re.IGNORECASE)
            if match:
                text = match.group(1).strip()
                if text:
                    milestones.append(text)
        return milestones

    def _poll_result(
        self, async_result, milestone_index: int, total: int
    ) -> Generator[OutputEvent, None, None]:
        """Poll a Celery AsyncResult, yielding progress events."""
        start = time.time()
        while not async_result.ready():
            elapsed = time.time() - start
            if elapsed > _MILESTONE_TIMEOUT:
                yield self._emit(
                    f"[Coordinator] Milestone {milestone_index + 1}/{total} timed out after {elapsed:.0f}s",
                    EventType.ERROR,
                )
                return
            yield self._emit(
                f"[Coordinator] Milestone {milestone_index + 1}/{total} running ({elapsed:.0f}s elapsed)...",
                EventType.THINKING,
            )
            time.sleep(_POLL_INTERVAL)

    @staticmethod
    def _update_context(
        context: Dict[str, Any], result: MilestoneResult
    ) -> Dict[str, Any]:
        """Accumulate context from a completed milestone for the next one."""
        prior_files = list(set(context.get("prior_files_changed", []) + result.files_changed))
        prior_summary = context.get("prior_summary", "")
        if result.summary:
            prior_summary += f"\n- {result.summary}" if prior_summary else result.summary
        return {
            "prior_summary": prior_summary,
            "prior_files_changed": prior_files,
        }

    @staticmethod
    def _emit(
        content: str,
        event_type: EventType = EventType.TEXT,
        source: str = "coordinator",
    ) -> OutputEvent:
        return OutputEvent(
            event_type=event_type, content=content, metadata={"source": source}
        )
