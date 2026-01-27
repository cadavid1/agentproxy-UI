"""
Coordinator Models
==================

Data models for multi-worker coordination.
MilestoneResult and serialization helpers for OutputEvent transport.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..models import EventType, OutputEvent


@dataclass
class MilestoneResult:
    """Result from executing a single milestone via a Celery worker.

    Attributes:
        status: "completed" or "error"
        events: List of serialized OutputEvent dicts
        files_changed: File paths modified during this milestone
        summary: Human-readable summary of what was accomplished
        duration: Wall-clock seconds the milestone took
        milestone_index: Position in the milestone sequence
    """

    status: str  # "completed" | "error"
    events: List[Dict[str, Any]] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    summary: str = ""
    duration: float = 0.0
    milestone_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for Celery result backend (Redis JSON)."""
        return {
            "status": self.status,
            "events": self.events,
            "files_changed": self.files_changed,
            "summary": self.summary,
            "duration": self.duration,
            "milestone_index": self.milestone_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MilestoneResult":
        """Deserialize from Celery result backend."""
        return cls(
            status=data.get("status", "error"),
            events=data.get("events", []),
            files_changed=data.get("files_changed", []),
            summary=data.get("summary", ""),
            duration=data.get("duration", 0.0),
            milestone_index=data.get("milestone_index", 0),
        )


def serialize_output_event(event: OutputEvent) -> Dict[str, Any]:
    """Convert an OutputEvent to a JSON-serializable dict."""
    return {
        "event_type": event.event_type.name,
        "content": event.content,
        "timestamp": event.timestamp.isoformat(),
        "metadata": event.metadata,
    }


def deserialize_output_event(data: Dict[str, Any]) -> OutputEvent:
    """Reconstruct an OutputEvent from a serialized dict."""
    return OutputEvent(
        event_type=EventType[data["event_type"]],
        content=data.get("content", ""),
        timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
        metadata=data.get("metadata", {}),
    )
