"""
Data Models for PA (Proxy Agent)
=================================

Defines all data structures used throughout the system.
Optimized for clarity and type safety.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any, List


# =============================================================================
# PA Reasoning Structures
# =============================================================================

@dataclass
class PAReasoning:
    """
    PA's reasoning output for a single loop iteration.
    
    Attributes:
        current_state: Understanding of where we are in the task.
        claude_progress: Assessment of Claude's progress.
        insights: Project-level insights and observations.
        decision: What action to take and why.
    """
    current_state: str
    claude_progress: str
    insights: str
    decision: str


# =============================================================================
# Agent Loop I/O Structures
# =============================================================================

@dataclass
class AgentLoopOutput:
    """
    Output from a single PA agent loop iteration.
    
    Attributes:
        reasoning: PA's reasoning for this iteration.
        function_call: The function PA decided to execute.
    """
    reasoning: PAReasoning
    function_call: Any  # FunctionCall (imported separately to avoid circular import)


# =============================================================================
# Event Types
# =============================================================================


class EventType(Enum):
    """Types of events emitted by Claude CLI."""
    
    # Core output types
    TEXT = auto()           # Regular text response
    THINKING = auto()       # Extended thinking content
    
    # Tool-related events
    TOOL_CALL = auto()      # Claude is calling a tool
    TOOL_RESULT = auto()    # Result from tool execution
    
    # Interactive events
    PROMPT = auto()         # Waiting for user input
    CONFIRMATION = auto()   # Asking y/n confirmation
    
    # Status events
    STARTED = auto()        # Process started
    COMPLETED = auto()      # Task completed
    ERROR = auto()          # Error occurred
    
    # System events
    RAW = auto()            # Raw unparsed output


class ControllerState(Enum):
    """States of the Claude Code Controller."""
    
    IDLE = auto()           # Not running
    STARTING = auto()       # Process starting up
    READY = auto()          # Waiting for input
    PROCESSING = auto()     # Executing a task
    WAITING_CONFIRM = auto() # Waiting for user confirmation
    STOPPING = auto()       # Shutting down
    ERROR = auto()          # Error state


@dataclass
class OutputEvent:
    """
    Represents a single event from Claude CLI output.
    
    Attributes:
        event_type: The type of event
        content: Main content/message of the event
        timestamp: When the event occurred
        metadata: Additional structured data (tool params, etc.)
        raw: Original raw output before parsing
    """
    event_type: EventType
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable representation."""
        prefix = self._get_prefix()
        return f"{prefix} {self.content[:100]}{'...' if len(self.content) > 100 else ''}"
    
    def _get_prefix(self) -> str:
        """Get display prefix based on event type."""
        prefixes = {
            EventType.TEXT: "📝",
            EventType.THINKING: "💭",
            EventType.TOOL_CALL: "🔧",
            EventType.TOOL_RESULT: "✅",
            EventType.PROMPT: "❓",
            EventType.CONFIRMATION: "⚠️",
            EventType.STARTED: "🚀",
            EventType.COMPLETED: "✨",
            EventType.ERROR: "❌",
            EventType.RAW: "📄",
        }
        return prefixes.get(self.event_type, "•")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.name,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ToolCall:
    """
    Represents a tool invocation by Claude.
    
    Attributes:
        name: Tool name (e.g., "Read", "Write", "Bash")
        parameters: Tool input parameters
        timestamp: When the tool was called
    """
    name: str
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        params_str = ", ".join(f"{k}={repr(v)[:30]}" for k, v in self.parameters.items())
        return f"{self.name}({params_str})"


@dataclass
class SessionInfo:
    """
    Information about a Claude CLI session.
    
    Attributes:
        session_id: Unique identifier for this session
        working_dir: Working directory for the session
        started_at: When the session started
        task_count: Number of tasks executed
        events: List of events in this session
    """
    session_id: str
    working_dir: str
    started_at: datetime = field(default_factory=datetime.now)
    task_count: int = 0
    events: List[OutputEvent] = field(default_factory=list)
    
    def add_event(self, event: OutputEvent) -> None:
        """Add an event to the session history."""
        self.events.append(event)
    
    def get_events_by_type(self, event_type: EventType) -> List[OutputEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]
