"""
CCP Memory System
=================

Three-tier memory system for the Code Custodian Persona:

1. Best Practices (Tier 1) - Static rules from prompts/*.md
2. Session Context (Tier 2) - High-level mission, persists hours/days
3. Interaction History (Tier 3) - Rolling window of Claude interactions
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid


# =============================================================================
# Tier 1: Best Practices (Static / Loaded from .md files)
# =============================================================================

@dataclass
class BestPractices:
    """
    Tier 1: Static rules loaded from prompts/*.md files.
    
    These define CCP's persona, coding standards, and review criteria.
    Loaded once at startup, rarely changes during runtime.
    """
    system_prompt: str = ""
    coding_standards: str = ""
    review_checklist: str = ""
    security_rules: str = ""
    qa_patterns: str = ""
    
    @classmethod
    def load(cls, prompts_dir: Path) -> "BestPractices":
        """
        Load all .md files from prompts directory.
        
        Expected files:
            - ccp_system_prompt.md (required)
            - coding_standards.md (optional)
            - review_checklist.md (optional)
            - security_rules.md (optional)
            - qa_patterns.md (optional)
        """
        prompts_dir = Path(prompts_dir)
        
        def read_file(name: str) -> str:
            path = prompts_dir / name
            if path.exists():
                return path.read_text(encoding="utf-8")
            return ""
        
        return cls(
            system_prompt=read_file("ccp_system_prompt.md"),
            coding_standards=read_file("coding_standards.md"),
            review_checklist=read_file("review_checklist.md"),
            security_rules=read_file("security_rules.md"),
            qa_patterns=read_file("qa_patterns.md"),
        )
    
    def get_combined_context(self) -> str:
        """Get combined context for LLM prompts."""
        parts = [self.system_prompt]
        
        if self.coding_standards:
            parts.append(f"\n## CODING STANDARDS\n{self.coding_standards}")
        if self.security_rules:
            parts.append(f"\n## SECURITY RULES\n{self.security_rules}")
        
        return "\n".join(parts)


# =============================================================================
# Tier 2: Session Context (Per-Session / Persisted)
# =============================================================================

@dataclass
class DebtItem:
    """Technical debt created during session."""
    description: str
    containment: str
    followup_ticket: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SessionContext:
    """
    Tier 2: High-level context persisting across hours/days.
    
    This is the "big picture" memory that keeps CCP focused on:
    - What the user ultimately wants to achieve
    - Constraints and requirements
    - Acceptance criteria for completion
    - Technical debt being tracked
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_mission: str = ""
    user_prompt: str = ""  # Original user prompt
    user_constraints: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    project_files: Dict[str, str] = field(default_factory=dict)  # filename -> summary
    technical_debt: List[DebtItem] = field(default_factory=list)
    working_dir: str = "."
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_active: str = field(default_factory=lambda: datetime.now().isoformat())
    task_count: int = 0
    
    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.now().isoformat()
    
    def set_mission(self, mission: str) -> None:
        """Set or update the user's mission."""
        self.user_mission = mission
        self.touch()
    
    def add_constraint(self, constraint: str) -> None:
        """Add a constraint."""
        if constraint not in self.user_constraints:
            self.user_constraints.append(constraint)
            self.touch()
    
    def add_acceptance_criterion(self, criterion: str) -> None:
        """Add an acceptance criterion."""
        if criterion not in self.acceptance_criteria:
            self.acceptance_criteria.append(criterion)
            self.touch()
    
    def track_file(self, filename: str, summary: str = "") -> None:
        """Track a file created/modified in this session."""
        self.project_files[filename] = summary
        self.touch()
    
    def add_debt(self, description: str, containment: str, followup: str = "") -> None:
        """Track technical debt."""
        self.technical_debt.append(DebtItem(
            description=description,
            containment=containment,
            followup_ticket=followup,
        ))
        self.touch()
    
    def increment_task(self) -> None:
        """Increment task counter."""
        self.task_count += 1
        self.touch()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert DebtItem list to dicts
        data["technical_debt"] = [asdict(d) for d in self.technical_debt]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create from dictionary."""
        # Convert debt dicts back to DebtItem
        debt_list = [DebtItem(**d) for d in data.get("technical_debt", [])]
        data["technical_debt"] = debt_list
        return cls(**data)
    
    def save(self, sessions_dir: Path) -> None:
        """Persist to {sessions_dir}/{session_id}.json"""
        sessions_dir = Path(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = sessions_dir / f"{self.session_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, sessions_dir: Path, session_id: str) -> Optional["SessionContext"]:
        """Load existing session or return None."""
        filepath = Path(sessions_dir) / f"{session_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def get_summary(self) -> str:
        """Get a summary for LLM context."""
        lines = [
            f"SESSION: {self.session_id}",
            f"MISSION: {self.user_mission or 'Not specified'}",
        ]
        
        if self.user_constraints:
            lines.append(f"CONSTRAINTS: {', '.join(self.user_constraints)}")
        
        if self.acceptance_criteria:
            lines.append(f"ACCEPTANCE CRITERIA: {', '.join(self.acceptance_criteria)}")
        
        if self.project_files:
            lines.append(f"FILES: {', '.join(self.project_files.keys())}")
        
        if self.technical_debt:
            debt_summary = "; ".join(d.description for d in self.technical_debt[-3:])
            lines.append(f"TRACKED DEBT: {debt_summary}")
        
        return "\n".join(lines)


# =============================================================================
# Tier 3: Interaction History (Rolling Window)
# =============================================================================

@dataclass
class InteractionEvent:
    """A single interaction event."""
    event_type: str  # claude_output, ccp_decision, verification, qa_review
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CCPDecision:
    """A CCP thinking decision."""
    action: str  # CONTINUE, VERIFY, DONE
    reasoning: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class InteractionHistory:
    """
    Tier 3: Rolling window of recent interactions.
    
    This provides immediate context about:
    - What Claude recently did
    - What CCP decided and why
    - Recent verification results
    - QA review outcomes
    """
    max_events: int = 100
    events: List[InteractionEvent] = field(default_factory=list)
    ccp_decisions: List[CCPDecision] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    verification_results: List[Dict[str, Any]] = field(default_factory=list)
    qa_reviews: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_event(self, event_type: str, content: str, metadata: Dict = None) -> None:
        """Add an interaction event."""
        self.events.append(InteractionEvent(
            event_type=event_type,
            content=content,
            metadata=metadata or {},
        ))
        self._trim()
    
    def add_decision(self, action: str, reasoning: str) -> None:
        """Record a CCP decision."""
        self.ccp_decisions.append(CCPDecision(action=action, reasoning=reasoning))
        self._trim()
    
    def track_file(self, filepath: str) -> None:
        """Track a created/modified file."""
        if filepath not in self.files_created:
            self.files_created.append(filepath)
    
    def add_verification(self, task: str, success: bool, output: str, analysis: str) -> None:
        """Add a verification result."""
        self.verification_results.append({
            "task": task,
            "success": success,
            "output": output,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        })
    
    def add_qa_review(self, prompt: str, result: str, issues: List[str] = None) -> None:
        """Add a QA review result."""
        self.qa_reviews.append({
            "prompt": prompt,
            "result": result,
            "issues": issues or [],
            "timestamp": datetime.now().isoformat(),
        })
    
    def _trim(self) -> None:
        """Trim to max_events."""
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
        if len(self.ccp_decisions) > self.max_events:
            self.ccp_decisions = self.ccp_decisions[-self.max_events:]
    
    def get_recent_claude_outputs(self, n: int = 10) -> List[str]:
        """Get last N Claude responses."""
        claude_events = [
            e for e in self.events
            if e.event_type == "claude_output"
        ]
        return [e.content for e in claude_events[-n:]]
    
    def get_recent_decisions(self, n: int = 5) -> List[CCPDecision]:
        """Get last N CCP decisions."""
        return self.ccp_decisions[-n:]
    
    def get_history_for_llm(self, max_chars: int = 2000) -> List[Dict]:
        """Get history formatted for LLM context."""
        result = []
        char_count = 0
        
        for event in reversed(self.events[-20:]):
            event_dict = {
                "type": event.event_type,
                "content": event.content[:200],
            }
            event_str = str(event_dict)
            
            if char_count + len(event_str) > max_chars:
                break
            
            result.insert(0, event_dict)
            char_count += len(event_str)
        
        return result
    
    def clear(self) -> None:
        """Clear all history."""
        self.events.clear()
        self.ccp_decisions.clear()
        self.files_created.clear()
        self.verification_results.clear()
        self.qa_reviews.clear()


# =============================================================================
# Unified Memory Manager
# =============================================================================

class CCPMemory:
    """
    Unified three-tier memory system for CCP.
    
    Usage:
        memory = CCPMemory(working_dir="./myproject")
        memory.session.set_mission("Build a REST API")
        memory.history.add_event("claude_output", "Created app.py")
    """
    
    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        prompts_dir: Optional[str] = None,
        sessions_dir: Optional[str] = None,
    ):
        self.working_dir = Path(working_dir).resolve()
        
        # Default directories relative to this module
        module_dir = Path(__file__).parent
        self.prompts_dir = Path(prompts_dir) if prompts_dir else module_dir / "prompts"
        self.sessions_dir = Path(sessions_dir) if sessions_dir else module_dir / "sessions"
        
        # Tier 1: Load best practices
        self.best_practices = BestPractices.load(self.prompts_dir)
        
        # Tier 2: Load or create session
        if session_id:
            loaded = SessionContext.load(self.sessions_dir, session_id)
            self.session = loaded if loaded else SessionContext(session_id=session_id)
        else:
            self.session = SessionContext()
        
        self.session.working_dir = str(self.working_dir)
        
        # Tier 3: Fresh interaction history
        self.history = InteractionHistory()
    
    def save_session(self) -> None:
        """Persist session to disk."""
        self.session.save(self.sessions_dir)
    
    def get_full_context(self) -> str:
        """
        Get combined context from all tiers for LLM prompts.
        
        Returns a string combining:
        - Tier 1: Best practices
        - Tier 2: Session summary
        - Tier 3: Recent history
        """
        parts = []
        
        # Tier 1
        if self.best_practices.system_prompt:
            parts.append("=== CCP PERSONA ===")
            parts.append(self.best_practices.system_prompt[:1000])
        
        # Tier 2
        parts.append("\n=== SESSION CONTEXT ===")
        parts.append(self.session.get_summary())
        
        # Tier 3
        recent = self.history.get_history_for_llm(max_chars=1000)
        if recent:
            parts.append("\n=== RECENT HISTORY ===")
            for event in recent[-5:]:
                parts.append(f"- {event['type']}: {event['content'][:100]}")
        
        return "\n".join(parts)
    
    @staticmethod
    def list_sessions(sessions_dir: Path) -> List[Dict[str, Any]]:
        """List all available sessions."""
        sessions_dir = Path(sessions_dir)
        if not sessions_dir.exists():
            return []
        
        sessions = []
        for f in sessions_dir.glob("*.json"):
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                    sessions.append({
                        "session_id": data.get("session_id"),
                        "mission": data.get("user_mission", "")[:50],
                        "last_active": data.get("last_active"),
                        "task_count": data.get("task_count", 0),
                    })
            except Exception:
                pass
        
        return sorted(sessions, key=lambda x: x.get("last_active", ""), reverse=True)
