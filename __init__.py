"""
Interactive Labs - PA (Proxy Agent)
===============================================

Unified AI agent that supervises Claude Code with:
- PA-Thinking: Reasoning about Claude's actions
- PA-QA: Generating and running tests
- PA-Verify: Executing and validating code

Three-tier memory system:
1. Best Practices: Static rules from prompts/*.md
2. Session Context: High-level mission, persists hours/days
3. Interaction History: Rolling window of Claude interactions

Usage:
    from interactive_labs import PA
    
    pa = PA(working_dir=".", user_mission="Build a REST API")
    for event in pa.run_task("Create user endpoints"):
        print(event)
"""

from models import OutputEvent, EventType, ControllerState
from pa import PA, create_pa, list_sessions
from pa_memory import PAMemory, BestPractices, SessionContext, InteractionHistory
from process_manager import ClaudeProcessManager
from display import RealtimeDisplay

__all__ = [
    # Primary API
    "PA",
    "create_pa",
    "list_sessions",
    # Memory system
    "PAMemory",
    "BestPractices",
    "SessionContext",
    "InteractionHistory",
    # Core components
    "ClaudeProcessManager", 
    "RealtimeDisplay",
    "OutputEvent",
    "EventType",
    "ControllerState",
]

__version__ = "0.2.0"
