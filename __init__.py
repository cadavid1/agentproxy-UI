"""
Interactive Labs - CCP (Code Custodian Persona)
===============================================

Unified AI agent that supervises Claude Code with:
- CCP-Thinking: Reasoning about Claude's actions
- CCP-QA: Generating and running tests
- CCP-Verify: Executing and validating code

Three-tier memory system:
1. Best Practices: Static rules from prompts/*.md
2. Session Context: High-level mission, persists hours/days
3. Interaction History: Rolling window of Claude interactions

Usage:
    from interactive_labs import CCP
    
    ccp = CCP(working_dir=".", user_mission="Build a REST API")
    for event in ccp.run_task("Create user endpoints"):
        print(event)
"""

from models import OutputEvent, EventType, ControllerState
from ccp import CCP, create_ccp, list_sessions
from ccp_memory import CCPMemory, BestPractices, SessionContext, InteractionHistory
from process_manager import ClaudeProcessManager
from display import RealtimeDisplay

__all__ = [
    # Primary API
    "CCP",
    "create_ccp",
    "list_sessions",
    # Memory system
    "CCPMemory",
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
