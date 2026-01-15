"""
PA (Proxy Agent) - Main Orchestrator
=====================================

After each Claude iteration, PA thinks holistically and decides:
- CONTINUE: More work needed
- VERIFY: Run independent verification
- REVIEW: Code quality review
- DONE: Task complete
"""

import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from display import create_display
from file_tracker import FileChangeTracker
from function_executor import FunctionName, FunctionResult
from models import ControllerState, EventType, OutputEvent
from pa_agent import PAAgent
from pa_memory import PAMemory


class PADecision(str, Enum):
    """PA's decision after analyzing Claude's output."""
    CONTINUE = "continue"
    VERIFY = "verify"
    REVIEW = "review"
    DONE = "done"


class PA:
    """PA Orchestrator - runs Claude Code with PA supervision."""
    
    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
        display_mode: str = "rich",
        auto_verify: bool = True,
        auto_qa: bool = True,
        context_dir: Optional[str] = None,
    ) -> None:
        self.working_dir = working_dir
        self.auto_verify = auto_verify
        self.auto_qa = auto_qa
        
        self.agent = PAAgent(working_dir, session_id, user_mission, context_dir)
        self._display = create_display(display_mode)
        self._file_tracker = FileChangeTracker(working_dir)
        
        self._state = ControllerState.IDLE
        self._claude_output_buffer: List[str] = []
        self._session_files_changed: List[str] = []
        self._previous_summary = self.agent.load_session_summary()
    
    @property
    def memory(self) -> PAMemory:
        return self.agent.memory
    
    @property
    def session_id(self) -> str:
        return self.agent.memory.session.session_id
    
    def stop(self) -> None:
        self._state = ControllerState.IDLE
    
    def run_task(self, task: str, max_iterations: int = 100) -> Generator[OutputEvent, None, None]:
        """Execute a task with PA supervising Claude."""
        self._state = ControllerState.PROCESSING
        self._session_files_changed = []
        
        # Self-check before starting
        is_ready, status = self.agent.self_check()
        yield self._emit(f"[PA Self-Check]\n{status}", EventType.TEXT, source="pa")
        if not is_ready:
            yield self._emit("[PA] Self-check failed - cannot proceed", EventType.ERROR, source="pa")
            self._state = ControllerState.IDLE
            return
        
        yield self._emit("Starting task...", EventType.STARTED)
        yield from self._setup_task_breakdown(task)
        
        if self._previous_summary:
            yield self._emit(f"[PA] Resuming session", EventType.THINKING, source="pa-thinking")
        
        current_instruction = task
        iteration = 0
        
        for iteration in range(max_iterations):
            if self.agent.is_done:
                break
            
            if iteration > 0 and iteration % 3 == 0:
                progress = self.agent.review_task_progress()
                yield self._emit(f"[PA Tasks] {progress}", EventType.THINKING, source="pa-thinking")
            
            yield self._emit(f"[Iteration {iteration + 1}] Executing Claude...", EventType.TEXT)
            
            # Show what PA is sending to Claude BEFORE execution (full text)
            yield self._emit(current_instruction, EventType.TEXT, source="pa-to-claude")
            
            claude_output_lines = []
            for event in self._stream_claude(current_instruction):
                yield event
                if event.content:
                    claude_output_lines.append(event.content)
            
            claude_output = "\n".join(claude_output_lines)
            self._claude_output_buffer.append(claude_output)
            
            changed_files = self._file_tracker.get_changed_files()
            if changed_files:
                self._session_files_changed.extend(changed_files)
            
            task_update = self.agent.smart_update_task_status(claude_output)
            if task_update:
                yield self._emit(f"[PA Tasks] {task_update}", EventType.THINKING, source="pa-thinking")
            
            # ALWAYS verify after Claude finishes before moving on
            yield self._emit("[PA] Verifying Claude's work...", EventType.TEXT, source="pa")
            yield from self._run_auto_verification(task, changed_files or [])
            
            # Detect if Claude is confused/asking for task
            if self._claude_is_confused(claude_output):
                yield self._emit("[PA] Claude lost context - re-sending original task", EventType.THINKING, source="pa-thinking")
                current_instruction = task
                continue
            
            reasoning, result = self.agent.run_iteration(claude_output)
            
            # Show PA's thinking process
            thinking_output = f"State: {reasoning.current_state}\nProgress: {reasoning.claude_progress}\nDecision: {reasoning.decision}"
            yield self._emit(thinking_output, EventType.THINKING, source="pa-thinking")
            yield self._emit(f"[{result.name.value}] {result.output[:300]}", EventType.TOOL_RESULT, source="pa")
            
            next_instruction = self.agent.get_claude_instruction()
            current_instruction = next_instruction if next_instruction else self._synthesize_instruction(result)
        
        all_files = list(set(self._session_files_changed))
        summary = self.agent.generate_session_summary(task, self._claude_output_buffer, all_files)
        self.agent.save_session_summary(summary)
        
        if self.agent.is_done:
            yield self._emit("Task completed", EventType.COMPLETED)
        else:
            yield self._emit("Max iterations reached", EventType.ERROR)
        
        self._state = ControllerState.IDLE
    
    def _stream_claude(self, instruction: str) -> Generator[OutputEvent, None, None]:
        self._file_tracker.reset()
        try:
            process = subprocess.Popen(
                ["claude", "-p", instruction, "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.working_dir,
                bufsize=1  # Line-buffered for faster output
            )
            # Use readline() for unbuffered, real-time output
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                line = line.rstrip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self._file_tracker.process_event(data)
                    for event in self._parse_claude_event(data):
                        yield event
                except json.JSONDecodeError:
                    yield self._emit(line, EventType.RAW, source="claude")
            process.wait(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            yield self._emit("[Claude timed out]", EventType.ERROR, source="claude")
        except FileNotFoundError:
            yield self._emit("[Claude CLI not found]", EventType.ERROR, source="claude")
        except Exception as e:
            yield self._emit(f"[Error: {e}]", EventType.ERROR, source="claude")
    
    def _parse_claude_event(self, data: Dict[str, Any]) -> Generator[OutputEvent, None, None]:
        event_type = data.get("type", "")
        if event_type == "assistant":
            for item in data.get("message", {}).get("content", []):
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text", "").strip():
                        yield self._emit(item["text"][:500], EventType.TEXT, source="claude")
                    elif item.get("type") == "tool_use":
                        yield self._emit(f"{item.get('name', '?')}(...)", EventType.TOOL_CALL, source="claude")
        elif event_type == "tool_result":
            content = data.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(c) for c in content)
            if str(content).strip():
                yield self._emit(f"-> {str(content)[:200]}", EventType.TOOL_RESULT, source="claude")
        elif event_type == "result":
            yield self._emit(f"Claude finished", EventType.COMPLETED, source="claude")
        elif event_type == "error":
            yield self._emit(str(data.get("error", "Error")), EventType.ERROR, source="claude")
    
    def _setup_task_breakdown(self, task: str) -> Generator[OutputEvent, None, None]:
        existing = self.agent.load_task_breakdown()
        if not existing:
            breakdown = self.agent.generate_task_breakdown(task)
            self.agent.save_task_breakdown(breakdown)
            yield self._emit("[PA] Task breakdown created", EventType.THINKING, source="pa-thinking")
    
    def _claude_is_confused(self, claude_output: str) -> bool:
        """Detect if Claude is confused/asking for a task instead of working."""
        confused_signals = [
            "i don't see a current task",
            "what would you like help with",
            "let me know what you'd like",
            "what would you like to work on",
            "how can i help you",
            "what can i assist",
            "this appears to be the start",
            "start of our conversation",
        ]
        output_lower = claude_output.lower()
        return any(sig in output_lower for sig in confused_signals)
    
    def _run_auto_verification(self, task: str, changed_files: List[str]) -> Generator[OutputEvent, None, None]:
        verify_result = self.agent.executor._verify_product({"product_type": "auto", "port": 3000})
        yield self._emit(f"[PA Verify] {verify_result.output[:300]}", EventType.TOOL_RESULT, source="pa")
        if not verify_result.success:
            self.agent.executor._claude_queue.put({"type": "instruction", "instruction": f"Fix: {verify_result.output}", "context": "verify"})
        elif self.auto_qa:
            review = self.agent.executor._review_changes({"file_paths": changed_files, "context": task[:100]})
            if review.metadata.get("has_issues"):
                self.agent.executor._claude_queue.put({"type": "instruction", "instruction": f"Fix: {review.output}", "context": "review"})
    
    def _synthesize_instruction(self, result: FunctionResult) -> str:
        if result.name == FunctionName.VERIFY_CODE:
            return "Continue" if result.success else f"Fix: {result.output[:200]}"
        elif result.name == FunctionName.RUN_TESTS:
            return "Continue" if result.success else f"Fix tests: {result.output[:200]}"
        return "Continue with the task."
    
    def _emit(self, content: str, event_type: EventType = EventType.TEXT, source: str = "pa") -> OutputEvent:
        return OutputEvent(event_type=event_type, content=content, metadata={"source": source})


def create_pa(working_dir: str = ".", session_id: Optional[str] = None, user_mission: Optional[str] = None, **kwargs: Any) -> PA:
    return PA(working_dir=working_dir, session_id=session_id, user_mission=user_mission, **kwargs)


def list_sessions() -> List[Dict[str, Any]]:
    return PAMemory.list_sessions(Path(__file__).parent / "sessions")
