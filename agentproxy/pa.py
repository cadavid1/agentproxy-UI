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
import socket
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .display import create_display
from .file_tracker import FileChangeTracker
from .function_executor import FunctionName, FunctionResult
from .event_processors import process_tool_event
from .models import ControllerState, EventType, OutputEvent
from .pa_agent import PAAgent
from .pa_memory import PAMemory
from .telemetry import get_telemetry


class PADecision(str, Enum):
    """PA's decision after analyzing Claude's output."""
    CONTINUE = "continue"
    VERIFY = "verify"
    REVIEW = "review"
    DONE = "done"


class PA:
    """
    PA Orchestrator - runs Claude Code with PA supervision.

    State Machine (enables supervisor/worker composition):
        IDLE ------> PROCESSING ------> DONE
          ^              |                |
          |              v                |
          |            ERROR              |
          |              |                |
          |              v                |
          +----------STOPPED <------------+
                       |
                    reset()

    Valid transitions:
        - IDLE → PROCESSING (via run_task())
        - PROCESSING → DONE (task completes successfully)
        - PROCESSING → ERROR (task fails or max iterations)
        - PROCESSING → STOPPED (user calls stop())
        - {DONE, ERROR, STOPPED} → IDLE (via reset())

    Terminal states (DONE, ERROR, STOPPED) persist until explicit reset(),
    enabling external observers (e.g., supervisor PA) to check completion status.
    """

    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
        display_mode: str = "rich",
        auto_verify: bool = True,
        auto_qa: bool = True,
        context_dir: Optional[str] = None,
        claude_bin: Optional[str] = None,
    ) -> None:
        self.working_dir = working_dir
        self.auto_verify = auto_verify
        self.auto_qa = auto_qa
        self.claude_bin = claude_bin or "claude"

        self.agent = PAAgent(working_dir, session_id, user_mission, context_dir)
        self._display = create_display(display_mode)
        self._file_tracker = FileChangeTracker(working_dir)

        self._state = ControllerState.IDLE
        self._claude_output_buffer: List[str] = []
        self._session_files_changed: List[str] = []
        self._previous_summary = self.agent.load_session_summary()

        # Track the original task and last valid instruction for error recovery
        self._original_task: str = ""
        self._last_valid_instruction: str = ""
    
    @property
    def memory(self) -> PAMemory:
        return self.agent.memory

    @property
    def session_id(self) -> str:
        return self.agent.memory.session.session_id

    @property
    def state(self) -> ControllerState:
        """Current execution state. Terminal states persist until reset()."""
        return self._state

    def stop(self) -> None:
        """Stop the current task execution."""
        self._state = ControllerState.STOPPED

    def reset(self) -> None:
        """
        Reset PA to IDLE state for reuse.

        State Machine:
            IDLE → PROCESSING (via run_task)
            PROCESSING → DONE | ERROR | STOPPED (task completes/fails/interrupted)
            {DONE, ERROR, STOPPED} → IDLE (via reset)

        This enables PA reuse in supervisor/worker hierarchies where a parent
        PA may spawn child PA instances and check their completion status.
        """
        self._state = ControllerState.IDLE
        self._claude_output_buffer = []
        self._session_files_changed = []

    def _get_subprocess_env_with_trace_context(self) -> Dict[str, str]:
        """
        Get environment for Claude subprocess with session-specific OTEL metadata.

        OTEL config (endpoints, protocols, exporters) is inherited from parent env.
        We only add session-specific resource attributes for linking PA ↔ Claude.
        """
        import os
        telemetry = get_telemetry()
        env = os.environ.copy()

        # Enable Claude's telemetry and set service name
        env["CLAUDE_CODE_ENABLE_TELEMETRY"] = "1"
        env["OTEL_SERVICE_NAME"] = "claude-code"

        # Session-specific resource attributes for linking
        user_id = os.getenv("AGENTPROXY_OWNER_ID", os.getenv("USER", "unknown"))
        project_id = os.getenv("AGENTPROXY_PROJECT_ID", "default")
        namespace = os.getenv("OTEL_SERVICE_NAMESPACE", f"{user_id}.{project_id}")

        resource_attrs = [
            f"service.name=claude-code",
            f"service.namespace={namespace}",
            f"host.name={os.getenv('HOSTNAME', socket.gethostname())}",
            f"agentproxy.owner={user_id}",
            f"agentproxy.project_id={project_id}",
            f"agentproxy.role=worker",
            f"agentproxy.master_session_id={self.session_id}",
        ]
        env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)

        telemetry.log(f"Claude env: master_session_id={self.session_id[:8]}...")

        return env

    def _ensure_git_repo(self) -> None:
        """
        Ensure working directory is a git repo with a baseline commit.

        This enables reliable LOC tracking via `git diff --numstat HEAD`
        even when the user's -d target isn't already a git repo.
        Future: supports git worktree for parallel workers.
        """
        import os
        git_dir = os.path.join(self.working_dir, ".git")
        if os.path.exists(git_dir):
            return  # Already a git repo

        telemetry = get_telemetry()

        try:
            os.makedirs(self.working_dir, exist_ok=True)
            subprocess.run(
                ["git", "init"],
                cwd=self.working_dir, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.working_dir, capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "commit", "--allow-empty", "-m", "baseline"],
                cwd=self.working_dir, capture_output=True, timeout=5,
            )
            if telemetry.enabled:
                telemetry.log("Pre-work: initialized git repo for LOC tracking")
        except Exception as e:
            # Non-fatal: LOC tracking degrades gracefully to (0, 0)
            if telemetry.enabled:
                telemetry.log(f"Pre-work: git init failed ({e}), LOC tracking unavailable")

    def _process_tool_enrichments(self, event_data: Dict[str, Any]) -> None:
        """
        Run tool adapters on Claude's stream-json events for enriched telemetry.

        Extracts tool_use items from assistant messages and passes them through
        the appropriate adapter (Bash → git detection, Write → file metadata, etc.)
        """
        if event_data.get("type") != "assistant":
            return

        telemetry = get_telemetry()
        if not telemetry.enabled:
            return

        message = event_data.get("message", {})
        for item in message.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "tool_use":
                continue

            tool_name = item.get("name", "")
            tool_input = item.get("input", {})
            enrichment = process_tool_event(tool_name, tool_input)

            if enrichment and enrichment.labels:
                telemetry.tool_executions.add(1, {
                    "tool_name": tool_name,
                    "success": "true",
                    **enrichment.labels,  # already validated by ToolEnrichment
                })
                if enrichment.tags:
                    telemetry.log(f"Tool enrichment: {tool_name} tags={enrichment.tags}")

    def _should_use_multi_worker(self) -> bool:
        """Check whether multi-worker dispatch should be used.

        All three conditions must be true:
        1. AGENTPROXY_MULTI_WORKER=1 env var is set
        2. celery package is importable
        3. redis package is importable
        """
        import os
        if os.getenv("AGENTPROXY_MULTI_WORKER", "0") != "1":
            return False
        try:
            from .coordinator import is_celery_available
            return is_celery_available()
        except ImportError:
            return False

    def run_task(self, task: str, max_iterations: int = 100) -> Generator[OutputEvent, None, None]:
        """Execute a task with PA supervising Claude.

        If multi-worker mode is enabled (``AGENTPROXY_MULTI_WORKER=1`` and
        Celery+Redis are available), tasks are decomposed into milestones
        and dispatched to Celery workers.  Otherwise the existing
        single-worker path is used.
        """
        if self._should_use_multi_worker():
            yield from self._run_task_multi_worker(task, max_iterations)
        else:
            yield from self._run_task_single_worker(task, max_iterations)

    def _run_task_multi_worker(self, task: str, max_iterations: int = 100) -> Generator[OutputEvent, None, None]:
        """Execute a task via Celery-based multi-worker coordination."""
        import os
        telemetry = get_telemetry()

        # Start OTEL span if enabled
        if telemetry.enabled and telemetry.tracer:
            span = telemetry.tracer.start_span(
                "pa.run_task",
                attributes={
                    "pa.session_id": self.session_id,
                    "pa.task.description": task[:100],
                    "pa.working_dir": self.working_dir,
                    "pa.max_iterations": max_iterations,
                    "pa.project_id": os.getenv("AGENTPROXY_PROJECT_ID", "default"),
                    "pa.mode": "multi_worker",
                }
            )
            telemetry.tasks_started.add(1)
            telemetry.active_sessions.add(1)
            task_start_time = time.time()
        else:
            span = None

        try:
            self._state = ControllerState.PROCESSING
            self._session_files_changed = []
            self._ensure_git_repo()

            from .coordinator import Coordinator
            queue = os.getenv("AGENTPROXY_TASK_QUEUE", "default")
            coord = Coordinator(self, queue=queue)
            yield from coord.run_task_multi_worker(task, max_iterations)

            if self._state == ControllerState.PROCESSING:
                self._state = ControllerState.DONE

            status = "completed" if self._state == ControllerState.DONE else "error"

            if span:
                duration = time.time() - task_start_time
                telemetry.tasks_completed.add(1, {"status": status})
                telemetry.task_duration.record(duration)
                span.set_attribute("pa.status", status)
                span.end()
                telemetry.active_sessions.add(-1)

                from .telemetry import flush_telemetry
                flush_telemetry()

        except Exception as e:
            if span:
                telemetry.active_sessions.add(-1)
                try:
                    from opentelemetry import trace as otel_trace
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                except ImportError:
                    pass
                span.end()
            raise

    def _run_task_single_worker(self, task: str, max_iterations: int = 100) -> Generator[OutputEvent, None, None]:
        """Execute a task in single-worker mode (original run_task logic)."""
        telemetry = get_telemetry()

        # Start OTEL span if enabled
        if telemetry.enabled and telemetry.tracer:
            import os
            span = telemetry.tracer.start_span(
                "pa.run_task",
                attributes={
                    "pa.session_id": self.session_id,
                    "pa.task.description": task[:100],  # Truncate for readability
                    "pa.working_dir": self.working_dir,
                    "pa.max_iterations": max_iterations,
                    "pa.project_id": os.getenv("AGENTPROXY_PROJECT_ID", "default"),
                }
            )
            telemetry.log(f"Started span 'pa.run_task' (session={self.session_id[:8]})")
            telemetry.tasks_started.add(1)
            telemetry.log(f"Metric: tasks_started +1")
            telemetry.active_sessions.add(1)
            telemetry.log(f"Metric: active_sessions +1")
            task_start_time = time.time()
        else:
            span = None

        try:
            self._state = ControllerState.PROCESSING
            self._session_files_changed = []

            # Pre-work: ensure git repo for LOC tracking
            self._ensure_git_repo()

            # Store original task for error recovery
            self._original_task = task
            self._last_valid_instruction = task

            # Self-check before starting
            is_ready, status = self.agent.self_check()
            yield self._emit(f"[PA Self-Check]\n{status}", EventType.TEXT, source="pa")
            if not is_ready:
                yield self._emit("[PA] Self-check failed - cannot proceed", EventType.ERROR, source="pa")
                self._state = ControllerState.ERROR
                if span:
                    span.set_attribute("pa.status", "failed_self_check")
                    span.end()
                    telemetry.active_sessions.add(-1)
                return

            yield self._emit("Starting task...", EventType.STARTED)
            yield from self._setup_task_breakdown(task)

            if self._previous_summary:
                yield self._emit(f"[PA] Resuming session", EventType.THINKING, source="pa-thinking")

            current_instruction = task
            iteration = 0

            for iteration in range(max_iterations):
                if self.agent.is_done or self._state == ControllerState.DONE:
                    break

                if iteration > 0 and iteration % 3 == 0:
                    progress = self.agent.review_task_progress()
                    yield self._emit(f"[PA Tasks] {progress}", EventType.THINKING, source="pa-thinking")

                yield self._emit(f"[Iteration {iteration + 1}] Executing Claude...", EventType.TEXT)

                # Show what PA is sending to Claude BEFORE execution (full text)
                yield self._emit(current_instruction, EventType.TEXT, source="pa-to-claude")

                claude_output_lines = []
                for event in self._stream_claude(current_instruction, iteration=iteration):
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
                verification_result = None
                for event in self._run_auto_verification(task, changed_files or []):
                    if event.event_type == EventType.TOOL_RESULT and event.metadata.get("source") == "pa":
                        verification_result = event.content
                    yield event

                # Detect if Claude is confused/asking for task
                if self._claude_is_confused(claude_output):
                    yield self._emit("[PA] Claude lost context - re-sending original task", EventType.THINKING, source="pa-thinking")
                    current_instruction = task
                    continue

                # Include verification results in PA's reasoning context
                context_with_verification = claude_output
                if verification_result:
                    context_with_verification += f"\n\n[VERIFICATION RESULT]\n{verification_result}"

                reasoning, result = self.agent.run_iteration(context_with_verification)

                # Show PA's thinking process
                thinking_output = f"State: {reasoning.current_state}\nProgress: {reasoning.claude_progress}\nDecision: {reasoning.decision}"
                yield self._emit(thinking_output, EventType.THINKING, source="pa-thinking")
                yield self._emit(f"[{result.name.value}] {result.output[:300]}", EventType.TOOL_RESULT, source="pa")

                # Check if task is marked done
                if result.metadata.get("done"):
                    self._state = ControllerState.DONE
                    break

                # Check if we should exit (session save or other terminal state)
                if result.metadata.get("exit_gracefully"):
                    # Save session with error context for resumption
                    yield self._emit(
                        f"[PA] Session saved due to errors. Resume with session_id: {self.session_id}",
                        EventType.TEXT,
                        source="pa"
                    )
                    break

                # Get next instruction (prioritize queued, then synthesized)
                next_instruction = self.agent.get_claude_instruction()
                if next_instruction:
                    current_instruction = next_instruction
                    self._last_valid_instruction = next_instruction
                else:
                    synthesized = self._synthesize_instruction(result)
                    # Empty string means "no instruction change" - keep previous instruction
                    # This prevents sending vague "Continue" commands during errors
                    if synthesized:
                        current_instruction = synthesized
                        self._last_valid_instruction = synthesized
                    else:
                        # During error states (NO_OP), keep the last valid instruction
                        # This ensures Claude maintains context even when PA can't reason
                        current_instruction = self._last_valid_instruction

            all_files = list(set(self._session_files_changed))
            summary = self.agent.generate_session_summary(task, self._claude_output_buffer, all_files)
            self.agent.save_session_summary(summary)

            if self.agent.is_done:
                yield self._emit("Task completed", EventType.COMPLETED)
                status = "completed"
                # State already set to DONE when MARK_DONE was called
            else:
                yield self._emit("Max iterations reached", EventType.ERROR)
                status = "max_iterations"
                self._state = ControllerState.ERROR

            # Record OTEL completion
            if span:
                duration = time.time() - task_start_time
                telemetry.tasks_completed.add(1, {"status": status})
                telemetry.log(f"Metric: tasks_completed +1 (status={status})")
                telemetry.task_duration.record(duration)
                telemetry.log(f"Metric: task_duration {duration:.2f}s")
                span.set_attribute("pa.status", status)
                span.set_attribute("pa.iterations", iteration + 1)
                span.end()
                telemetry.log(f"Ended span 'pa.run_task' ({iteration + 1} iterations, {duration:.2f}s)")
                telemetry.active_sessions.add(-1)
                telemetry.log(f"Metric: active_sessions -1")

                # Track code changes
                if telemetry.enabled:
                    lines_added, lines_removed = self._file_tracker.get_code_changes()
                    if lines_added > 0:
                        telemetry.code_lines_added.add(lines_added)
                        telemetry.log(f"Metric: code_lines_added +{lines_added}")
                    if lines_removed > 0:
                        telemetry.code_lines_removed.add(lines_removed)
                        telemetry.log(f"Metric: code_lines_removed +{lines_removed}")

                    files_modified = len(self._file_tracker._changed_files)
                    if files_modified > 0:
                        telemetry.code_files_modified.add(files_modified)
                        telemetry.log(f"Metric: code_files_modified +{files_modified}")

                # Flush telemetry to ensure all data is exported immediately on task completion
                # Even though we export every OTEL_TRACE_EXPORT_INTERVAL ms (default 1s),
                # the last batch might still be queued, so we force flush on completion
                from .telemetry import flush_telemetry
                flush_telemetry()

            # State persists (DONE or ERROR) - use reset() to return to IDLE

        except Exception as e:
            # Record OTEL error
            if span:
                telemetry.active_sessions.add(-1)
                # Only use trace API if OTEL is available
                try:
                    from opentelemetry import trace as otel_trace
                    span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                except ImportError:
                    pass
                span.end()
            raise
    
    def _stream_claude(self, instruction: str, iteration: int = 0) -> Generator[OutputEvent, None, None]:
        telemetry = get_telemetry()
        self._file_tracker.reset()

        # Start OTEL span for Claude subprocess if enabled
        if telemetry.enabled and telemetry.tracer:
            import os
            claude_span = telemetry.tracer.start_span(
                "claude.subprocess",
                attributes={
                    "claude.prompt_length": len(instruction),
                    "agentproxy.iteration": iteration,
                    "agentproxy.master_session_id": self.session_id,
                    "agentproxy.project_id": os.getenv("AGENTPROXY_PROJECT_ID", "default"),
                }
            )
            telemetry.log(f"Started span 'claude.subprocess' (iteration={iteration})")
            telemetry.claude_iterations.add(1)
            telemetry.log(f"Metric: claude_iterations +1")
        else:
            claude_span = None

        try:
            # Get environment with OTEL trace context and session linking
            # This enables Claude Code's built-in telemetry and links it to PA's session
            env = self._get_subprocess_env_with_trace_context()

            process = subprocess.Popen(
                [self.claude_bin, "-p", instruction, "--output-format", "stream-json", "--verbose", "--dangerously-skip-permissions"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.working_dir,
                env=env,  # Pass OTEL env vars and session linking to Claude
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
                    self._process_tool_enrichments(data)
                    for event in self._parse_claude_event(data):
                        yield event
                except json.JSONDecodeError:
                    yield self._emit(line, EventType.RAW, source="claude")
            process.wait(timeout=300)

            # Complete Claude span
            if claude_span:
                claude_span.set_attribute("claude.completed", True)
                claude_span.end()
                telemetry.log(f"Ended span 'claude.subprocess' (iteration={iteration})")

        except subprocess.TimeoutExpired:
            process.kill()
            yield self._emit("[Claude timed out]", EventType.ERROR, source="claude")
            if claude_span:
                try:
                    from opentelemetry import trace as otel_trace
                    claude_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, "Timeout"))
                except ImportError:
                    pass
                claude_span.end()
        except FileNotFoundError:
            yield self._emit("[Claude CLI not found]", EventType.ERROR, source="claude")
            if claude_span:
                try:
                    from opentelemetry import trace as otel_trace
                    claude_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, "CLI not found"))
                except ImportError:
                    pass
                claude_span.end()
        except Exception as e:
            yield self._emit(f"[Error: {e}]", EventType.ERROR, source="claude")
            if claude_span:
                try:
                    from opentelemetry import trace as otel_trace
                    claude_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    claude_span.record_exception(e)
                except ImportError:
                    pass
                claude_span.end()
    
    def _parse_claude_event(self, data: Dict[str, Any]) -> Generator[OutputEvent, None, None]:
        event_type = data.get("type", "")
        if event_type == "assistant":
            for item in data.get("message", {}).get("content", []):
                if isinstance(item, dict):
                    if item.get("type") == "text" and item.get("text", "").strip():
                        yield self._emit(item["text"][:500], EventType.TEXT, source="claude")
                    elif item.get("type") == "tool_use":
                        tool_name = item.get("name", "?")
                        tool_input = item.get("input", {})
                        # Extract key details based on tool type
                        detail = self._format_tool_detail(tool_name, tool_input)
                        yield self._emit(f"{tool_name}({detail})", EventType.TOOL_CALL, source="claude")
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
    
    def _format_tool_detail(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Extract key details from tool input for display."""
        name_lower = tool_name.lower()
        
        # File operations - show path
        if name_lower in ("read", "read_file", "readfile"):
            path = tool_input.get("file_path") or tool_input.get("path") or ""
            return path.split("/")[-1] if "/" in path else path[:50]
        
        if name_lower in ("write", "write_file", "writefile", "edit", "edit_file"):
            path = tool_input.get("file_path") or tool_input.get("path") or ""
            return path.split("/")[-1] if "/" in path else path[:50]
        
        # Glob/search - show pattern
        if name_lower in ("glob", "search", "find", "list"):
            pattern = tool_input.get("pattern") or tool_input.get("glob") or tool_input.get("query") or ""
            return pattern[:40]
        
        # Bash/command - show command
        if name_lower in ("bash", "run", "execute", "shell", "command"):
            cmd = tool_input.get("command") or tool_input.get("cmd") or ""
            return cmd[:60]
        
        # TodoWrite - show task
        if name_lower in ("todowrite", "todo"):
            todos = tool_input.get("todos") or []
            if todos and isinstance(todos, list):
                return f"{len(todos)} items"
            return ""
        
        # Default - show first string value found
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:40]
        return ""
    
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

        # Record verification metric (bypasses execute() so we record here)
        telemetry = get_telemetry()
        if telemetry.enabled:
            result_str = "pass" if verify_result.success else "fail"
            telemetry.verifications.add(1, {"type": "verify_product", "result": result_str})
            telemetry.log(f"Metric: verification ({result_str})")
        if not verify_result.success:
            self.agent.executor._claude_queue.put({"type": "instruction", "instruction": f"Fix: {verify_result.output}", "context": "verify"})
        elif self.auto_qa:
            review = self.agent.executor._review_changes({"file_paths": changed_files, "context": task[:100]})
            if review.metadata.get("has_issues"):
                self.agent.executor._claude_queue.put({"type": "instruction", "instruction": f"Fix: {review.output}", "context": "review"})
    
    def _synthesize_instruction(self, result: FunctionResult) -> str:
        """
        Synthesize an instruction to send to Claude based on function result.

        For NO_OP results (which occur during errors), we avoid sending vague
        instructions and instead remain silent to let Claude continue naturally.
        """
        if result.name == FunctionName.VERIFY_CODE:
            return "Continue" if result.success else f"Fix: {result.output[:200]}"
        elif result.name == FunctionName.RUN_TESTS:
            return "Continue" if result.success else f"Fix tests: {result.output[:200]}"
        elif result.name == FunctionName.NO_OP:
            # For NO_OP, remain silent (empty string signals no instruction change)
            # This prevents sending confusing "Continue" messages during error states
            # The main loop will keep the previous instruction instead
            return ""
        elif result.name == FunctionName.SAVE_SESSION:
            # Session save triggered - let it exit naturally
            return ""

        return "Continue with the task."
    
    def _emit(self, content: str, event_type: EventType = EventType.TEXT, source: str = "pa") -> OutputEvent:
        return OutputEvent(event_type=event_type, content=content, metadata={"source": source})


def create_pa(working_dir: str = ".", session_id: Optional[str] = None, user_mission: Optional[str] = None, **kwargs: Any) -> PA:
    return PA(working_dir=working_dir, session_id=session_id, user_mission=user_mission, **kwargs)


def list_sessions(working_dir: str = ".") -> List[Dict[str, Any]]:
    """List sessions for a specific project directory."""
    sessions_dir = Path(working_dir).resolve() / ".pa_sessions"
    return PAMemory.list_sessions(sessions_dir)
