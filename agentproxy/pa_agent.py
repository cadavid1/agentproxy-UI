"""
PA Agent Core
=============

The core PAAgent class that runs the reasoning loop.
Observes Claude's output, reasons about progress, and decides next actions.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from .gemini_client import GeminiClient
from .telemetry import get_telemetry

# Load .env from project root (go up from agentproxy/ to project root)
# Override=True ensures .env values take precedence over shell environment
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    # Try current working directory as fallback
    load_dotenv(override=True)
from .function_executor import (
    FunctionExecutor,
    FunctionCall,
    FunctionResult,
    FunctionName,
    FUNCTION_DECLARATIONS,
)
from .models import PAReasoning, AgentLoopOutput
from .pa_memory import PAMemory


class PAAgent:
    """
    PA Agent that runs an independent reasoning loop alongside Claude.
    
    Each loop iteration:
        1. Receives context + recent Claude output + available functions
        2. Produces reasoning about current state
        3. Decides which function to call
        4. Function executes and result feeds back
    """
    
    def _build_system_prompt(self) -> str:
        """Assemble system prompt from resource files.

        Loads pa_agent_loop.md, pa_done_detection.md, and pa_stall_prevention.md
        from the prompts directory, then injects the dynamic functions description.
        """
        parts = []
        prompts_dir = Path(__file__).parent / "prompts"

        for name in ["pa_agent_loop.md", "pa_done_detection.md",
                      "pa_stall_prevention.md"]:
            path = prompts_dir / name
            if path.exists():
                parts.append(path.read_text(encoding="utf-8"))

        # Inject available functions (dynamic)
        functions_desc = self._build_functions_description()
        combined = "\n\n".join(parts)
        return combined.replace("{functions}", functions_desc)

    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
        context_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize the PA Agent.
        
        Args:
            working_dir: Working directory for operations.
            session_id: Optional session ID for persistence.
            user_mission: High-level mission description.
            context_dir: Directory containing project context files.
        """
        self.working_dir = working_dir
        self.user_mission = user_mission
        self.context_dir = context_dir
        
        # Initialize components
        self._gemini = self._init_gemini()
        self._memory = self._init_memory(session_id, user_mission)
        self._executor = FunctionExecutor(working_dir, memory=self._memory)
        
        # Load project context from context_dir
        self._project_context, self._context_images = self._load_context_dir()
        
        # State
        self._history: List[Dict[str, Any]] = []
        self._is_done = False
        self._consecutive_errors = 0  # Track consecutive Gemini errors
        self._iteration = 0
    
    # =========================================================================
    # Initialization
    # =========================================================================
    
    def _init_gemini(self) -> Optional[GeminiClient]:
        """Initialize Gemini client."""
        try:
            return GeminiClient()
        except ValueError:
            return None
    
    def self_check(self) -> Tuple[bool, str]:
        """
        Perform self-check to verify PA is ready to operate.
        
        Returns:
            Tuple of (is_ready, status_message)
        """
        checks = []
        all_ok = True
        
        # Check 1: Gemini API
        if self._gemini:
            try:
                response = self._gemini.call(
                    system_prompt="Respond with only: OK",
                    user_prompt="Health check",
                    max_tokens=10
                )
                if "OK" in response.upper() or len(response) < 50:
                    checks.append("✓ Gemini API: Online")
                else:
                    checks.append("⚠ Gemini API: Unexpected response")
            except Exception as e:
                checks.append(f"✗ Gemini API: {str(e)[:50]}")
                all_ok = False
        else:
            checks.append("✗ Gemini API: Not initialized (check GEMINI_API_KEY)")
            all_ok = False
        
        # Check 2: Working directory
        if Path(self.working_dir).exists():
            checks.append(f"✓ Working dir: {self.working_dir}")
        else:
            checks.append(f"✗ Working dir: {self.working_dir} not found")
            all_ok = False
        
        # Check 3: Memory system
        if self._memory:
            checks.append(f"✓ Memory: Session {self._memory.session.session_id[:8]}")
        else:
            checks.append("✗ Memory: Not initialized")
            all_ok = False
        
        status = "\n".join(checks)
        return all_ok, status
    
    def _init_memory(
        self,
        session_id: Optional[str],
        user_mission: Optional[str],
    ) -> PAMemory:
        """Initialize memory system."""
        sessions_dir = Path(self.working_dir) / ".pa_sessions"
        prompts_dir = Path(__file__).parent / "prompts"
        
        memory = PAMemory(
            working_dir=self.working_dir,
            session_id=session_id,
            prompts_dir=str(prompts_dir),
            sessions_dir=str(sessions_dir),
        )
        
        if user_mission:
            memory.session.set_mission(user_mission)
        
        return memory
    
    def _load_context_dir(self) -> Tuple[str, Dict[str, str]]:
        """
        Load context files from the configured context directory.
        
        Returns:
            Tuple of (text_content, image_dict) where image_dict maps 
            filename -> full_path.
        """
        if not self.context_dir:
            return "", {}
        
        context_path = Path(self.context_dir)
        if not context_path.exists():
            return "", {}
        
        context_parts = []
        context_images: Dict[str, str] = {}
        
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
        
        # Load text files (top level only)
        for file_path in sorted(context_path.iterdir()):
            if not file_path.is_file():
                continue
            ext = file_path.suffix.lower()
            if ext in {".md", ".txt"}:
                try:
                    content = file_path.read_text(encoding="utf-8")
                    if content.strip():
                        context_parts.append(f"### {file_path.name}\n{content}")
                except (IOError, UnicodeDecodeError):
                    pass
        
        # Load images recursively
        for ext in image_extensions:
            for file_path in context_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    context_images[file_path.name] = str(file_path)
                    context_images[file_path.stem] = str(file_path)
                    context_images[file_path.name.lower()] = str(file_path)
                    context_images[file_path.stem.lower()] = str(file_path)
        
        # Build text content
        text_content = ""
        if context_parts:
            text_content = "## PROJECT CONTEXT\n\n" + "\n\n".join(context_parts)
        
        if context_images:
            unique_images = list(set(context_images.values()))
            image_list = "\n".join(f"  - {Path(p).name}" for p in sorted(unique_images))
            text_content += f"\n\n## REFERENCE IMAGES ({len(unique_images)} images)\n{image_list}"
        
        return text_content, context_images
    
    def _load_project_memory_folder(self) -> Tuple[str, List[str]]:
        """
        Load all contents from the project_memory folder.
        
        Returns:
            Tuple of (text_content, list_of_image_paths)
        """
        project_memory_dir = Path(self.working_dir) / "project_memory"
        text_content = ""
        image_paths: List[str] = []
        
        if not project_memory_dir.exists():
            return text_content, image_paths
        
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        
        for file_path in sorted(project_memory_dir.iterdir()):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    image_paths.append(str(file_path))
                elif ext in {".txt", ".md", ".json", ".yaml", ".yml"}:
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            text_content += f"\n### {file_path.name}\n{content}\n"
                    except (IOError, UnicodeDecodeError):
                        pass
        
        return text_content, image_paths
    
    # =========================================================================
    # Core Agent Loop
    # =========================================================================
    
    def run_iteration(self, recent_claude_output: str) -> Tuple[PAReasoning, FunctionResult]:
        """
        Run a single agent loop iteration.

        Args:
            recent_claude_output: Most recent output from Claude.

        Returns:
            Tuple of (PA reasoning, function execution result).
        """
        telemetry = get_telemetry()
        self._iteration += 1

        # Start OTEL span for PA reasoning if enabled
        if telemetry.enabled and telemetry.tracer:
            reasoning_span = telemetry.tracer.start_span(
                "pa.reasoning_loop",
                attributes={
                    "pa.iteration": self._iteration,
                    "pa.output_length": len(recent_claude_output),
                }
            )
            start_time = time.time()
        else:
            reasoning_span = None

        try:
            # Build prompt
            user_prompt = self._build_iteration_prompt(recent_claude_output)
            system_prompt = self._build_system_prompt()

            # Get Gemini response
            if not self._gemini:
                output = self._fallback_output()
            else:
                image_paths = self._collect_image_paths()
                response = self._gemini.call(system_prompt, user_prompt, image_paths or None)
                output = self._parse_agent_output(response)

            # Execute function
            result = self._executor.execute(output.function_call)

            # Record OTEL metrics
            if reasoning_span:
                duration = time.time() - start_time
                telemetry.pa_reasoning_duration.record(duration)
                telemetry.pa_decisions.add(1, {"function": output.function_call.name.value})
                reasoning_span.set_attribute("pa.decision", output.reasoning.decision)
                reasoning_span.set_attribute("pa.function", output.function_call.name.value)
                reasoning_span.end()
        except Exception as e:
            if reasoning_span:
                try:
                    from opentelemetry import trace as otel_trace
                    reasoning_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    reasoning_span.record_exception(e)
                except ImportError:
                    pass
                reasoning_span.end()
            raise
        
        # Check if done
        if result.metadata.get("done"):
            self._is_done = True
        
        # Record in history
        self._history.append({
            "iteration": self._iteration,
            "reasoning": {
                "current_state": output.reasoning.current_state,
                "claude_progress": output.reasoning.claude_progress,
                "insights": output.reasoning.insights,
                "decision": output.reasoning.decision,
            },
            "function_call": {
                "name": output.function_call.name.value,
                "arguments": output.function_call.arguments,
            },
            "result": {
                "success": result.success,
                "output": result.output[:500],
            },
        })
        
        return output.reasoning, result
    
    def _build_iteration_prompt(self, recent_claude_output: str) -> str:
        """Build the prompt for a single iteration."""
        # Load task breakdown
        task_breakdown = self.load_task_breakdown() or "No task breakdown available"
        task_progress = self.review_task_progress()
        
        # Load data room
        data_room_text, _ = self._load_project_memory_folder()
        
        # Session context
        session_context = f"""
MISSION: {self._memory.session.user_mission or 'Not specified'}
CURRENT TASK: {self._memory.session.user_prompt or 'Not specified'}
FILES TRACKED: {', '.join(self._memory.session.project_files.keys()) or 'None'}

## DATA ROOM
{data_room_text if data_room_text else '(No data room content)'}
"""
        
        return f"""
{self._project_context}

## ITERATION STATUS
Iteration: {self._iteration}

## ORIGINAL TASK (THE ONLY REQUIREMENT)
{self._memory.session.user_prompt or 'Not specified'}

## TASK BREAKDOWN (SUGGESTED APPROACH ONLY - NOT ADDITIONAL REQUIREMENTS)
{task_breakdown}

## CURRENT PROGRESS
{task_progress}

## SESSION CONTEXT
{session_context}

## BEST PRACTICES
{self._memory.best_practices.get_combined_context()[:2000]}

## HISTORY (last {min(len(self._history), 10)} items)
{json.dumps(self._history[-10:], indent=2) if self._history else 'No history yet'}

## RECENT CLAUDE OUTPUT
{recent_claude_output[:3000]}

---
REMEMBER: When verification succeeds, check if the ORIGINAL TASK is satisfied, NOT the breakdown.
Based on current progress, provide your REASONING and FUNCTION_CALL in JSON format.
"""
    
    def _build_functions_description(self) -> str:
        """Build description of available functions."""
        lines = []
        for func in FUNCTION_DECLARATIONS:
            params = json.dumps(func.parameters, indent=2)
            lines.append(f"### {func.name.value}\n{func.description}\nParameters: {params}\n")
        return "\n".join(lines)
    
    def _collect_image_paths(self) -> List[str]:
        """Collect all image paths to send to Gemini."""
        image_paths = []
        
        # Reference screenshots
        if self._memory.session.reference_screenshots:
            image_paths.extend(
                ss.path for ss in self._memory.session.reference_screenshots if ss.path
            )
        
        # Data room images
        _, data_room_images = self._load_project_memory_folder()
        if data_room_images:
            image_paths.extend(data_room_images)
        
        # Context dir images
        if self._context_images:
            image_paths.extend(set(self._context_images.values()))
        
        return image_paths
    
    def _parse_agent_output(self, response: str) -> AgentLoopOutput:
        """Parse Gemini's response into structured output."""
        # Detect Gemini error responses
        if response.startswith("[GEMINI_ERROR:"):
            error_info = self._parse_gemini_error(response)
            return self._error_output(error_info)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                reasoning = PAReasoning(
                    current_state=data.get("reasoning", {}).get("current_state", ""),
                    claude_progress=data.get("reasoning", {}).get("claude_progress", ""),
                    insights=data.get("reasoning", {}).get("insights", ""),
                    decision=data.get("reasoning", {}).get("decision", ""),
                )

                func_data = data.get("function_call", {})
                func_name = func_data.get("name", "send_to_claude")

                try:
                    name_enum = FunctionName(func_name)
                except ValueError:
                    name_enum = FunctionName.SEND_TO_CLAUDE

                function_call = FunctionCall(
                    name=name_enum,
                    arguments=func_data.get("arguments", {}),
                )

                # Reset error counter on successful parse
                self._consecutive_errors = 0

                return AgentLoopOutput(reasoning=reasoning, function_call=function_call)

        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback for unexpected parsing errors
        return self._fallback_output("Response format invalid")

    def _parse_gemini_error(self, error_string: str) -> dict:
        """
        Parse Gemini error string to extract error metadata.

        Format: [GEMINI_ERROR:error_type:status_code:message] or [GEMINI_ERROR:error_type:message]

        Returns:
            dict with keys: error_type, status_code, message
        """
        try:
            # Remove brackets and split
            content = error_string.strip("[]")
            parts = content.split(":", 3)  # Split into max 4 parts

            if len(parts) >= 3:
                # Has status code
                if parts[2].isdigit():
                    return {
                        "error_type": parts[1],
                        "status_code": int(parts[2]),
                        "message": parts[3] if len(parts) > 3 else "Unknown error",
                    }
                # No status code
                return {
                    "error_type": parts[1],
                    "status_code": None,
                    "message": ":".join(parts[2:]),
                }
        except (IndexError, ValueError):
            pass

        return {
            "error_type": "unknown",
            "status_code": None,
            "message": error_string,
        }

    def _error_output(self, error_info: dict) -> AgentLoopOutput:
        """Return output for Gemini API errors after retries exhausted."""
        error_type = error_info.get("error_type", "unknown")
        status_code = error_info.get("status_code")
        message = error_info.get("message", "Unknown error")

        # Increment consecutive error count
        self._consecutive_errors += 1

        # After 3 consecutive errors, request session save
        if self._consecutive_errors >= 3:
            session_id = self._memory.session.session_id

            return AgentLoopOutput(
                reasoning=PAReasoning(
                    current_state=f"Gemini API failure: {error_type} (3+ consecutive errors)",
                    claude_progress="Unable to verify - API unavailable",
                    insights=f"Repeated Gemini errors: {message}. Saving session for resumption.",
                    decision=f"Save session state and exit gracefully. Session ID: {session_id}",
                ),
                function_call=FunctionCall(
                    name=FunctionName.SAVE_SESSION,
                    arguments={
                        "reason": f"Gemini API failure after 3 consecutive errors: {error_type} - {message}",
                        "error_type": error_type,
                        "status_code": status_code,
                    }
                ),
            )

        # For first few errors, use NO_OP to allow Claude to continue
        # The main loop will preserve the last valid instruction
        return AgentLoopOutput(
            reasoning=PAReasoning(
                current_state=f"Gemini API error: {error_type}",
                claude_progress="Unknown (verification unavailable)",
                insights=f"Error #{self._consecutive_errors}: {message}. Allowing Claude to continue with last instruction.",
                decision=f"Monitor without changing instruction (will save session after {3 - self._consecutive_errors} more errors)",
            ),
            function_call=FunctionCall(
                name=FunctionName.NO_OP,
                arguments={
                    "reason": f"Gemini {error_type} error (attempt {self._consecutive_errors}/3) - preserving instruction context",
                    "error_type": error_type,
                    "status_code": status_code,
                }
            ),
        )

    def _fallback_output(self, reason: str = "Response format invalid") -> AgentLoopOutput:
        """Return fallback output when parsing fails - uses NO_OP to avoid overwriting queued instructions."""
        # Increment error counter for parse failures too
        self._consecutive_errors += 1

        return AgentLoopOutput(
            reasoning=PAReasoning(
                current_state="Parsing error - continuing observation",
                claude_progress="Unknown",
                insights=f"Failed to parse Gemini response: {reason}. Preserving instruction context.",
                decision=f"Continue monitoring Claude with last valid instruction (parse error #{self._consecutive_errors})",
            ),
            function_call=FunctionCall(
                name=FunctionName.NO_OP,
                arguments={"reason": f"Gemini response parsing failed: {reason} - preserving instruction context"}
            ),
        )
    
    # =========================================================================
    # Properties & Accessors
    # =========================================================================
    
    @property
    def is_done(self) -> bool:
        """Return True if task is marked complete."""
        return self._is_done
    
    @property
    def memory(self) -> PAMemory:
        """Access the memory system."""
        return self._memory
    
    @property
    def executor(self) -> FunctionExecutor:
        """Access the function executor."""
        return self._executor
    
    def get_claude_instruction(self) -> Optional[str]:
        """Get the next instruction for Claude from queued function calls."""
        pending = self._executor.get_pending_claude_instruction()
        if pending:
            return pending.get("instruction")
        return None
    
    # =========================================================================
    # Task Management
    # =========================================================================
    
    def _get_sys_dir(self) -> Path:
        """Get the context/sys directory path, creating it if needed."""
        if self.context_dir:
            sys_dir = Path(self.context_dir) / "sys"
        else:
            sys_dir = Path(self.working_dir) / "context" / "sys"
        sys_dir.mkdir(parents=True, exist_ok=True)
        return sys_dir
    
    def generate_task_breakdown(self, task: str) -> str:
        """Generate a task breakdown for the given task using Gemini."""
        if not self._gemini:
            return f"# Task: {task}\n\n- [ ] Complete the task"
        
        breakdown_prompt = f"""Break down this coding task into a SUGGESTED APPROACH.

TASK: {task}

RULES:
- Create only 2-4 suggested steps
- Each step is a suggested milestone, NOT a requirement
- Steps should be coarse suggestions, NOT detailed specifications
- DO NOT add requirements that aren't in the original task
- DO NOT specify implementation details like file names unless the task explicitly requires them

FORMAT:
## Goal
[Restate the original task in one sentence]

## Suggested Approach
- [ ] Step 1: [Suggested milestone]
- [ ] Step 2: [Suggested milestone]

## Success Criteria
[What would prove the ORIGINAL TASK is complete - nothing more, nothing less]"""
        
        try:
            return self._gemini.call(
                system_prompt="You are a senior tech lead. Create coarse, high-level task milestones.",
                user_prompt=breakdown_prompt
            )
        except Exception:
            return f"# Task: {task}\n\n- [ ] Complete the task"
    
    def save_task_breakdown(self, breakdown: str) -> str:
        """Save task breakdown to context/sys/tasks.txt."""
        sys_dir = self._get_sys_dir()
        tasks_file = sys_dir / "tasks.txt"
        tasks_file.write_text(breakdown, encoding="utf-8")
        return str(tasks_file)
    
    def load_task_breakdown(self) -> Optional[str]:
        """Load existing task breakdown if it exists."""
        sys_dir = self._get_sys_dir()
        tasks_file = sys_dir / "tasks.txt"
        
        if tasks_file.exists():
            try:
                return tasks_file.read_text(encoding="utf-8")
            except (IOError, UnicodeDecodeError):
                pass
        return None
    
    def smart_update_task_status(self, claude_output: str) -> Optional[str]:
        """Use Gemini to detect which task steps are complete."""
        breakdown = self.load_task_breakdown()
        if not breakdown or not self._gemini:
            return None
        
        lines = breakdown.split("\n")
        unchecked = [line for line in lines if "- [ ]" in line]
        if not unchecked:
            return None
        
        prompt = f"""Based on Claude's output, which task steps appear COMPLETE?

UNCHECKED STEPS:
{chr(10).join(unchecked)}

CLAUDE'S RECENT OUTPUT:
{claude_output[:2000]}

Respond with ONLY the step numbers that are now complete (e.g., "1, 3") or "NONE"."""
        
        try:
            response = self._gemini.call(
                system_prompt="You analyze coding progress. Be conservative.",
                user_prompt=prompt
            )
            
            if "NONE" in response.upper():
                return None
            
            completed_steps = []
            updated_lines = []
            for line in lines:
                if "- [ ]" in line:
                    for num in response.replace(",", " ").split():
                        if num.isdigit() and f"Step {num}" in line:
                            line = line.replace("- [ ]", "- [x]")
                            completed_steps.append(f"Step {num}")
                            break
                updated_lines.append(line)
            
            if completed_steps:
                self.save_task_breakdown("\n".join(updated_lines))
                return f"Completed: {', '.join(completed_steps)}"
            
        except Exception:
            pass
        
        return None
    
    def review_task_progress(self) -> str:
        """Review current task progress and determine what's next."""
        breakdown = self.load_task_breakdown()
        if not breakdown:
            return "No task breakdown found."
        
        lines = breakdown.split("\n")
        completed = sum(1 for line in lines if "- [x]" in line)
        pending = sum(1 for line in lines if "- [ ]" in line)
        
        next_step = None
        for line in lines:
            if "- [ ]" in line:
                next_step = line.replace("- [ ]", "").strip()
                break
        
        progress = f"Progress: {completed}/{completed + pending} steps completed"
        if next_step:
            progress += f"\nNext: {next_step}"
        
        return progress
    
    # =========================================================================
    # Session Management
    # =========================================================================
    
    def generate_session_summary(
        self,
        task: str,
        claude_outputs: List[str],
        files_changed: List[str]
    ) -> str:
        """Generate a session summary for persistence."""
        if self._gemini:
            history_text = json.dumps(self._history[-20:], indent=2) if self._history else "No history"
            claude_text = "\n---\n".join(claude_outputs[-5:]) if claude_outputs else "No output"
            
            summary_prompt = f"""Summarize this coding session concisely.

## Original Task
{task}

## PA Decision History (last 20 iterations)
{history_text}

## Claude's Recent Output
{claude_text[:3000]}

## Files Changed
{chr(10).join(files_changed) if files_changed else 'None'}

---
Write a structured summary with sections:
1. TASK SUMMARY - What was requested
2. ACTIONS TAKEN - Key decisions and actions
3. FILES MODIFIED - What changed and why
4. CURRENT STATE - Where we left off
5. NEXT STEPS - What to do if resuming"""
            
            try:
                return self._gemini.call(
                    system_prompt="You are a technical session summarizer.",
                    user_prompt=summary_prompt
                )
            except Exception:
                pass
        
        # Fallback summary
        lines = [
            "# Session Summary",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Task",
            task,
            "",
            "## Iterations",
            f"Total: {self._iteration}",
            f"Completed: {self._is_done}",
            "",
            "## Files Changed",
        ]
        for file in files_changed:
            lines.append(f"- {file}")
        
        return "\n".join(lines)
    
    def save_session_summary(self, summary: str) -> str:
        """Save session summary to context/sys/ directory."""
        sys_dir = self._get_sys_dir()
        session_id = self._memory.session.session_id
        summary_file = sys_dir / f"session_{session_id}.txt"
        summary_file.write_text(summary, encoding="utf-8")
        return str(summary_file)
    
    def load_session_summary(self) -> Optional[str]:
        """Load previous session summary if it exists."""
        session_id = self._memory.session.session_id

        possible_paths = []
        if self.context_dir:
            possible_paths.append(Path(self.context_dir) / "sys" / f"session_{session_id}.txt")
        possible_paths.append(Path(self.working_dir) / "context" / "sys" / f"session_{session_id}.txt")

        for path in possible_paths:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8")
                except (IOError, UnicodeDecodeError):
                    pass

        return None

    # =========================================================================
    # Done Classification
    # =========================================================================

    def classify_done(
        self,
        original_task: str,
        signals_text: str,
        deltas_text: str,
        recent_output: str,
        verification_output: str,
    ) -> Tuple[str, float, str]:
        """Ask Gemini for a state transition decision.

        Passes artifacts as separate Gemini parts to avoid escaping.

        Returns:
            (decision, confidence, reason) where decision is one of
            "DONE", "CONTINUE", "ERROR", "STOP".
            On parse failure returns ("STOP", 0.0, "classifier parse error").
        """
        prompts_dir = Path(__file__).parent / "prompts"
        template_path = prompts_dir / "pa_done_classifier.md"
        if not template_path.exists():
            return "STOP", 0.0, "classifier prompt not found"

        template = template_path.read_text(encoding="utf-8")
        user_prompt = template.replace("{signals}", signals_text)
        user_prompt = user_prompt.replace("{deltas}", deltas_text)

        extra_parts = [
            f"[ARTIFACT: ORIGINAL TASK]\n{original_task}",
            f"[ARTIFACT: RECENT CLAUDE OUTPUT]\n{recent_output}",
            f"[ARTIFACT: VERIFICATION OUTPUT]\n{verification_output}",
        ]

        response = self._gemini.call(
            system_prompt="You are a task state classifier. Output only JSON.",
            user_prompt=user_prompt,
            extra_parts=extra_parts,
            temperature=0.2,
            max_tokens=256,
        )

        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            decision = str(data.get("decision", "CONTINUE")).upper()
            if decision not in ("DONE", "CONTINUE", "ERROR", "STOP"):
                decision = "CONTINUE"
            return (
                decision,
                float(data.get("confidence", 0.0)),
                str(data.get("reason", "")),
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            return "STOP", 0.0, f"classifier parse error: {response[:100]}"
