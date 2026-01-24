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
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Try current working directory as fallback
    load_dotenv()
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
    
    SYSTEM_PROMPT = """You are PA (Proxy Agent), an AI agent that supervises Claude Code.

You run in a continuous loop, observing Claude's work and deciding actions.

## YOUR ROLE
- Monitor Claude's progress on the task
- Verify claims with actual execution
- Guide Claude when stuck or off-track
- Ensure quality and completeness

## EACH ITERATION
You receive:
1. System context and best practices
2. History of reasoning and Claude's steps
3. Most recent Claude output
4. Available functions you can call

You must output:
1. REASONING: Your analysis of current state
2. FUNCTION_CALL: One function to execute

## OUTPUT FORMAT (JSON)
```json
{{
  "reasoning": {{
    "current_state": "Where we are in the task...",
    "claude_progress": "What Claude has accomplished...",
    "insights": "Observations from project perspective...",
    "decision": "What I will do and why..."
  }},
  "function_call": {{
    "name": "function_name",
    "arguments": {{...}}
  }}
}}
```

## AVAILABLE FUNCTIONS
{functions}

## GUIDELINES
- You are the HUMAN'S PROXY - act on their behalf autonomously
- Be SKEPTICAL of Claude's claims - verify with actual execution
- Use SEND_TO_CLAUDE to guide Claude's next action
- Use VERIFY_CODE / RUN_TESTS before marking done
- Only MARK_DONE when ALL requirements are verified working
- NEVER request human input - YOU are the human proxy
- If Claude asks questions, answer them based on the mission/task context
- Keep pushing Claude until the task is ACTUALLY DONE and VERIFIED

## TASK MANAGEMENT
- At the START of a new task, break it down into subtasks using CREATE_TASK
- Track progress by updating task status as work progresses
- Mark tasks complete when verified done
- Use the task list to decide what to assign Claude next
"""

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
            system_prompt = self.SYSTEM_PROMPT.format(
                functions=self._build_functions_description()
            )

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
                telemetry.pa_decisions.add(1, {"decision": output.reasoning.decision})
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

## TASK BREAKDOWN
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
                
                return AgentLoopOutput(reasoning=reasoning, function_call=function_call)
                
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback
        return self._fallback_output()
    
    def _fallback_output(self) -> AgentLoopOutput:
        """Return fallback output when parsing fails - uses NO_OP to avoid overwriting queued instructions."""
        return AgentLoopOutput(
            reasoning=PAReasoning(
                current_state="Parsing error - continuing observation",
                claude_progress="Unknown",
                insights="",
                decision="Continue monitoring Claude (Gemini unavailable)",
            ),
            function_call=FunctionCall(
                name=FunctionName.NO_OP,
                arguments={"reason": "Gemini response parsing failed"}
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
        
        breakdown_prompt = f"""Break down this coding task into HIGH-LEVEL MILESTONES.

TASK: {task}

RULES FOR STEPS:
- Create only 3-5 steps maximum
- Each step must be INDEPENDENTLY EXECUTABLE by an AI coding agent
- Each step should take 1-3 Claude iterations to complete
- Steps should be coarse milestones, NOT micro-tasks

FORMAT:
## Goal
[One sentence]

## Steps
- [ ] Step 1: [High-level milestone]
- [ ] Step 2: [High-level milestone]
- [ ] Step 3: [High-level milestone]

## Done When
[Success criteria]"""
        
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
