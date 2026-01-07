"""
CCP Agent Loop Architecture.

This module implements CCP as an agent loop that runs in parallel with Claude Code.
Both are independent processes that communicate through structured messages.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        CCP Agent Loop                           │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Input (each loop):                                        │  │
    │  │   - System instruction + best practices + project memory  │  │
    │  │   - History: CCP reasonings + Claude steps                │  │
    │  │   - Most recent Claude outputs                            │  │
    │  │   - Available function declarations                       │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                            ↓                                    │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Output (each loop):                                       │  │
    │  │   - CCP reasoning (state, progress, insights)             │  │
    │  │   - Function call (to execute in parallel)                │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
                                 ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Function Execution Layer                     │
    │  Executes function call → Result feeds back to:                 │
    │    1. CCP session understanding                                 │
    │    2. Claude's next step instruction                            │
    └─────────────────────────────────────────────────────────────────┘
                                 ↕
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Claude Code Process                        │
    │  Independent process that receives instructions and outputs     │
    └─────────────────────────────────────────────────────────────────┘
"""

# Standard library imports
import json
import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

# Third-party imports
from dotenv import load_dotenv

# Local imports
from ccp_memory import CCPMemory
from display import create_display
from models import ControllerState, EventType, OutputEvent

# Load environment variables
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


# =============================================================================
# Data Structures
# =============================================================================

class FunctionName(str, Enum):
    """Available functions that CCP can call."""
    
    SEND_TO_CLAUDE = "send_to_claude"
    VERIFY_CODE = "verify_code"
    RUN_TESTS = "run_tests"
    CHECK_SERVER = "check_server"
    READ_FILE = "read_file"
    MARK_DONE = "mark_done"
    CREATE_TASK = "create_task"
    UPDATE_TASK = "update_task"
    COMPLETE_TASK = "complete_task"


@dataclass
class FunctionDeclaration:
    """
    Declaration of a function available to CCP.
    
    Attributes:
        name: Unique function identifier.
        description: What the function does.
        parameters: JSON schema for function parameters.
    """
    
    name: FunctionName
    description: str
    parameters: Dict[str, Any]


@dataclass
class FunctionCall:
    """
    A function call decided by CCP.
    
    Attributes:
        name: Which function to call.
        arguments: Arguments to pass to the function.
    """
    
    name: FunctionName
    arguments: Dict[str, Any]


@dataclass
class FunctionResult:
    """
    Result from executing a function.
    
    Attributes:
        name: Which function was called.
        success: Whether execution succeeded.
        output: The function's output or error message.
        metadata: Additional context about the execution.
    """
    
    name: FunctionName
    success: bool
    output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CCPReasoning:
    """
    CCP's reasoning output for a single loop iteration.
    
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


@dataclass
class AgentLoopInput:
    """
    Input to a single CCP agent loop iteration.
    
    Attributes:
        system_instruction: Core CCP system prompt.
        best_practices: Loaded best practices from prompts/.
        project_memory: Session and project context.
        history: Past CCP reasonings and Claude steps.
        recent_claude_output: Most recent output from Claude.
        available_functions: Functions CCP can call.
    """
    
    system_instruction: str
    best_practices: str
    project_memory: str
    history: List[Dict[str, Any]]
    recent_claude_output: str
    available_functions: List[FunctionDeclaration]


@dataclass
class AgentLoopOutput:
    """
    Output from a single CCP agent loop iteration.
    
    Attributes:
        reasoning: CCP's reasoning for this iteration.
        function_call: The function CCP decided to execute.
    """
    
    reasoning: CCPReasoning
    function_call: FunctionCall


# =============================================================================
# Gemini Client (for CCP reasoning)
# =============================================================================

class GeminiClient:
    """Client for Gemini API that powers CCP's reasoning."""
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found")
    
    def call(self, system_prompt: str, user_prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Make API call to Gemini with optional images.
        
        Args:
            system_prompt: System instruction for the model.
            user_prompt: User prompt/question.
            image_paths: Optional list of image file paths to include.
        """
        import urllib.request
        import urllib.error
        import base64
        
        url = f"{self.API_URL}?key={self.api_key}"
        
        # Build parts: text first, then images
        parts = [
            {"text": system_prompt},
            {"text": user_prompt},
        ]
        
        # Add images as inline_data
        if image_paths:
            for img_path in image_paths:
                try:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    
                    # Determine MIME type from extension
                    ext = img_path.lower().split(".")[-1]
                    mime_map = {
                        "png": "image/png",
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "gif": "image/gif",
                        "webp": "image/webp",
                        "bmp": "image/bmp",
                    }
                    mime_type = mime_map.get(ext, "image/png")
                    
                    parts.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_data
                        }
                    })
                except (IOError, OSError):
                    pass  # Skip unreadable images
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
            }
        }
        
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method='POST'
            )
            
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Error: {str(e)[:100]}]"


# =============================================================================
# Function Definitions
# =============================================================================

# All functions available to CCP
FUNCTION_DECLARATIONS: List[FunctionDeclaration] = [
    FunctionDeclaration(
        name=FunctionName.SEND_TO_CLAUDE,
        description="Send an instruction or follow-up to Claude Code. Use this to guide Claude's next action.",
        parameters={
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "The instruction to send to Claude"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about why this instruction"
                }
            },
            "required": ["instruction"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.VERIFY_CODE,
        description="Execute Python files to verify they work correctly.",
        parameters={
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of Python file paths to execute"
                }
            },
            "required": ["file_paths"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.RUN_TESTS,
        description="Run test files or pytest to verify functionality.",
        parameters={
            "type": "object",
            "properties": {
                "test_command": {
                    "type": "string",
                    "description": "Test command to run (e.g., 'pytest', 'python -m unittest')"
                }
            },
            "required": ["test_command"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.CHECK_SERVER,
        description="Check if a web server is running at a URL.",
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to check (e.g., http://localhost:5000)"
                }
            },
            "required": ["url"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.READ_FILE,
        description="Read contents of a file to understand what was created.",
        parameters={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                }
            },
            "required": ["file_path"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.MARK_DONE,
        description="Mark the task as complete. Only use when ALL requirements are verified.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary of what was accomplished"
                },
                "verified_items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of items that were verified working"
                }
            },
            "required": ["summary"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.CREATE_TASK,
        description="Break down work into a subtask and add it to the task list.",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What needs to be done"
                },
                "assignee": {
                    "type": "string",
                    "enum": ["claude", "ccp"],
                    "description": "Who should do this task"
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority (0=highest)"
                }
            },
            "required": ["description"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.UPDATE_TASK,
        description="Update a task's status (in_progress, blocked, etc).",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to update"
                },
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"],
                    "description": "New status"
                },
                "notes": {
                    "type": "string",
                    "description": "Optional notes about the update"
                }
            },
            "required": ["task_id", "status"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.COMPLETE_TASK,
        description="Mark a specific task as completed.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "Task ID to mark complete"
                },
                "notes": {
                    "type": "string",
                    "description": "Completion notes"
                }
            },
            "required": ["task_id"]
        }
    ),
]


# =============================================================================
# Function Executor
# =============================================================================

class FunctionExecutor:
    """
    Executes functions called by CCP in parallel with Claude.
    
    Each function runs independently and returns a FunctionResult.
    """
    
    def __init__(self, working_dir: str = ".", memory: Any = None) -> None:
        self.working_dir = working_dir
        self._claude_queue: Queue = Queue()
        self._memory = memory  # Reference to CCPMemory for task management
    
    def execute(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call and return the result.
        
        Args:
            call: The function call to execute.
            
        Returns:
            FunctionResult with success status and output.
        """
        handlers: Dict[FunctionName, Callable] = {
            FunctionName.SEND_TO_CLAUDE: self._send_to_claude,
            FunctionName.VERIFY_CODE: self._verify_code,
            FunctionName.RUN_TESTS: self._run_tests,
            FunctionName.CHECK_SERVER: self._check_server,
            FunctionName.READ_FILE: self._read_file,
            FunctionName.MARK_DONE: self._mark_done,
            FunctionName.CREATE_TASK: self._create_task,
            FunctionName.UPDATE_TASK: self._update_task,
            FunctionName.COMPLETE_TASK: self._complete_task,
        }
        
        handler = handlers.get(call.name)
        if not handler:
            return FunctionResult(
                name=call.name,
                success=False,
                output=f"Unknown function: {call.name}"
            )
        
        try:
            return handler(call.arguments)
        except Exception as e:
            return FunctionResult(
                name=call.name,
                success=False,
                output=f"Execution error: {str(e)[:200]}"
            )
    
    def _send_to_claude(self, args: Dict[str, Any]) -> FunctionResult:
        """Queue instruction for Claude."""
        instruction = args.get("instruction", "")
        context = args.get("context", "")
        
        self._claude_queue.put({
            "type": "instruction",
            "instruction": instruction,
            "context": context,
        })
        
        return FunctionResult(
            name=FunctionName.SEND_TO_CLAUDE,
            success=True,
            output=f"Queued instruction for Claude: {instruction}",
            metadata={"instruction": instruction}
        )
    
    def _verify_code(self, args: Dict[str, Any]) -> FunctionResult:
        """Execute Python files to verify they work."""
        file_paths = args.get("file_paths", [])
        results = []
        
        for filepath in file_paths:
            if not os.path.isabs(filepath):
                filepath = os.path.join(self.working_dir, filepath)
            
            if not os.path.exists(filepath):
                results.append(f"{filepath}: FILE NOT FOUND")
                continue
            
            try:
                result = subprocess.run(
                    ["python3", filepath],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.working_dir
                )
                
                if result.returncode == 0:
                    output = result.stdout[:500] if result.stdout else "OK"
                    results.append(f"{filepath}: SUCCESS\n{output}")
                else:
                    error = result.stderr[:500] if result.stderr else "Unknown error"
                    results.append(f"{filepath}: FAILED (code {result.returncode})\n{error}")
                    
            except subprocess.TimeoutExpired:
                results.append(f"{filepath}: TIMEOUT")
            except Exception as e:
                results.append(f"{filepath}: ERROR - {str(e)[:100]}")
        
        all_success = all("SUCCESS" in r or "OK" in r for r in results)
        
        return FunctionResult(
            name=FunctionName.VERIFY_CODE,
            success=all_success,
            output="\n---\n".join(results),
            metadata={"files_tested": file_paths}
        )
    
    def _run_tests(self, args: Dict[str, Any]) -> FunctionResult:
        """Run test command."""
        test_command = args.get("test_command", "pytest")
        
        try:
            result = subprocess.run(
                test_command.split(),
                capture_output=True,
                text=True,
                timeout=120,
                cwd=self.working_dir
            )
            
            output = result.stdout + result.stderr
            success = result.returncode == 0
            
            return FunctionResult(
                name=FunctionName.RUN_TESTS,
                success=success,
                output=output[:2000],
                metadata={"return_code": result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            return FunctionResult(
                name=FunctionName.RUN_TESTS,
                success=False,
                output="Test execution timed out"
            )
        except Exception as e:
            return FunctionResult(
                name=FunctionName.RUN_TESTS,
                success=False,
                output=f"Test error: {str(e)[:200]}"
            )
    
    def _check_server(self, args: Dict[str, Any]) -> FunctionResult:
        """Check if server is running."""
        import urllib.request
        import urllib.error
        
        url = args.get("url", "http://localhost:5000")
        
        try:
            req = urllib.request.Request(url, method='GET')
            req.add_header('User-Agent', 'CCP-Agent/1.0')
            
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
                content_length = len(response.read())
                
                return FunctionResult(
                    name=FunctionName.CHECK_SERVER,
                    success=True,
                    output=f"Server UP at {url} - Status {status}, {content_length} bytes",
                    metadata={"status": status, "url": url}
                )
                
        except urllib.error.HTTPError as e:
            return FunctionResult(
                name=FunctionName.CHECK_SERVER,
                success=True,  # Server responded, even if error
                output=f"Server responded with {e.code}: {e.reason}",
                metadata={"status": e.code, "url": url}
            )
        except Exception as e:
            return FunctionResult(
                name=FunctionName.CHECK_SERVER,
                success=False,
                output=f"Server DOWN at {url}: {str(e)[:100]}",
                metadata={"url": url}
            )
    
    def _read_file(self, args: Dict[str, Any]) -> FunctionResult:
        """Read file contents."""
        file_path = args.get("file_path", "")
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.working_dir, file_path)
        
        if not os.path.exists(file_path):
            return FunctionResult(
                name=FunctionName.READ_FILE,
                success=False,
                output=f"File not found: {file_path}"
            )
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Truncate if too long
            if len(content) > 5000:
                content = content[:5000] + "\n... [truncated]"
            
            return FunctionResult(
                name=FunctionName.READ_FILE,
                success=True,
                output=content,
                metadata={"file_path": file_path, "size": len(content)}
            )
            
        except Exception as e:
            return FunctionResult(
                name=FunctionName.READ_FILE,
                success=False,
                output=f"Read error: {str(e)[:100]}"
            )
    
    def _mark_done(self, args: Dict[str, Any]) -> FunctionResult:
        """Mark task as complete."""
        summary = args.get("summary", "Task completed")
        verified_items = args.get("verified_items", [])
        
        return FunctionResult(
            name=FunctionName.MARK_DONE,
            success=True,
            output=f"TASK COMPLETE: {summary}",
            metadata={
                "summary": summary,
                "verified_items": verified_items,
                "done": True
            }
        )
    
    def _create_task(self, args: Dict[str, Any]) -> FunctionResult:
        """Create a new task in the task list."""
        if not self._memory:
            return FunctionResult(
                name=FunctionName.CREATE_TASK,
                success=False,
                output="No memory reference - cannot create task"
            )
        
        description = args.get("description", "")
        assignee = args.get("assignee", "claude")
        priority = args.get("priority", 0)
        
        task = self._memory.session.add_task(
            description=description,
            assignee=assignee,
            priority=priority
        )
        self._memory.save_session()
        
        return FunctionResult(
            name=FunctionName.CREATE_TASK,
            success=True,
            output=f"Created task [{task.id}]: {description}",
            metadata={"task_id": task.id, "task": task.to_dict()}
        )
    
    def _update_task(self, args: Dict[str, Any]) -> FunctionResult:
        """Update a task's status."""
        if not self._memory:
            return FunctionResult(
                name=FunctionName.UPDATE_TASK,
                success=False,
                output="No memory reference - cannot update task"
            )
        
        task_id = args.get("task_id", "")
        status = args.get("status", "")
        notes = args.get("notes", "")
        
        success = self._memory.session.update_task_status(task_id, status, notes)
        if success:
            self._memory.save_session()
            return FunctionResult(
                name=FunctionName.UPDATE_TASK,
                success=True,
                output=f"Task [{task_id}] updated to: {status}",
                metadata={"task_id": task_id, "status": status}
            )
        else:
            return FunctionResult(
                name=FunctionName.UPDATE_TASK,
                success=False,
                output=f"Task [{task_id}] not found"
            )
    
    def _complete_task(self, args: Dict[str, Any]) -> FunctionResult:
        """Mark a task as completed."""
        if not self._memory:
            return FunctionResult(
                name=FunctionName.COMPLETE_TASK,
                success=False,
                output="No memory reference - cannot complete task"
            )
        
        task_id = args.get("task_id", "")
        notes = args.get("notes", "")
        
        success = self._memory.session.update_task_status(task_id, "completed", notes)
        if success:
            self._memory.save_session()
            return FunctionResult(
                name=FunctionName.COMPLETE_TASK,
                success=True,
                output=f"Task [{task_id}] completed",
                metadata={"task_id": task_id}
            )
        else:
            return FunctionResult(
                name=FunctionName.COMPLETE_TASK,
                success=False,
                output=f"Task [{task_id}] not found"
            )
    
    def get_pending_claude_instruction(self) -> Optional[Dict[str, Any]]:
        """Get next queued instruction for Claude, if any."""
        try:
            return self._claude_queue.get_nowait()
        except Empty:
            return None
    
    def get_pending_human_request(self) -> Optional[str]:
        """Deprecated - CCP is human proxy, no human requests needed."""
        return None


# =============================================================================
# CCP Agent
# =============================================================================

class CCPAgent:
    """
    CCP Agent that runs an independent loop alongside Claude.
    
    Each loop iteration:
        1. Receives input (context + recent Claude output + functions)
        2. Produces reasoning about current state
        3. Decides which function to call
        4. Function executes in parallel
        5. Result feeds back into next iteration
    """
    
    AGENT_SYSTEM_PROMPT = """You are CCP (Code Custodian Persona), an AI agent that supervises Claude Code.

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
- NEVER request human input - YOU are the human proxy, make decisions yourself
- If Claude asks questions, answer them based on the mission/task context
- If Claude needs permissions, approve them (permissions are auto-granted)
- Keep pushing Claude until the task is ACTUALLY DONE and VERIFIED

## TASK MANAGEMENT
- At the START of a new task, break it down into subtasks using CREATE_TASK
- Each subtask should be small, specific, and assignable to Claude
- Track progress by updating task status (UPDATE_TASK) as work progresses
- Mark tasks complete (COMPLETE_TASK) when verified done
- Use the task list to decide what to assign Claude next
- Refer to pending tasks when deciding next actions
"""

    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
    ) -> None:
        """
        Initialize the CCP Agent.
        
        Args:
            working_dir: Working directory for operations.
            session_id: Optional session ID for persistence.
            user_mission: High-level mission description.
        """
        self.working_dir = working_dir
        self.user_mission = user_mission
        
        # Initialize components
        self._gemini = self._init_gemini()
        self._memory = self._init_memory(session_id, user_mission)
        self._executor = FunctionExecutor(working_dir, memory=self._memory)
        self._display = create_display("rich")
        
        # State
        self._history: List[Dict[str, Any]] = []
        self._is_done = False
        self._iteration = 0
    
    def _init_gemini(self) -> Optional[GeminiClient]:
        """Initialize Gemini client."""
        try:
            return GeminiClient()
        except ValueError:
            return None
    
    def _init_memory(
        self,
        session_id: Optional[str],
        user_mission: Optional[str],
    ) -> CCPMemory:
        """Initialize memory system."""
        sessions_dir = Path(self.working_dir) / ".ccp_sessions"
        prompts_dir = Path(__file__).parent / "prompts"
        
        memory = CCPMemory(
            working_dir=self.working_dir,
            session_id=session_id,
            prompts_dir=str(prompts_dir),
            sessions_dir=str(sessions_dir),
        )
        
        # Set mission if provided
        if user_mission:
            memory.session.set_mission(user_mission)
        
        return memory
    
    def _build_functions_description(self) -> str:
        """Build description of available functions for the prompt."""
        lines = []
        for func in FUNCTION_DECLARATIONS:
            params = json.dumps(func.parameters, indent=2)
            lines.append(f"### {func.name.value}\n{func.description}\nParameters: {params}\n")
        return "\n".join(lines)
    
    def _build_loop_input(self, recent_claude_output: str) -> AgentLoopInput:
        """
        Build input for a single agent loop iteration.
        
        Args:
            recent_claude_output: Most recent output from Claude.
            
        Returns:
            AgentLoopInput with all context for this iteration.
        """
        # System instruction
        system_instruction = self.AGENT_SYSTEM_PROMPT.format(
            functions=self._build_functions_description()
        )
        
        # Best practices from Tier 1
        best_practices = self._memory.best_practices.get_combined_context()
        
        # Project memory from Tier 2
        screenshots_info = ""
        if self._memory.session.reference_screenshots:
            screenshots_info = "\nREFERENCE SCREENSHOTS (images attached below):\n" + "\n".join(
                f"  - {ss.path}: {ss.description or 'UI reference'}"
                for ss in self._memory.session.reference_screenshots
            )
        
        project_memory = f"""
MISSION: {self._memory.session.user_mission or 'Not specified'}
CURRENT TASK: {self._memory.session.user_prompt or 'Not specified'}
FILES TRACKED: {', '.join(self._memory.session.project_files.keys()) or 'None'}{screenshots_info}
"""
        
        return AgentLoopInput(
            system_instruction=system_instruction,
            best_practices=best_practices,
            project_memory=project_memory,
            history=self._history.copy(),
            recent_claude_output=recent_claude_output,
            available_functions=FUNCTION_DECLARATIONS,
        )
    
    def _parse_agent_output(self, response: str) -> AgentLoopOutput:
        """
        Parse Gemini's response into structured output.
        
        Args:
            response: Raw response from Gemini.
            
        Returns:
            AgentLoopOutput with reasoning and function call.
        """
        # Try to extract JSON from response
        try:
            # Find JSON block
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)
                
                reasoning = CCPReasoning(
                    current_state=data.get("reasoning", {}).get("current_state", ""),
                    claude_progress=data.get("reasoning", {}).get("claude_progress", ""),
                    insights=data.get("reasoning", {}).get("insights", ""),
                    decision=data.get("reasoning", {}).get("decision", ""),
                )
                
                func_data = data.get("function_call", {})
                func_name = func_data.get("name", "send_to_claude")
                
                # Map string to enum
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
        
        # Fallback: default to continuing with Claude
        return AgentLoopOutput(
            reasoning=CCPReasoning(
                current_state="Parsing error - continuing observation",
                claude_progress="Unknown",
                insights="",
                decision="Continue monitoring Claude",
            ),
            function_call=FunctionCall(
                name=FunctionName.SEND_TO_CLAUDE,
                arguments={"instruction": "Continue with the current task."}
            ),
        )
    
    def run_iteration(self, recent_claude_output: str) -> Tuple[CCPReasoning, FunctionResult]:
        """
        Run a single agent loop iteration.
        
        Args:
            recent_claude_output: Most recent output from Claude.
            
        Returns:
            Tuple of (CCP reasoning, function execution result).
        """
        self._iteration += 1
        
        # Build input
        loop_input = self._build_loop_input(recent_claude_output)
        
        # Build prompt for Gemini
        user_prompt = f"""
## PROJECT CONTEXT
{loop_input.project_memory}

## BEST PRACTICES
{loop_input.best_practices[:2000]}

## HISTORY (last {min(len(loop_input.history), 10)} items)
{json.dumps(loop_input.history[-10:], indent=2) if loop_input.history else 'No history yet'}

## RECENT CLAUDE OUTPUT
{loop_input.recent_claude_output[:3000]}

---
Based on the above, provide your REASONING and FUNCTION_CALL in JSON format.
"""
        
        # Get Gemini response
        if not self._gemini:
            # Fallback if no Gemini
            output = AgentLoopOutput(
                reasoning=CCPReasoning(
                    current_state="Gemini unavailable",
                    claude_progress="Cannot assess",
                    insights="Manual verification needed",
                    decision="Request human help",
                ),
                function_call=FunctionCall(
                    name=FunctionName.REQUEST_HUMAN,
                    arguments={"question": "Gemini unavailable - please review manually"}
                ),
            )
        else:
            # Collect screenshot paths to send as images
            image_paths = [
                ss.path for ss in self._memory.session.reference_screenshots
                if ss.path
            ] if self._memory.session.reference_screenshots else None
            
            response = self._gemini.call(loop_input.system_instruction, user_prompt, image_paths)
            output = self._parse_agent_output(response)
        
        # Execute function
        result = self._executor.execute(output.function_call)
        
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
    
    @property
    def is_done(self) -> bool:
        """Return True if task is marked complete."""
        return self._is_done
    
    def get_claude_instruction(self) -> Optional[str]:
        """
        Get the next instruction for Claude from queued function calls.
        
        Returns:
            Instruction string if available, None otherwise.
        """
        pending = self._executor.get_pending_claude_instruction()
        if pending:
            return pending.get("instruction")
        return None
    
    def get_human_request(self) -> Optional[str]:
        """Get pending human request if any."""
        return self._executor.get_pending_human_request()


# =============================================================================
# Main Orchestrator
# =============================================================================

class CCP:
    """
    CCP Agent - orchestrates Claude Code with an agent loop architecture.
    
    Both CCP and Claude run as parallel processes communicating through
    structured messages. CCP reasons about Claude's output and decides
    which function to execute next.
    """
    
    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
        display_mode: str = "rich",
        auto_verify: bool = True,
        auto_qa: bool = True,
    ) -> None:
        """
        Initialize CCP.
        
        Args:
            working_dir: Working directory for operations.
            session_id: Optional session ID for persistence.
            user_mission: High-level mission description.
            display_mode: Display mode ("rich", "simple", "json", "quiet").
            auto_verify: Whether to auto-verify code (used by agent).
            auto_qa: Whether to auto-run QA (used by agent).
        """
        self.working_dir = working_dir
        self.auto_verify = auto_verify
        self.auto_qa = auto_qa
        self.agent = CCPAgent(working_dir, session_id, user_mission)
        self._display = create_display(display_mode)
        self._claude_output_buffer: List[str] = []
        self._state = ControllerState.IDLE
    
    @property
    def memory(self) -> CCPMemory:
        """Access the memory system."""
        return self.agent._memory
    
    @property
    def session_id(self) -> str:
        """Get the current session ID."""
        return self.agent._memory.session.session_id
    
    def stop(self) -> None:
        """Stop the current task."""
        self._state = ControllerState.IDLE
    
    def run_task(
        self,
        task: str,
        max_iterations: int = 100,
    ) -> Generator[OutputEvent, None, None]:
        """
        Execute a task with CCP supervising Claude.
        
        Args:
            task: The task to execute.
            max_iterations: Maximum CCP loop iterations.
            
        Yields:
            OutputEvent objects for display.
        """
        self._state = ControllerState.PROCESSING
        
        yield self._emit_event(EventType.STARTED, f"Starting task: {task[:100]}...")
        
        # Initialize Claude with the task
        current_instruction = task
        
        for iteration in range(max_iterations):
            if self.agent.is_done:
                break
            
            # === CCP THINKING: Before sending to Claude ===
            yield self._emit_event(
                EventType.THINKING,
                f"━━━ CCP Iteration {iteration + 1}/{max_iterations} ━━━\n"
                f"📋 Current instruction to send:\n{current_instruction}",
                source="ccp-thinking"
            )
            
            # Run Claude with current instruction
            yield self._emit_event(
                EventType.TEXT,
                f"[Iteration {iteration + 1}] Executing Claude...",
                source="ccp"
            )
            
            # Stream Claude output in real-time
            claude_output_lines = []
            for event in self._stream_claude(current_instruction):
                yield event
                if event.content:
                    claude_output_lines.append(event.content)
            
            claude_output = "\n".join(claude_output_lines)
            self._claude_output_buffer.append(claude_output)
            
            # === CCP THINKING: Analyzing Claude's output ===
            yield self._emit_event(
                EventType.THINKING,
                f"🔍 Analyzing Claude's response ({len(claude_output)} chars)...\n"
                f"📊 Invoking Gemini for CCP reasoning...",
                source="ccp-thinking"
            )
            
            # Run CCP agent iteration
            reasoning, result = self.agent.run_iteration(claude_output)
            
            # === CCP THINKING: Full reasoning output ===
            yield self._emit_event(
                EventType.THINKING,
                f"━━━ CCP Reasoning ━━━\n"
                f"📍 STATE: {reasoning.current_state}\n\n"
                f"📈 PROGRESS: {reasoning.claude_progress}\n\n"
                f"💡 INSIGHTS: {reasoning.insights}\n\n"
                f"🎯 DECISION: {reasoning.decision}",
                source="ccp-thinking"
            )
            
            # === CCP THINKING: Function execution ===
            yield self._emit_event(
                EventType.THINKING,
                f"⚡ Executing function: {result.name.value}",
                source="ccp-thinking"
            )
            
            # Display function result
            yield self._emit_event(
                EventType.TOOL_RESULT,
                f"[{result.name.value}] {'✓' if result.success else '✗'}: {result.output}",
                source="ccp-function"
            )
            
            # CCP is the human proxy - handle any "human requests" autonomously
            human_request = self.agent.get_human_request()
            if human_request:
                # Don't stop - CCP handles it as human proxy
                yield self._emit_event(
                    EventType.TEXT,
                    f"[CCP as human proxy] Handling: {human_request}",
                    source="ccp"
                )
                # Continue to next iteration - CCP will respond via send_to_claude
            
            # Get next instruction for Claude
            next_instruction = self.agent.get_claude_instruction()
            if next_instruction:
                current_instruction = next_instruction
                yield self._emit_event(
                    EventType.THINKING,
                    "📤 Next instruction queued from function result",
                    source="ccp-thinking"
                )
            else:
                # If no explicit instruction, synthesize from result
                current_instruction = self._synthesize_instruction(result)
                yield self._emit_event(
                    EventType.THINKING,
                    f"🔄 Synthesized next instruction:\n{current_instruction}",
                    source="ccp-thinking"
                )
        
        # === CCP THINKING: Final status ===
        yield self._emit_event(
            EventType.THINKING,
            f"━━━ CCP Loop Complete ━━━\n"
            f"Iterations used: {iteration + 1}/{max_iterations}\n"
            f"Task done: {self.agent.is_done}",
            source="ccp-thinking"
        )
        
        # Final status
        if self.agent.is_done:
            yield self._emit_event(
                EventType.COMPLETED,
                "Task completed successfully",
                source="ccp"
            )
        else:
            yield self._emit_event(
                EventType.ERROR,
                "Task did not complete within max iterations",
                source="ccp"
            )
        
        self._state = ControllerState.IDLE
    
    def _stream_claude(self, instruction: str) -> Generator[OutputEvent, None, None]:
        """
        Stream Claude Code output in real-time as events.
        
        Args:
            instruction: Instruction to send to Claude.
            
        Yields:
            OutputEvent for each meaningful piece of output.
        """
        try:
            process = subprocess.Popen(
                [
                    "claude", "-p", instruction,
                    "--output-format", "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",  # CCP acts as human proxy - auto-approve
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.working_dir
            )
            
            for line in process.stdout:
                line = line.rstrip()
                if not line.strip():
                    continue
                
                # Try to parse as JSON
                try:
                    data = json.loads(line)
                    event_type = data.get("type", "")
                    
                    # Extract content based on event type
                    if event_type == "assistant":
                        msg = data.get("message", {})
                        for item in msg.get("content", []):
                            if isinstance(item, dict):
                                if item.get("type") == "text":
                                    text = item.get("text", "")
                                    if text.strip():
                                        yield self._emit_event(
                                            EventType.TEXT,
                                            text[:500],
                                            source="claude"
                                        )
                                elif item.get("type") == "tool_use":
                                    yield self._emit_event(
                                        EventType.TOOL_CALL,
                                        f"Tool: {item.get('name', 'unknown')}",
                                        source="claude"
                                    )
                    
                    elif event_type == "tool_result":
                        content = str(data.get("content", ""))
                        yield self._emit_event(
                            EventType.TOOL_RESULT,
                            content,
                            source="claude"
                        )
                    
                    elif event_type == "result":
                        # Claude finished - extract the result text
                        result_text = data.get("result", "")
                        if result_text:
                            yield self._emit_event(
                                EventType.TEXT,
                                result_text,
                                source="claude"
                            )
                        yield self._emit_event(
                            EventType.COMPLETED,
                            f"Claude finished (exit: {data.get('subtype', 'unknown')})",
                            source="claude"
                        )
                    
                    elif event_type == "error":
                        yield self._emit_event(
                            EventType.ERROR,
                            str(data.get("error", "Unknown error")),
                            source="claude"
                        )
                    
                except json.JSONDecodeError:
                    # Raw text output
                    yield self._emit_event(
                        EventType.RAW,
                        line,
                        source="claude"
                    )
            
            process.wait(timeout=300)
            
        except subprocess.TimeoutExpired:
            process.kill()
            yield self._emit_event(EventType.ERROR, "[Claude timed out]", source="claude")
        except FileNotFoundError:
            yield self._emit_event(EventType.ERROR, "[Claude CLI not found]", source="claude")
        except Exception as e:
            yield self._emit_event(EventType.ERROR, f"[Error: {str(e)[:100]}]", source="claude")
    
    def _synthesize_instruction(self, result: FunctionResult) -> str:
        """
        Synthesize next Claude instruction from function result.
        
        Args:
            result: Result from the last function execution.
            
        Returns:
            Instruction string for Claude.
        """
        if result.name == FunctionName.VERIFY_CODE:
            if result.success:
                return "Code verification passed. Please confirm completion or continue with any remaining work."
            else:
                return f"Code verification FAILED:\n{result.output[:500]}\n\nPlease fix the issues."
        
        elif result.name == FunctionName.RUN_TESTS:
            if result.success:
                return "Tests passed. Please confirm completion or continue with any remaining work."
            else:
                return f"Tests FAILED:\n{result.output[:500]}\n\nPlease fix the failing tests."
        
        elif result.name == FunctionName.CHECK_SERVER:
            if result.success:
                return f"Server is running: {result.output}. Continue with any remaining work."
            else:
                return f"Server check failed: {result.output}. Please investigate."
        
        elif result.name == FunctionName.READ_FILE:
            return "File contents reviewed. Continue with the current task."
        
        else:
            return "Continue with the current task."
    
    def _emit_event(
        self,
        event_type: EventType,
        content: str,
        source: str = "ccp",
    ) -> OutputEvent:
        """Create an OutputEvent."""
        return OutputEvent(
            event_type=event_type,
            content=content,
            metadata={"source": source},
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ccp(
    working_dir: str = ".",
    session_id: Optional[str] = None,
    user_mission: Optional[str] = None,
    **kwargs: Any,
) -> CCP:
    """Factory function to create a CCP instance."""
    return CCP(
        working_dir=working_dir,
        session_id=session_id,
        user_mission=user_mission,
        **kwargs,
    )


def list_sessions() -> List[Dict[str, Any]]:
    """List all available sessions."""
    module_dir = Path(__file__).parent
    sessions_dir = module_dir / "sessions"
    return CCPMemory.list_sessions(sessions_dir)
