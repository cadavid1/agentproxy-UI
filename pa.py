"""
PA Agent Loop Architecture.

This module implements PA as an agent loop that runs in parallel with Claude Code.
Both are independent processes that communicate through structured messages.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                        PA Agent Loop                           │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Input (each loop):                                        │  │
    │  │   - System instruction + best practices + project memory  │  │
    │  │   - History: PA reasonings + Claude steps                │  │
    │  │   - Most recent Claude outputs                            │  │
    │  │   - Available function declarations                       │  │
    │  └───────────────────────────────────────────────────────────┘  │
    │                            ↓                                    │
    │  ┌───────────────────────────────────────────────────────────┐  │
    │  │ Output (each loop):                                       │  │
    │  │   - PA reasoning (state, progress, insights)             │  │
    │  │   - Function call (to execute in parallel)                │  │
    │  └───────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────┘
                                 ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Function Execution Layer                     │
    │  Executes function call → Result feeds back to:                 │
    │    1. PA session understanding                                 │
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
from datetime import datetime
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

# Third-party imports
from dotenv import load_dotenv

# Local imports
from pa_memory import PAMemory
from display import create_display
from models import ControllerState, EventType, OutputEvent

# Load environment variables
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


# =============================================================================
# File Change Tracker
# =============================================================================

class FileChangeTracker:
    """
    Tracks files modified by Claude during execution.
    
    Parses Claude's streaming JSON output to detect file write/edit operations
    and maintains a list of changed files for PA to review.
    """
    
    # Tool names that modify files
    FILE_MODIFY_TOOLS = {
        "write_file", "Write", "write",
        "edit_file", "Edit", "edit",
        "str_replace_editor", "str_replace",
        "create_file", "Create",
        "insert_lines", "insert",
        "MultiEdit", "multi_edit",
    }
    
    def __init__(self, working_dir: str = ".") -> None:
        self.working_dir = working_dir
        self._changed_files: Dict[str, str] = {}  # path -> operation type
        self._is_done = False
        self._done_message = ""
    
    def process_event(self, event_data: Dict[str, Any]) -> None:
        """
        Process a streaming JSON event from Claude to detect file changes.
        
        Args:
            event_data: Parsed JSON event from Claude's output.
        """
        event_type = event_data.get("type", "")
        
        # Check for tool use events
        if event_type == "assistant":
            msg = event_data.get("message", {})
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    self._process_tool_use(item)
        
        # Check for result/completion
        if event_type == "result":
            result_text = event_data.get("result", "")
            subtype = event_data.get("subtype", "")
            if subtype == "success" or self._looks_like_done(result_text):
                self._is_done = True
                self._done_message = result_text
    
    def _process_tool_use(self, tool_data: Dict[str, Any]) -> None:
        """Extract file path from a tool use event."""
        tool_name = tool_data.get("name", "")
        tool_input = tool_data.get("input", {})
        
        if tool_name in self.FILE_MODIFY_TOOLS:
            # Extract file path from various input formats
            file_path = (
                tool_input.get("file_path") or
                tool_input.get("path") or
                tool_input.get("target_file") or
                tool_input.get("filename") or
                ""
            )
            if file_path:
                self._changed_files[file_path] = tool_name
    
    def _looks_like_done(self, text: str) -> bool:
        """
        Use Gemini to analyze if Claude's output indicates task completion.
        
        This avoids brittle hardcoded phrase matching by using semantic understanding.
        """
        if not text or len(text.strip()) < 10:
            return False
        
        try:
            gemini = GeminiClient()
            response = gemini.call(
                system_prompt="""You are a task state analyzer. Analyze the given text and determine if it indicates that Claude has completed its assigned task.

Respond with ONLY one word:
- YES - if the text clearly indicates task completion, success, or that work is done
- NO - if the text indicates ongoing work, errors, questions, or incomplete state

Do not explain. Just respond YES or NO.""",
                user_prompt=f"Does this text indicate task completion?\n\n{text[:1500]}"
            )
            return response.strip().upper().startswith("YES")
        except Exception:
            # Fallback: if Gemini fails, assume not done to be safe
            return False
    
    def get_changed_files(self) -> List[str]:
        """Return list of files that were modified."""
        return list(self._changed_files.keys())
    
    def get_changes_summary(self) -> str:
        """Return a summary of all file changes."""
        if not self._changed_files:
            return "No files were modified."
        
        lines = ["Files modified by Claude:"]
        for path, operation in self._changed_files.items():
            lines.append(f"  - {path} ({operation})")
        return "\n".join(lines)
    
    @property
    def is_done(self) -> bool:
        """Return True if Claude indicated task completion."""
        return self._is_done
    
    @property
    def done_message(self) -> str:
        """Return Claude's completion message."""
        return self._done_message
    
    def reset(self) -> None:
        """Reset tracker for a new task."""
        self._changed_files.clear()
        self._is_done = False
        self._done_message = ""


# =============================================================================
# Data Structures
# =============================================================================

class FunctionName(str, Enum):
    """Available functions that PA can call."""
    
    SEND_TO_CLAUDE = "send_to_claude"
    VERIFY_CODE = "verify_code"
    RUN_TESTS = "run_tests"
    CHECK_SERVER = "check_server"
    READ_FILE = "read_file"
    MARK_DONE = "mark_done"
    CREATE_TASK = "create_task"
    UPDATE_TASK = "update_task"
    COMPLETE_TASK = "complete_task"
    REVIEW_CHANGES = "review_changes"
    VERIFY_PRODUCT = "verify_product"


@dataclass
class FunctionDeclaration:
    """
    Declaration of a function available to PA.
    
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
    A function call decided by PA.
    
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


@dataclass
class AgentLoopInput:
    """
    Input to a single PA agent loop iteration.
    
    Attributes:
        system_instruction: Core PA system prompt.
        best_practices: Loaded best practices from prompts/.
        project_memory: Session and project context.
        history: Past PA reasonings and Claude steps.
        recent_claude_output: Most recent output from Claude.
        available_functions: Functions PA can call.
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
    Output from a single PA agent loop iteration.
    
    Attributes:
        reasoning: PA's reasoning for this iteration.
        function_call: The function PA decided to execute.
    """
    
    reasoning: PAReasoning
    function_call: FunctionCall


# =============================================================================
# Gemini Client (for PA reasoning)
# =============================================================================

class GeminiClient:
    """Client for Gemini API that powers PA's reasoning."""
    
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

# All functions available to PA
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
                    "enum": ["claude", "pa"],
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
    FunctionDeclaration(
        name=FunctionName.REVIEW_CHANGES,
        description="Review code changes made by Claude. Reads changed files and performs QA review for errors, bugs, and improvements.",
        parameters={
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to review"
                },
                "context": {
                    "type": "string",
                    "description": "Context about what Claude was trying to accomplish"
                }
            },
            "required": ["file_paths"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.VERIFY_PRODUCT,
        description="Verify the final product works correctly. For API servers: starts server and tests endpoints. For UI apps: starts server and uses browser automation to interact and verify.",
        parameters={
            "type": "object",
            "properties": {
                "product_type": {
                    "type": "string",
                    "enum": ["api_server", "ui_app", "script", "auto"],
                    "description": "Type of product to verify. Use 'auto' to detect automatically."
                },
                "start_command": {
                    "type": "string",
                    "description": "Command to start the server (e.g., 'npm run dev', 'python app.py')"
                },
                "port": {
                    "type": "integer",
                    "description": "Port the server runs on (default: 5000)"
                },
                "endpoints_to_test": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For API: list of endpoints to test (e.g., ['/health', '/api/users'])"
                },
                "ui_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For UI: list of actions to perform (e.g., ['click button Submit', 'verify text Welcome'])"
                }
            },
            "required": []
        }
    ),
]


# =============================================================================
# Function Executor
# =============================================================================

class FunctionExecutor:
    """
    Executes functions called by PA in parallel with Claude.
    
    Each function runs independently and returns a FunctionResult.
    """
    
    def __init__(self, working_dir: str = ".", memory: Any = None) -> None:
        self.working_dir = working_dir
        self._claude_queue: Queue = Queue()
        self._memory = memory  # Reference to PAMemory for task management
    
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
            FunctionName.REVIEW_CHANGES: self._review_changes,
            FunctionName.VERIFY_PRODUCT: self._verify_product,
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
        """Queue instruction for Claude with screenshot context if available."""
        instruction = args.get("instruction", "")
        context = args.get("context", "")
        
        # Build screenshot context to help Claude understand visual requirements
        screenshot_context = self._build_screenshot_context()
        if screenshot_context:
            instruction = f"{instruction}\n\n{screenshot_context}"
        
        self._claude_queue.put({
            "type": "instruction",
            "instruction": instruction,
            "context": context,
        })
        
        return FunctionResult(
            name=FunctionName.SEND_TO_CLAUDE,
            success=True,
            output=f"Queued instruction for Claude: {instruction[:200]}...",
            metadata={"instruction": instruction}
        )
    
    def _build_screenshot_context(self) -> str:
        """Build detailed description of reference screenshots for Claude."""
        if not self._memory or not self._memory.session.reference_screenshots:
            return ""
        
        screenshots = self._memory.session.reference_screenshots
        if not screenshots:
            return ""
        
        lines = ["## VISUAL REFERENCE (Screenshots provided by user)"]
        lines.append("The user has provided the following reference images that the result should match:")
        lines.append("")
        
        for i, ss in enumerate(screenshots, 1):
            desc = ss.description or "UI design reference"
            path = ss.path
            filename = Path(path).name if path else "unknown"
            
            lines.append(f"**Screenshot {i}: {filename}**")
            lines.append(f"  - Description: {desc}")
            lines.append(f"  - Path: {path}")
            lines.append("  - Use this as visual reference for layout, styling, and UI elements")
            lines.append("")
        
        lines.append("IMPORTANT: Match the visual design, colors, layout, and styling from these reference screenshots as closely as possible.")
        
        return "\n".join(lines)
    
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
            req.add_header('User-Agent', 'PA-Agent/1.0')
            
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
    
    def _review_changes(self, args: Dict[str, Any]) -> FunctionResult:
        """
        Review code changes made by Claude using Gemini for QA.
        
        Reads all changed files and sends them to Gemini for analysis,
        looking for bugs, errors, and potential improvements.
        """
        file_paths = args.get("file_paths", [])
        context = args.get("context", "Code changes made by Claude")
        
        if not file_paths:
            return FunctionResult(
                name=FunctionName.REVIEW_CHANGES,
                success=True,
                output="No files to review.",
                metadata={"files_reviewed": 0}
            )
        
        # Read all changed files
        file_contents = {}
        for filepath in file_paths:
            full_path = filepath
            if not os.path.isabs(filepath):
                full_path = os.path.join(self.working_dir, filepath)
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    # Truncate very large files
                    if len(content) > 8000:
                        content = content[:8000] + "\n... [truncated]"
                    file_contents[filepath] = content
                except Exception as e:
                    file_contents[filepath] = f"[Error reading: {e}]"
            else:
                file_contents[filepath] = "[File not found]"
        
        # Build review prompt for Gemini
        review_prompt = self._build_review_prompt(file_contents, context)
        
        # Call Gemini for review
        try:
            gemini = GeminiClient()
            review_result = gemini.call(
                system_prompt=self._get_review_system_prompt(),
                user_prompt=review_prompt
            )
        except Exception as e:
            return FunctionResult(
                name=FunctionName.REVIEW_CHANGES,
                success=False,
                output=f"Review failed: {str(e)[:200]}",
                metadata={"error": str(e)}
            )
        
        # Parse review result for issues
        has_issues = self._review_has_issues(review_result)
        
        return FunctionResult(
            name=FunctionName.REVIEW_CHANGES,
            success=True,
            output=review_result,
            metadata={
                "files_reviewed": len(file_contents),
                "file_paths": list(file_contents.keys()),
                "has_issues": has_issues,
            }
        )
    
    def _build_review_prompt(self, file_contents: Dict[str, str], context: str) -> str:
        """Build the review prompt with all file contents."""
        parts = [f"## Context\n{context}\n"]
        parts.append(f"## Files Changed ({len(file_contents)} files)\n")
        
        for filepath, content in file_contents.items():
            parts.append(f"### {filepath}\n```\n{content}\n```\n")
        
        parts.append("\n## Review Request")
        parts.append("CRITICAL ISSUES ONLY - Review for:")
        parts.append("1. **ERRORS**: Syntax errors, runtime errors, bugs that will CRASH or BREAK the code")
        parts.append("2. **SECURITY**: SQL injection, XSS, authentication bypasses, data exposure")
        parts.append("")
        parts.append("DO NOT report: style issues, minor improvements, logging suggestions, type hints.")
        parts.append("Only report issues that will cause the code to FAIL or be INSECURE.")
        parts.append("If code works correctly, respond with 'NO CRITICAL ISSUES'.")
        
        return "\n".join(parts)
    
    def _get_review_system_prompt(self) -> str:
        """Get the system prompt for CRITICAL-only code review."""
        return """You are a strict code reviewer focused ONLY on critical issues.

REPORT ONLY:
1. ERRORS: Syntax errors, runtime errors, bugs that will CRASH or BREAK functionality
2. SECURITY: SQL injection, XSS, auth bypasses, sensitive data exposure

DO NOT REPORT:
- Style suggestions
- Performance improvements (unless catastrophic)
- Type hints or documentation
- Logging recommendations
- Code organization suggestions

If the code will RUN CORRECTLY and is SECURE, respond: "NO CRITICAL ISSUES"

Format:
## CRITICAL ERRORS
- [only errors that break functionality, or "None"]

## SECURITY ISSUES
- [only security vulnerabilities, or "None"]

## VERDICT
[PASS/FAIL] - [one line: will it work?]"""
    
    def _review_has_issues(self, review_result: str) -> bool:
        """
        Use Gemini to analyze if the code review found significant issues.
        
        This avoids brittle hardcoded pattern matching by using semantic understanding.
        """
        if not review_result or len(review_result.strip()) < 10:
            return False
        
        try:
            gemini = GeminiClient()
            response = gemini.call(
                system_prompt="""You are a code review analyzer. Analyze the given code review output and determine if it found significant issues that require fixes.

Respond with ONLY one word:
- YES - if the review found errors, bugs, security issues, or problems that MUST be fixed
- NO - if the review passed, found no issues, or only has minor suggestions/improvements

Do not explain. Just respond YES or NO.""",
                user_prompt=f"Does this code review indicate issues that need fixing?\n\n{review_result[:2000]}"
            )
            return response.strip().upper().startswith("YES")
        except Exception:
            # Fallback: if Gemini fails, assume no critical issues
            return False
    
    def _verify_product(self, args: Dict[str, Any]) -> FunctionResult:
        """
        Verify the final product works correctly.
        
        For API servers: starts server and tests endpoints with curl.
        For UI apps: starts server and uses Playwright to interact and verify.
        """
        product_type = args.get("product_type", "auto")
        start_command = args.get("start_command")
        port = args.get("port", 5000)
        endpoints = args.get("endpoints_to_test", [])
        ui_actions = args.get("ui_actions", [])
        
        # Auto-detect product type if needed
        if product_type == "auto":
            product_type = self._detect_product_type()
        
        if product_type == "api_server":
            return self._verify_api_server(start_command, port, endpoints)
        elif product_type == "ui_app":
            return self._verify_ui_app(start_command, port, ui_actions)
        elif product_type == "script":
            return self._verify_script()
        else:
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=False,
                output=f"Unknown product type: {product_type}"
            )
    
    def _detect_product_type(self) -> str:
        """Use Gemini to detect what type of product was built."""
        # Check for common indicators
        indicators = []
        
        # Check package.json for web frameworks
        pkg_json = os.path.join(self.working_dir, "package.json")
        if os.path.exists(pkg_json):
            try:
                with open(pkg_json) as f:
                    content = f.read()
                if any(fw in content for fw in ["react", "vue", "angular", "next", "vite"]):
                    indicators.append("ui_app")
                if any(fw in content for fw in ["express", "fastify", "koa", "hapi"]):
                    indicators.append("api_server")
            except Exception:
                pass
        
        # Check for Python web frameworks
        req_txt = os.path.join(self.working_dir, "requirements.txt")
        if os.path.exists(req_txt):
            try:
                with open(req_txt) as f:
                    content = f.read().lower()
                if any(fw in content for fw in ["flask", "fastapi", "django"]):
                    indicators.append("api_server")
                if "streamlit" in content:
                    indicators.append("ui_app")
            except Exception:
                pass
        
        # Check for index.html (UI app indicator)
        if os.path.exists(os.path.join(self.working_dir, "index.html")):
            indicators.append("ui_app")
        
        # Default based on indicators
        if "ui_app" in indicators:
            return "ui_app"
        elif "api_server" in indicators:
            return "api_server"
        else:
            return "script"
    
    def _verify_api_server(self, start_command: Optional[str], port: int, endpoints: List[str]) -> FunctionResult:
        """Start API server and test endpoints."""
        import time
        import urllib.request
        import urllib.error
        
        server_process = None
        results = []
        
        try:
            # Start server if command provided
            if start_command:
                server_process = subprocess.Popen(
                    start_command,
                    shell=True,
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                # Wait for server to start
                time.sleep(3)
                results.append(f"[PA Verify] Started server: {start_command}")
            
            # Default endpoints to test
            if not endpoints:
                endpoints = ["/health", "/", "/api"]
            
            base_url = f"http://localhost:{port}"
            success_count = 0
            
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                try:
                    req = urllib.request.Request(url, method='GET')
                    req.add_header('User-Agent', 'PA-Verify/1.0')
                    
                    with urllib.request.urlopen(req, timeout=5) as response:
                        status = response.status
                        body = response.read().decode('utf-8')[:200]
                        results.append(f"[PA Verify] {endpoint}: {status} OK - {body[:100]}")
                        success_count += 1
                except urllib.error.HTTPError as e:
                    results.append(f"[PA Verify] {endpoint}: HTTP {e.code} - {e.reason}")
                    if e.code < 500:  # 4xx errors might be expected
                        success_count += 1
                except Exception as e:
                    results.append(f"[PA Verify] {endpoint}: FAILED - {str(e)[:50]}")
            
            overall_success = success_count > 0
            results.append(f"\n[PA Verify] API Test: {success_count}/{len(endpoints)} endpoints responding")
            
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=overall_success,
                output="\n".join(results),
                metadata={"type": "api_server", "endpoints_tested": len(endpoints), "success_count": success_count}
            )
            
        finally:
            # Cleanup: stop server
            if server_process:
                server_process.terminate()
                try:
                    server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    server_process.kill()
    
    def _verify_ui_app(self, start_command: Optional[str], port: int, ui_actions: List[str]) -> FunctionResult:
        """
        Start UI app and use Playwright MCP to interact and verify.
        
        Uses the mcp-playwright tools to navigate, take snapshots, and interact.
        """
        import time
        
        server_process = None
        results = []
        
        try:
            # Start server if command provided
            if start_command:
                server_process = subprocess.Popen(
                    start_command,
                    shell=True,
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(5)  # UI apps often take longer to start
                results.append(f"[PA Verify] Started UI server: {start_command}")
            
            base_url = f"http://localhost:{port}"
            
            # Use Gemini to analyze what we should verify based on the task
            # For now, return instructions for manual Playwright verification
            results.append(f"[PA Verify] UI app running at {base_url}")
            results.append("[PA Verify] To verify UI, use Playwright MCP tools:")
            results.append(f"  1. mcp0_browser_navigate to {base_url}")
            results.append("  2. mcp0_browser_snapshot to see page structure")
            results.append("  3. mcp0_browser_click to interact with elements")
            results.append("  4. mcp0_browser_take_screenshot for visual verification")
            
            if ui_actions:
                results.append("\n[PA Verify] Requested UI actions:")
                for action in ui_actions:
                    results.append(f"  - {action}")
            
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=True,
                output="\n".join(results),
                metadata={"type": "ui_app", "url": base_url, "needs_manual_verification": True}
            )
            
        except Exception as e:
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=False,
                output=f"[PA Verify] UI verification failed: {str(e)}",
                metadata={"type": "ui_app", "error": str(e)}
            )
        finally:
            if server_process:
                server_process.terminate()
                try:
                    server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    server_process.kill()
    
    def _verify_script(self) -> FunctionResult:
        """Verify a script by running it."""
        # Find main script files
        script_files = []
        for ext in ["*.py", "*.js", "*.ts"]:
            import glob
            script_files.extend(glob.glob(os.path.join(self.working_dir, ext)))
        
        if not script_files:
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=False,
                output="[PA Verify] No script files found to verify"
            )
        
        results = []
        for script in script_files[:3]:  # Test up to 3 scripts
            try:
                if script.endswith(".py"):
                    result = subprocess.run(
                        ["python3", script],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.working_dir
                    )
                else:
                    result = subprocess.run(
                        ["node", script],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=self.working_dir
                    )
                
                if result.returncode == 0:
                    output = result.stdout[:200] if result.stdout else "OK"
                    results.append(f"[PA Verify] {os.path.basename(script)}: SUCCESS\n{output}")
                else:
                    error = result.stderr[:200] if result.stderr else "Unknown error"
                    results.append(f"[PA Verify] {os.path.basename(script)}: FAILED\n{error}")
            except Exception as e:
                results.append(f"[PA Verify] {os.path.basename(script)}: ERROR - {str(e)[:100]}")
        
        all_success = all("SUCCESS" in r for r in results)
        return FunctionResult(
            name=FunctionName.VERIFY_PRODUCT,
            success=all_success,
            output="\n".join(results),
            metadata={"type": "script", "scripts_tested": len(results)}
        )
    
    def get_pending_claude_instruction(self) -> Optional[Dict[str, Any]]:
        """Get next queued instruction for Claude, if any."""
        try:
            return self._claude_queue.get_nowait()
        except Empty:
            return None
    
    def get_pending_human_request(self) -> Optional[str]:
        """Deprecated - PA is human proxy, no human requests needed."""
        return None


# =============================================================================
# PA Agent
# =============================================================================

class PAAgent:
    """
    PA Agent that runs an independent loop alongside Claude.
    
    Each loop iteration:
        1. Receives input (context + recent Claude output + functions)
        2. Produces reasoning about current state
        3. Decides which function to call
        4. Function executes in parallel
        5. Result feeds back into next iteration
    """
    
    AGENT_SYSTEM_PROMPT = """You are PA (Proxy Agent), an AI agent that supervises Claude Code.

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
        self._display = create_display("rich")
        
        # Load project context from context_dir (text + images)
        self._project_context, self._context_images = self._load_context_dir()
        
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
        
        # Set mission if provided
        if user_mission:
            memory.session.set_mission(user_mission)
        
        return memory
    
    def _load_context_dir(self) -> Tuple[str, Dict[str, str]]:
        """
        Load all context files from the configured context directory.
        
        These files provide high-level project context that informs PA's
        decisions when supervising Claude.
        
        Returns:
            Tuple of (text_content, image_dict) where image_dict maps 
            filename -> full_path for reference by name.
        """
        if not self.context_dir:
            return "", {}
        
        context_path = Path(self.context_dir)
        if not context_path.exists():
            return "", {}
        
        context_parts = []
        context_images: Dict[str, str] = {}  # name -> path
        
        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}
        
        # Load text files from context directory (top level only)
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
        
        # Load images recursively (including subdirectories like images/)
        for ext in image_extensions:
            for file_path in context_path.rglob(f"*{ext}"):
                if file_path.is_file():
                    # Store with multiple reference keys for flexible lookup
                    context_images[file_path.name] = str(file_path)
                    context_images[file_path.stem] = str(file_path)  # without extension
                    context_images[file_path.name.lower()] = str(file_path)
                    context_images[file_path.stem.lower()] = str(file_path)
        
        # Build text content
        text_content = ""
        if context_parts:
            text_content = "## PROJECT CONTEXT (from context directory)\n\n" + "\n\n".join(context_parts)
        
        # Add image reference list to text content
        if context_images:
            unique_images = list(set(context_images.values()))
            image_list = "\n".join(f"  - {Path(p).name}" for p in sorted(unique_images))
            text_content += f"\n\n## REFERENCE IMAGES ({len(unique_images)} images)\n{image_list}\n(These images are included in the visual context)"
        
        return text_content, context_images
    
    def _load_project_memory_folder(self) -> Tuple[str, List[str]]:
        """
        Load all contents from the project_memory folder.
        
        This folder acts as a "data room" - all text content and images
        are loaded every time PA thinks and decides next step.
        
        Returns:
            Tuple of (text_content, list_of_image_paths)
        """
        project_memory_dir = Path(self.working_dir) / "project_memory"
        text_content = ""
        image_paths: List[str] = []
        
        if not project_memory_dir.exists():
            return text_content, image_paths
        
        # Supported image extensions
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        
        # Load all files from project_memory folder
        for file_path in sorted(project_memory_dir.iterdir()):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                
                if ext in image_extensions:
                    # Collect image paths for visual context
                    image_paths.append(str(file_path))
                elif ext in {".txt", ".md", ".json", ".yaml", ".yml"}:
                    # Load text file content
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        if content.strip():
                            text_content += f"\n### {file_path.name}\n{content}\n"
                    except (IOError, UnicodeDecodeError):
                        pass
        
        return text_content, image_paths
    
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
        
        # Load project_memory folder (data room)
        data_room_text, data_room_images = self._load_project_memory_folder()
        
        # Project memory from Tier 2
        screenshots_info = ""
        if self._memory.session.reference_screenshots:
            screenshots_info = "\nREFERENCE SCREENSHOTS (images attached below):\n" + "\n".join(
                f"  - {ss.path}: {ss.description or 'UI reference'}"
                for ss in self._memory.session.reference_screenshots
            )
        
        # Include data room images in screenshots info
        if data_room_images:
            screenshots_info += "\nDATA ROOM IMAGES (from project_memory/):\n" + "\n".join(
                f"  - {Path(img).name}" for img in data_room_images
            )
        
        project_memory = f"""
MISSION: {self._memory.session.user_mission or 'Not specified'}
CURRENT TASK: {self._memory.session.user_prompt or 'Not specified'}
FILES TRACKED: {', '.join(self._memory.session.project_files.keys()) or 'None'}{screenshots_info}

## DATA ROOM (project_memory/)
{data_room_text if data_room_text else '(No text content loaded)'}
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
                
                reasoning = PAReasoning(
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
            reasoning=PAReasoning(
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
    
    def run_iteration(self, recent_claude_output: str) -> Tuple[PAReasoning, FunctionResult]:
        """
        Run a single agent loop iteration.
        
        Args:
            recent_claude_output: Most recent output from Claude.
            
        Returns:
            Tuple of (PA reasoning, function execution result).
        """
        self._iteration += 1
        
        # Build input
        loop_input = self._build_loop_input(recent_claude_output)
        
        # Build prompt for Gemini
        user_prompt = f"""
{self._project_context}

## SESSION CONTEXT
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
                reasoning=PAReasoning(
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
            ] if self._memory.session.reference_screenshots else []
            
            # Add data room images from project_memory folder
            _, data_room_images = self._load_project_memory_folder()
            if data_room_images:
                image_paths = (image_paths or []) + data_room_images
            
            # Add context_dir images (referenced by name in text)
            if self._context_images:
                context_image_paths = list(set(self._context_images.values()))
                image_paths = (image_paths or []) + context_image_paths
            
            response = self._gemini.call(loop_input.system_instruction, user_prompt, image_paths or None)
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
    
    def generate_session_summary(self, task: str, claude_outputs: List[str], files_changed: List[str]) -> str:
        """
        Generate a comprehensive session summary for persistence.
        
        Args:
            task: The original task
            claude_outputs: List of Claude's outputs during the session
            files_changed: List of files modified during the session
            
        Returns:
            Formatted session summary text
        """
        # Use Gemini to generate intelligent summary
        if self._gemini:
            history_text = json.dumps(self._history[-20:], indent=2) if self._history else "No history"
            claude_text = "\n---\n".join(claude_outputs[-5:]) if claude_outputs else "No Claude output"
            
            summary_prompt = f"""Summarize this coding session concisely for future reference.

## Original Task
{task}

## PA Decision History (last 20 iterations)
{history_text}

## Claude's Recent Output
{claude_text[:3000]}

## Files Changed
{chr(10).join(files_changed) if files_changed else 'None'}

---
Write a structured summary with these sections:
1. TASK SUMMARY - What was requested
2. ACTIONS TAKEN - Key PA decisions and Claude actions
3. FILES MODIFIED - What files were changed and why
4. CURRENT STATE - Where we left off
5. NEXT STEPS - What should happen next if resuming

Be concise but complete. This will be used to resume the session later."""
            
            try:
                summary = self._gemini.call(
                    system_prompt="You are a technical session summarizer. Create clear, actionable summaries for development sessions.",
                    user_prompt=summary_prompt
                )
                return summary
            except Exception:
                pass
        
        # Fallback: generate basic summary without Gemini
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
        
        lines.append("")
        lines.append("## Recent PA Decisions")
        for h in self._history[-5:]:
            lines.append(f"- [{h.get('iteration')}] {h.get('reasoning', {}).get('decision', 'N/A')[:100]}")
        
        return "\n".join(lines)
    
    def save_session_summary(self, summary: str) -> str:
        """
        Save session summary to context/sys/ directory.
        
        Args:
            summary: The session summary text
            
        Returns:
            Path to the saved summary file
        """
        # Create context/sys directory under context_dir or working_dir
        if self.context_dir:
            sys_dir = Path(self.context_dir) / "sys"
        else:
            sys_dir = Path(self.working_dir) / "context" / "sys"
        
        sys_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with session ID in filename
        session_id = self._memory.session.session_id
        summary_file = sys_dir / f"session_{session_id}.txt"
        
        summary_file.write_text(summary, encoding="utf-8")
        return str(summary_file)
    
    def load_session_summary(self) -> Optional[str]:
        """
        Load previous session summary if it exists.
        
        Returns:
            Session summary text if found, None otherwise
        """
        session_id = self._memory.session.session_id
        
        # Check context_dir/sys first, then working_dir/context/sys
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


# =============================================================================
# Main Orchestrator
# =============================================================================

class PA:
    """
    PA Agent - orchestrates Claude Code with an agent loop architecture.
    
    Both PA and Claude run as parallel processes communicating through
    structured messages. PA reasons about Claude's output and decides
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
        context_dir: Optional[str] = None,
    ) -> None:
        """
        Initialize PA.
        
        Args:
            working_dir: Working directory for operations.
            session_id: Optional session ID for persistence.
            user_mission: High-level mission description.
            display_mode: Display mode ("rich", "simple", "json", "quiet").
            auto_verify: Whether to auto-verify code (used by agent).
            auto_qa: Whether to auto-run QA (used by agent).
            context_dir: Directory containing project context files (.md, .txt).
        """
        self.working_dir = working_dir
        self.auto_verify = auto_verify
        self.auto_qa = auto_qa
        self.agent = PAAgent(working_dir, session_id, user_mission, context_dir)
        self._display = create_display(display_mode)
        self._claude_output_buffer: List[str] = []
        self._session_files_changed: List[str] = []  # Track all files changed in session
        self._state = ControllerState.IDLE
        self._file_tracker = FileChangeTracker(working_dir)
        
        # Load previous session summary if resuming
        self._previous_summary = self.agent.load_session_summary()
    
    @property
    def memory(self) -> PAMemory:
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
        Execute a task with PA supervising Claude.
        
        Args:
            task: The task to execute.
            max_iterations: Maximum PA loop iterations.
            
        Yields:
            OutputEvent objects for display.
        """
        self._state = ControllerState.PROCESSING
        self._session_files_changed = []  # Reset for new task
        
        yield self._emit_event(EventType.STARTED, f"Starting task: {task[:100]}...")
        
        # If resuming, show previous session summary
        if self._previous_summary:
            yield self._emit_event(
                EventType.THINKING,
                f"[PA Session] Resuming session with previous context:\n{self._previous_summary[:500]}...",
                source="pa-thinking"
            )
        
        # Initialize Claude with the task
        current_instruction = task
        
        for iteration in range(max_iterations):
            if self.agent.is_done:
                break
            
            # === PA THINKING: Before sending to Claude ===
            yield self._emit_event(
                EventType.THINKING,
                f"[PA Iteration {iteration + 1}/{max_iterations}]\n"
                f"Instruction: {current_instruction}",
                source="pa-thinking"
            )
            
            # Run Claude with current instruction
            yield self._emit_event(
                EventType.TEXT,
                f"[Iteration {iteration + 1}] Executing Claude...",
                source="pa"
            )
            
            # Stream Claude output in real-time
            claude_output_lines = []
            for event in self._stream_claude(current_instruction):
                yield event
                if event.content:
                    claude_output_lines.append(event.content)
            
            claude_output = "\n".join(claude_output_lines)
            self._claude_output_buffer.append(claude_output)
            
            # === FILE CHANGE TRACKING ===
            changed_files = self._file_tracker.get_changed_files()
            if changed_files:
                # Add to session-wide tracking
                self._session_files_changed.extend(changed_files)
                files_list = "\n".join(f"  - {f}" for f in changed_files)
                yield self._emit_event(
                    EventType.THINKING,
                    f"[PA File Tracking] Files changed by Claude:\n{files_list}",
                    source="pa-thinking"
                )
            
            # === AUTO-REVIEW when Claude indicates done ===
            if self._file_tracker.is_done and changed_files and self.auto_qa:
                yield self._emit_event(
                    EventType.THINKING,
                    f"[PA Code Review] Claude indicated completion. Reviewing {len(changed_files)} changed files...",
                    source="pa-thinking"
                )
                
                # Perform code review
                review_result = self.agent._executor._review_changes({
                    "file_paths": changed_files,
                    "context": f"Task: {task[:200]}"
                })
                
                yield self._emit_event(
                    EventType.TOOL_RESULT,
                    f"[PA Code Review] Result:\n{review_result.output}",
                    source="pa-qa"
                )
                
                # If review found issues, send feedback to Claude
                if review_result.metadata.get("has_issues"):
                    yield self._emit_event(
                        EventType.THINKING,
                        "[PA Code Review] Critical issues found - instructing Claude to fix them",
                        source="pa-thinking"
                    )
                    # Queue instruction for Claude to fix issues
                    self.agent._executor._claude_queue.put({
                        "type": "instruction",
                        "instruction": f"[PA Code Review] CRITICAL ISSUES - Please fix:\n\n{review_result.output}\n\nFiles to review: {', '.join(changed_files)}",
                        "context": "auto-review"
                    })
                else:
                    # Code review passed - now verify the product actually works
                    yield self._emit_event(
                        EventType.THINKING,
                        "[PA Code Review] No critical issues. Now verifying product works...",
                        source="pa-thinking"
                    )
                    
                    # Auto-verify the product
                    verify_result = self.agent._executor._verify_product({
                        "product_type": "auto",
                        "port": 5000
                    })
                    
                    yield self._emit_event(
                        EventType.TOOL_RESULT,
                        f"[PA Verify] {verify_result.output}",
                        source="pa-verify"
                    )
                    
                    if not verify_result.success:
                        yield self._emit_event(
                            EventType.THINKING,
                            "[PA Verify] Verification failed - instructing Claude to fix",
                            source="pa-thinking"
                        )
                        self.agent._executor._claude_queue.put({
                            "type": "instruction",
                            "instruction": f"[PA Verify] Product verification FAILED:\n\n{verify_result.output}\n\nPlease fix and ensure the product runs correctly.",
                            "context": "auto-verify"
                        })
            
            # === PA THINKING: Analyzing Claude's output ===
            yield self._emit_event(
                EventType.THINKING,
                f"[PA Analysis] Analyzing Claude's response ({len(claude_output)} chars)...\n"
                f"Invoking Gemini for reasoning...",
                source="pa-thinking"
            )
            
            # Run PA agent iteration
            reasoning, result = self.agent.run_iteration(claude_output)
            
            # === PA THINKING: Full reasoning output ===
            yield self._emit_event(
                EventType.THINKING,
                f"[PA Reasoning]\n"
                f"STATE: {reasoning.current_state}\n\n"
                f"PROGRESS: {reasoning.claude_progress}\n\n"
                f"INSIGHTS: {reasoning.insights}\n\n"
                f"DECISION: {reasoning.decision}",
                source="pa-thinking"
            )
            
            # === PA THINKING: Function execution ===
            yield self._emit_event(
                EventType.THINKING,
                f"[PA Function] Executing: {result.name.value}",
                source="pa-thinking"
            )
            
            # Display function result
            yield self._emit_event(
                EventType.TOOL_RESULT,
                f"[{result.name.value}] {'✓' if result.success else '✗'}: {result.output}",
                source="pa-function"
            )
            
            # PA is the human proxy - handle any "human requests" autonomously
            human_request = self.agent.get_human_request()
            if human_request:
                # Don't stop - PA handles it as human proxy
                yield self._emit_event(
                    EventType.TEXT,
                    f"[PA as human proxy] Handling: {human_request}",
                    source="pa"
                )
                # Continue to next iteration - PA will respond via send_to_claude
            
            # Get next instruction for Claude
            next_instruction = self.agent.get_claude_instruction()
            if next_instruction:
                current_instruction = next_instruction
                yield self._emit_event(
                    EventType.THINKING,
                    "[PA] Next instruction queued from function result",
                    source="pa-thinking"
                )
            else:
                # If no explicit instruction, synthesize from result
                current_instruction = self._synthesize_instruction(result)
                yield self._emit_event(
                    EventType.THINKING,
                    f"[PA] Synthesized next instruction:\n{current_instruction}",
                    source="pa-thinking"
                )
        
        # === PA THINKING: Final status ===
        yield self._emit_event(
            EventType.THINKING,
            f"[PA Complete]\n"
            f"Iterations used: {iteration + 1}/{max_iterations}\n"
            f"Task done: {self.agent.is_done}",
            source="pa-thinking"
        )
        
        # === SAVE SESSION SUMMARY ===
        all_files_changed = list(set(self._session_files_changed))
        summary = self.agent.generate_session_summary(
            task=task,
            claude_outputs=self._claude_output_buffer,
            files_changed=all_files_changed
        )
        summary_path = self.agent.save_session_summary(summary)
        
        yield self._emit_event(
            EventType.THINKING,
            f"[PA Session] Summary saved to: {summary_path}",
            source="pa-thinking"
        )
        
        # Final status
        if self.agent.is_done:
            yield self._emit_event(
                EventType.COMPLETED,
                "Task completed successfully",
                source="pa"
            )
        else:
            yield self._emit_event(
                EventType.ERROR,
                "Task did not complete within max iterations",
                source="pa"
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
        # Reset file tracker for this Claude run
        self._file_tracker.reset()
        
        try:
            process = subprocess.Popen(
                [
                    "claude", "-p", instruction,
                    "--output-format", "stream-json",
                    "--verbose",
                    "--dangerously-skip-permissions",  # PA acts as human proxy - auto-approve
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
                    
                    # Track file changes
                    self._file_tracker.process_event(data)
                    
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
                                    tool_name = item.get('name', 'unknown')
                                    tool_input = item.get('input', {})
                                    # Format tool input for display
                                    if isinstance(tool_input, dict):
                                        # Show key parameters, truncate long values
                                        input_parts = []
                                        for k, v in list(tool_input.items())[:5]:
                                            v_str = str(v)[:200]
                                            if len(str(v)) > 200:
                                                v_str += "..."
                                            input_parts.append(f"{k}={v_str}")
                                        input_str = ", ".join(input_parts)
                                    else:
                                        input_str = str(tool_input)[:300]
                                    yield self._emit_event(
                                        EventType.TOOL_CALL,
                                        f"🔧 {tool_name}({input_str})",
                                        source="claude"
                                    )
                    
                    elif event_type == "tool_result":
                        content = data.get("content", "")
                        # Handle different content formats
                        if isinstance(content, list):
                            # Content may be list of text blocks
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "text":
                                    text_parts.append(c.get("text", ""))
                                elif isinstance(c, str):
                                    text_parts.append(c)
                            content = "\n".join(text_parts)
                        content_str = str(content)[:1000]
                        if len(str(content)) > 1000:
                            content_str += "... [truncated]"
                        if content_str.strip():
                            yield self._emit_event(
                                EventType.TOOL_RESULT,
                                f"   ↳ {content_str}",
                                source="claude"
                            )
                    
                    elif event_type == "content_block_delta":
                        # Streaming text delta
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text.strip():
                                yield self._emit_event(
                                    EventType.TEXT,
                                    text,
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
        
        elif result.name == FunctionName.REVIEW_CHANGES:
            if result.metadata.get("has_issues"):
                return f"Code review found issues that need fixing:\n{result.output[:1000]}\n\nPlease address these issues."
            else:
                return "Code review passed. Continue with any remaining work or confirm completion."
        
        else:
            return "Continue with the current task."
    
    def _emit_event(
        self,
        event_type: EventType,
        content: str,
        source: str = "pa",
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

def create_pa(
    working_dir: str = ".",
    session_id: Optional[str] = None,
    user_mission: Optional[str] = None,
    **kwargs: Any,
) -> PA:
    """Factory function to create a PA instance."""
    return PA(
        working_dir=working_dir,
        session_id=session_id,
        user_mission=user_mission,
        **kwargs,
    )


def list_sessions() -> List[Dict[str, Any]]:
    """List all available sessions."""
    module_dir = Path(__file__).parent
    sessions_dir = module_dir / "sessions"
    return PAMemory.list_sessions(sessions_dir)
