"""
Function Executor
=================

Executes functions called by PA during the agent loop.
Each function runs independently and returns a structured result.
"""

import glob
import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Callable, Dict, List, Optional

from .gemini_client import GeminiClient
from .telemetry import get_telemetry


# =============================================================================
# Browser Verification Helper
# =============================================================================

class BrowserVerifier:
    """Uses Playwright to verify web applications visually."""
    
    @staticmethod
    def verify_url(url: str, screenshot_path: Optional[str] = None, checks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Navigate to URL, take screenshot, and verify page loads.

        Args:
            url: URL to verify
            screenshot_path: Path to save screenshot (optional)
            checks: List of elements/text to verify on page (optional)

        Returns:
            Dict with success status, screenshot path, and any errors
        """
        result = {"success": False, "url": url, "errors": [], "checks": []}

        # Safely serialize parameters as JSON to prevent injection
        url_json = json.dumps(url)
        screenshot_path_json = json.dumps(screenshot_path) if screenshot_path else "null"
        checks_json = json.dumps(checks or [])

        # Playwright script to run
        script = f'''
const {{ chromium }} = require('playwright');

(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();
    const result = {{ success: false, title: '', errors: [], checks: [] }};

    // Safely parse parameters from JSON
    const url = {url_json};
    const screenshotPath = {screenshot_path_json};
    const checks = {checks_json};

    try {{
        const response = await page.goto(url, {{ timeout: 10000, waitUntil: 'networkidle' }});
        result.status = response ? response.status() : 0;
        result.title = await page.title();
        result.success = result.status >= 200 && result.status < 400;

        // Take screenshot if path provided
        if (screenshotPath) {{
            await page.screenshot({{ path: screenshotPath, fullPage: true }});
        }}

        // Check for specific elements/text
        for (const check of checks) {{
            try {{
                const found = await page.locator(`text=${{check}}`).count() > 0
                           || await page.locator(check).count() > 0;
                result.checks.push({{ item: check, found: found }});
            }} catch (e) {{
                result.checks.push({{ item: check, found: false, error: e.message }});
            }}
        }}

        // Get page content summary
        result.content = await page.evaluate(() => {{
            return document.body ? document.body.innerText.substring(0, 500) : '';
        }});

    }} catch (e) {{
        result.errors.push(e.message);
    }} finally {{
        await browser.close();
    }}

    console.log(JSON.stringify(result));
}})();
'''
        
        try:
            proc = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=os.getcwd()
            )
            
            if proc.returncode == 0 and proc.stdout.strip():
                output = json.loads(proc.stdout.strip().split('\n')[-1])
                result.update(output)
            else:
                result["errors"].append(proc.stderr[:200] if proc.stderr else "Unknown error")
                
        except subprocess.TimeoutExpired:
            result["errors"].append("Browser verification timed out")
        except json.JSONDecodeError as e:
            result["errors"].append(f"Failed to parse browser output: {e}")
        except FileNotFoundError:
            result["errors"].append("Node.js or Playwright not installed")
        except Exception as e:
            result["errors"].append(str(e)[:100])
        
        return result
    
    @staticmethod
    def take_snapshot(url: str) -> Dict[str, Any]:
        """Take accessibility snapshot of page for verification."""
        # Safely serialize URL as JSON to prevent injection
        url_json = json.dumps(url)

        script = f'''
const {{ chromium }} = require('playwright');

(async () => {{
    const browser = await chromium.launch({{ headless: true }});
    const page = await browser.newPage();

    // Safely parse URL from JSON
    const url = {url_json};

    try {{
        await page.goto(url, {{ timeout: 10000, waitUntil: 'domcontentloaded' }});
        const snapshot = await page.accessibility.snapshot();
        console.log(JSON.stringify({{ success: true, snapshot: snapshot }}));
    }} catch (e) {{
        console.log(JSON.stringify({{ success: false, error: e.message }}));
    }} finally {{
        await browser.close();
    }}
}})();
'''
        
        try:
            proc = subprocess.run(
                ["node", "-e", script],
                capture_output=True,
                text=True,
                timeout=15
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return json.loads(proc.stdout.strip().split('\n')[-1])
        except Exception as e:
            return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Unknown error"}


# =============================================================================
# Data Structures
# =============================================================================

class FunctionName(str, Enum):
    """Available functions that PA can call."""

    NO_OP = "no_op"
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
    SAVE_SESSION = "save_session"


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


# =============================================================================
# Function Declarations (available to PA)
# =============================================================================

FUNCTION_DECLARATIONS: List[FunctionDeclaration] = [
    FunctionDeclaration(
        name=FunctionName.SEND_TO_CLAUDE,
        description="Send an instruction or follow-up to Claude Code. Use this to guide Claude's next action.",
        parameters={
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "The instruction to send to Claude"},
                "context": {"type": "string", "description": "Optional context about why this instruction"}
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
                "test_command": {"type": "string", "description": "Test command to run (e.g., 'pytest')"}
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
                "url": {"type": "string", "description": "URL to check (e.g., http://localhost:5000)"}
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
                "file_path": {"type": "string", "description": "Path to the file to read"}
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
                "summary": {"type": "string", "description": "Summary of what was accomplished"},
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
                "description": {"type": "string", "description": "What needs to be done"},
                "assignee": {"type": "string", "enum": ["claude", "pa"], "description": "Who should do this task"},
                "priority": {"type": "integer", "description": "Priority (0=highest)"}
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
                "task_id": {"type": "string", "description": "Task ID to update"},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "blocked", "completed", "cancelled"],
                    "description": "New status"
                },
                "notes": {"type": "string", "description": "Optional notes about the update"}
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
                "task_id": {"type": "string", "description": "Task ID to mark complete"},
                "notes": {"type": "string", "description": "Completion notes"}
            },
            "required": ["task_id"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.REVIEW_CHANGES,
        description="Review code changes made by Claude. Reads changed files and performs QA review.",
        parameters={
            "type": "object",
            "properties": {
                "file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to review"
                },
                "context": {"type": "string", "description": "Context about what Claude was trying to accomplish"}
            },
            "required": ["file_paths"]
        }
    ),
    FunctionDeclaration(
        name=FunctionName.VERIFY_PRODUCT,
        description="Verify the final product works. For APIs: tests endpoints. For UIs: uses browser automation.",
        parameters={
            "type": "object",
            "properties": {
                "product_type": {
                    "type": "string",
                    "enum": ["api_server", "ui_app", "script", "auto"],
                    "description": "Type of product to verify. Use 'auto' to detect."
                },
                "start_command": {"type": "string", "description": "Command to start the server"},
                "port": {"type": "integer", "description": "Port the server runs on (default: 5000)"},
                "endpoints_to_test": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For API: list of endpoints to test"
                },
                "ui_actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "For UI: list of actions to perform"
                }
            },
            "required": []
        }
    ),
    FunctionDeclaration(
        name=FunctionName.SAVE_SESSION,
        description="Save current session state and exit gracefully. Use when PA encounters repeated failures that prevent progress.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Why session is being saved"},
                "error_type": {"type": "string", "description": "Type of error that triggered save"},
                "status_code": {"type": "integer", "description": "HTTP status code if applicable"}
            },
            "required": ["reason"]
        }
    ),
]


# =============================================================================
# Function Executor
# =============================================================================

class FunctionExecutor:
    """
    Executes functions called by PA during the agent loop.
    
    Each function runs independently and returns a FunctionResult.
    The executor maintains a queue for Claude instructions.
    """
    
    def __init__(self, working_dir: str = ".", memory: Any = None) -> None:
        """
        Initialize executor.
        
        Args:
            working_dir: Working directory for file operations.
            memory: Reference to PAMemory for task management.
        """
        self.working_dir = working_dir
        self._memory = memory
        self._claude_queue: Queue = Queue()
    
    def execute(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call and return the result.

        Args:
            call: The function call to execute.

        Returns:
            FunctionResult with success status and output.
        """
        telemetry = get_telemetry()

        # Start OTEL span for function execution if enabled
        if telemetry.enabled and telemetry.tracer:
            function_span = telemetry.tracer.start_span(
                f"pa.function.{call.name.value}",
                attributes={"pa.function.name": call.name.value}
            )
        else:
            function_span = None

        try:
            handlers: Dict[FunctionName, Callable] = {
                FunctionName.NO_OP: self._no_op,
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
                FunctionName.SAVE_SESSION: self._save_session,
            }

            handler = handlers.get(call.name)
            if not handler:
                result = FunctionResult(
                    name=call.name,
                    success=False,
                    output=f"Unknown function: {call.name}"
                )
                if function_span:
                    function_span.set_attribute("pa.function.success", False)
                    function_span.end()
                return result

            # Execute the handler
            result = handler(call.arguments)

            # Record verification metrics
            if call.name in [FunctionName.VERIFY_CODE, FunctionName.RUN_TESTS, FunctionName.VERIFY_PRODUCT]:
                if telemetry.enabled:
                    telemetry.verifications.add(1, {
                        "type": call.name.value,
                        "result": "pass" if result.success else "fail"
                    })

            if function_span:
                function_span.set_attribute("pa.function.success", result.success)
                function_span.end()

            return result

        except Exception as e:
            if function_span:
                try:
                    from opentelemetry import trace as otel_trace
                    function_span.set_status(otel_trace.Status(otel_trace.StatusCode.ERROR, str(e)))
                    function_span.record_exception(e)
                except ImportError:
                    pass
                function_span.end()
            return FunctionResult(
                name=call.name,
                success=False,
                output=f"Execution error: {str(e)[:200]}"
            )
    
    # =========================================================================
    # No-Op (fallback when Gemini fails)
    # =========================================================================
    
    def _no_op(self, args: Dict[str, Any]) -> FunctionResult:
        """No operation - used when Gemini fails to avoid overwriting queued instructions."""
        reason = args.get("reason", "No action needed")
        return FunctionResult(
            name=FunctionName.NO_OP,
            success=True,
            output=f"[no_op] {reason}",
            metadata={"reason": reason}
        )
    
    # =========================================================================
    # Claude Communication
    # =========================================================================
    
    def _send_to_claude(self, args: Dict[str, Any]) -> FunctionResult:
        """Queue instruction for Claude with screenshot context if available."""
        instruction = args.get("instruction", "")
        context = args.get("context", "")
        
        # Add screenshot context to help Claude understand visual requirements
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
        """Build description of reference screenshots for Claude."""
        if not self._memory or not self._memory.session.reference_screenshots:
            return ""
        
        screenshots = self._memory.session.reference_screenshots
        if not screenshots:
            return ""
        
        lines = ["## VISUAL REFERENCE (Screenshots provided by user)"]
        lines.append("The user has provided reference images the result should match:")
        lines.append("")
        
        for i, ss in enumerate(screenshots, 1):
            desc = ss.description or "UI design reference"
            path = ss.path
            filename = Path(path).name if path else "unknown"
            
            lines.append(f"**Screenshot {i}: {filename}**")
            lines.append(f"  - Description: {desc}")
            lines.append(f"  - Path: {path}")
            lines.append("")
        
        lines.append("IMPORTANT: Match the visual design from these screenshots.")
        return "\n".join(lines)
    
    def get_pending_claude_instruction(self) -> Optional[Dict[str, Any]]:
        """Get next queued instruction for Claude, if any."""
        try:
            return self._claude_queue.get_nowait()
        except Empty:
            return None
    
    # =========================================================================
    # Code Verification
    # =========================================================================
    
    def _verify_code(self, args: Dict[str, Any]) -> FunctionResult:
        """Execute Python files to verify they work."""
        file_paths = args.get("file_paths", [])
        results = []

        for filepath in file_paths:
            full_path = self._resolve_path(filepath)

            if not os.path.exists(full_path):
                results.append(f"{filepath}: FILE NOT FOUND")
                continue

            try:
                # Use relative path since we set cwd
                rel_path = os.path.relpath(full_path, self.working_dir)
                result = subprocess.run(
                    ["python3", rel_path],
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
        """Check if server is running at URL."""
        url = args.get("url", "http://localhost:5000")
        
        try:
            req = urllib.request.Request(url, method="GET")
            req.add_header("User-Agent", "PA-Agent/1.0")
            
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
                success=True,  # Server responded
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
        full_path = self._resolve_path(file_path)
        
        if not os.path.exists(full_path):
            return FunctionResult(
                name=FunctionName.READ_FILE,
                success=False,
                output=f"File not found: {file_path}"
            )
        
        try:
            with open(full_path, "r") as f:
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
    
    # =========================================================================
    # Task Management
    # =========================================================================
    
    def _mark_done(self, args: Dict[str, Any]) -> FunctionResult:
        """Mark task as complete."""
        summary = args.get("summary", "Task completed")
        verified_items = args.get("verified_items", [])

        return FunctionResult(
            name=FunctionName.MARK_DONE,
            success=True,
            output=f"TASK COMPLETE: {summary}",
            metadata={"summary": summary, "verified_items": verified_items, "done": True}
        )

    def _save_session(self, args: Dict[str, Any]) -> FunctionResult:
        """
        Save session state and signal graceful exit.

        This is used when PA encounters repeated failures (e.g., Gemini API errors)
        that prevent it from making progress. The session state is saved so it can
        be resumed later.
        """
        reason = args.get("reason", "Unknown reason")
        error_type = args.get("error_type", "unknown")
        status_code = args.get("status_code")

        # Save session state via memory
        if self._memory:
            try:
                self._memory.session.save()
                session_id = self._memory.session.session_id
            except Exception as e:
                return FunctionResult(
                    name=FunctionName.SAVE_SESSION,
                    success=False,
                    output=f"Failed to save session: {e}",
                    metadata={"error": str(e)}
                )
        else:
            session_id = None

        # Format output message
        output_lines = [
            "[PA] Session saved due to repeated failures",
            f"Reason: {reason}",
            f"Error type: {error_type}",
        ]
        if status_code:
            output_lines.append(f"Status code: {status_code}")
        if session_id:
            output_lines.append(f"Session ID: {session_id}")
            output_lines.append(f"\nTo resume this session, use: pa --session-id {session_id}")

        return FunctionResult(
            name=FunctionName.SAVE_SESSION,
            success=True,
            output="\n".join(output_lines),
            metadata={
                "reason": reason,
                "error_type": error_type,
                "status_code": status_code,
                "session_id": session_id,
                "done": True,  # Signal PA to exit
                "exit_gracefully": True,
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
    
    # =========================================================================
    # Code Review
    # =========================================================================
    
    def _review_changes(self, args: Dict[str, Any]) -> FunctionResult:
        """Review code changes made by Claude using Gemini for QA."""
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
        file_contents = self._read_multiple_files(file_paths)
        
        # Build and execute review
        review_prompt = self._build_review_prompt(file_contents, context)
        
        try:
            gemini = GeminiClient()
            review_result = gemini.call(
                system_prompt=self._get_review_system_prompt(),
                user_prompt=review_prompt
            )
            has_issues = gemini.analyze_review_issues(review_result)
        except Exception as e:
            return FunctionResult(
                name=FunctionName.REVIEW_CHANGES,
                success=False,
                output=f"Review failed: {str(e)[:200]}",
                metadata={"error": str(e)}
            )
        
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
    
    def _read_multiple_files(self, file_paths: List[str]) -> Dict[str, str]:
        """Read contents of multiple files."""
        file_contents = {}
        for filepath in file_paths:
            full_path = self._resolve_path(filepath)
            
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r") as f:
                        content = f.read()
                    if len(content) > 8000:
                        content = content[:8000] + "\n... [truncated]"
                    file_contents[filepath] = content
                except Exception as e:
                    file_contents[filepath] = f"[Error reading: {e}]"
            else:
                file_contents[filepath] = "[File not found]"
        
        return file_contents
    
    def _build_review_prompt(self, file_contents: Dict[str, str], context: str) -> str:
        """Build the review prompt with all file contents."""
        parts = [f"## Context\n{context}\n"]
        parts.append(f"## Files Changed ({len(file_contents)} files)\n")
        
        for filepath, content in file_contents.items():
            parts.append(f"### {filepath}\n```\n{content}\n```\n")
        
        parts.append("\n## Review Request")
        parts.append("CRITICAL ISSUES ONLY - Review for:")
        parts.append("1. **ERRORS**: Syntax errors, runtime errors, bugs that will CRASH")
        parts.append("2. **SECURITY**: SQL injection, XSS, authentication bypasses")
        parts.append("")
        parts.append("DO NOT report: style issues, minor improvements, type hints.")
        parts.append("If code works correctly, respond with 'NO CRITICAL ISSUES'.")
        
        return "\n".join(parts)
    
    def _get_review_system_prompt(self) -> str:
        """Get the system prompt for CRITICAL-only code review."""
        return """You are a strict code reviewer focused ONLY on critical issues.

REPORT ONLY:
1. ERRORS: Syntax errors, runtime errors, bugs that CRASH functionality
2. SECURITY: SQL injection, XSS, auth bypasses, sensitive data exposure

DO NOT REPORT: Style suggestions, performance improvements, type hints, documentation.

If the code will RUN CORRECTLY and is SECURE, respond: "NO CRITICAL ISSUES"

Format:
## CRITICAL ERRORS
- [only errors that break functionality, or "None"]

## SECURITY ISSUES
- [only security vulnerabilities, or "None"]

## VERDICT
[PASS/FAIL] - [one line: will it work?]"""
    
    # =========================================================================
    # Product Verification
    # =========================================================================
    
    def _verify_product(self, args: Dict[str, Any]) -> FunctionResult:
        """Verify the final product works correctly."""
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
        """Detect what type of product was built based on project files."""
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
        
        # Check requirements.txt for Python frameworks
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
        
        # Check for index.html
        if os.path.exists(os.path.join(self.working_dir, "index.html")):
            indicators.append("ui_app")
        
        # Return based on indicators
        if "ui_app" in indicators:
            return "ui_app"
        elif "api_server" in indicators:
            return "api_server"
        else:
            return "script"
    
    def _verify_api_server(
        self,
        start_command: Optional[str],
        port: int,
        endpoints: List[str]
    ) -> FunctionResult:
        """Start API server and test endpoints."""
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
                time.sleep(3)
                results.append(f"[PA Verify] Started server: {start_command}")
            
            # Default endpoints
            if not endpoints:
                endpoints = ["/health", "/", "/api"]
            
            base_url = f"http://localhost:{port}"
            success_count = 0
            
            for endpoint in endpoints:
                url = f"{base_url}{endpoint}"
                try:
                    req = urllib.request.Request(url, method="GET")
                    req.add_header("User-Agent", "PA-Verify/1.0")
                    
                    with urllib.request.urlopen(req, timeout=5) as response:
                        status = response.status
                        body = response.read().decode("utf-8")[:200]
                        results.append(f"[PA Verify] {endpoint}: {status} OK - {body[:100]}")
                        success_count += 1
                except urllib.error.HTTPError as e:
                    results.append(f"[PA Verify] {endpoint}: HTTP {e.code} - {e.reason}")
                    if e.code < 500:
                        success_count += 1
                except Exception as e:
                    results.append(f"[PA Verify] {endpoint}: FAILED - {str(e)[:50]}")
            
            overall_success = success_count > 0
            results.append(f"\n[PA Verify] API Test: {success_count}/{len(endpoints)} endpoints responding")
            
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=overall_success,
                output="\n".join(results),
                metadata={"type": "api_server", "endpoints_tested": len(endpoints)}
            )
            
        finally:
            if server_process:
                server_process.terminate()
                try:
                    server_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    server_process.kill()
    
    def _verify_ui_app(
        self,
        start_command: Optional[str],
        port: int,
        ui_actions: List[str]
    ) -> FunctionResult:
        """Start UI app and verify with browser automation."""
        server_process = None
        results = []
        
        try:
            if start_command:
                server_process = subprocess.Popen(
                    start_command,
                    shell=True,
                    cwd=self.working_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                time.sleep(5)
                results.append(f"[PA Verify] Started UI server: {start_command}")
            
            base_url = f"http://localhost:{port}"
            results.append(f"[PA Verify] Testing UI at {base_url}")
            
            # Actually verify with browser
            screenshot_dir = Path(self.working_dir) / "project_memory"
            screenshot_dir.mkdir(exist_ok=True)
            screenshot_path = str(screenshot_dir / f"verify_{int(time.time())}.png")
            
            browser_result = BrowserVerifier.verify_url(
                url=base_url,
                screenshot_path=screenshot_path,
                checks=ui_actions or None
            )
            
            if browser_result.get("success"):
                results.append(f"[PA Verify] ✓ Page loaded: {browser_result.get('title', 'No title')}")
                results.append(f"[PA Verify] ✓ Status: {browser_result.get('status', '?')}")
                results.append(f"[PA Verify] ✓ Screenshot saved: {screenshot_path}")
                
                # Report check results
                for check in browser_result.get("checks", []):
                    status = "✓" if check.get("found") else "✗"
                    results.append(f"[PA Verify] {status} Check '{check.get('item')}': {'Found' if check.get('found') else 'Not found'}")
                
                # Show content preview
                content = browser_result.get("content", "")[:200]
                if content:
                    results.append(f"[PA Verify] Content preview: {content}...")
                
                overall_success = True
            else:
                errors = browser_result.get("errors", ["Unknown error"])
                results.append(f"[PA Verify] ✗ Browser verification failed: {errors}")
                overall_success = False
            
            return FunctionResult(
                name=FunctionName.VERIFY_PRODUCT,
                success=overall_success,
                output="\n".join(results),
                metadata={"type": "ui_app", "url": base_url, "screenshot": screenshot_path, "browser_result": browser_result}
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
        script_files = []

        # Try multiple glob patterns to find scripts
        patterns = [
            os.path.join(self.working_dir, "**", "*.py"),
            os.path.join(self.working_dir, "**", "*.js"),
            os.path.join(self.working_dir, "**", "*.ts"),
            os.path.join(self.working_dir, "*.py"),
            os.path.join(self.working_dir, "*.js"),
            os.path.join(self.working_dir, "*.ts"),
        ]

        for pattern in patterns:
            found = glob.glob(pattern, recursive=True)
            script_files.extend(found)

        # Remove duplicates
        script_files = list(set(script_files))

        # Exclude common directories that shouldn't be executed
        excluded_dirs = {"venv", "node_modules", ".git", "__pycache__", "dist", "build", ".egg-info", ".pa_sessions", "context"}
        script_files = [
            f for f in script_files
            if not any(excluded in f for excluded in excluded_dirs)
        ]

        # Filter to keep only executable scripts (not library modules)
        # Look for files with __main__ guard or in top-level directory
        executable_scripts = []
        for f in script_files:
            # Skip __init__.py files as they're not meant to be run directly
            if f.endswith("__init__.py"):
                continue

            # Check if it's in the top level (likely a standalone script)
            rel_path = os.path.relpath(f, self.working_dir)
            if "/" not in rel_path and "\\" not in rel_path:
                executable_scripts.append(f)
                continue

            # Check if file has __name__ == "__main__" guard
            try:
                with open(f, "r") as file:
                    content = file.read()
                    if '__name__' in content and '__main__' in content:
                        executable_scripts.append(f)
            except Exception:
                pass

        # Use executable scripts if found, otherwise use all (best effort)
        script_files = executable_scripts if executable_scripts else script_files

        if not script_files:
            # Check if directory is truly empty or just has no scripts
            all_files = []
            for root, dirs, files in os.walk(self.working_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                all_files.extend(files)

            if not all_files:
                return FunctionResult(
                    name=FunctionName.VERIFY_PRODUCT,
                    success=True,
                    output="[PA Verify] Working directory is empty - verification skipped (waiting for files to be created)",
                    metadata={"type": "script", "scripts_tested": 0, "note": "empty_directory"}
                )
            else:
                return FunctionResult(
                    name=FunctionName.VERIFY_PRODUCT,
                    success=True,
                    output=f"[PA Verify] No executable scripts found to verify (found {len(all_files)} files - likely a library/package project)",
                    metadata={"type": "script", "scripts_tested": 0, "note": "library_project", "total_files": len(all_files)}
                )

        results = []
        for script in script_files[:3]:
            try:
                # Make path relative to working_dir since we set cwd
                rel_path = os.path.relpath(script, self.working_dir)

                cmd = ["python3", rel_path] if rel_path.endswith(".py") else ["node", rel_path]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=self.working_dir
                )
                
                basename = os.path.basename(script)
                if result.returncode == 0:
                    output = result.stdout[:200] if result.stdout else "OK"
                    results.append(f"[PA Verify] {basename}: SUCCESS\n{output}")
                else:
                    error = result.stderr[:200] if result.stderr else "Unknown error"
                    results.append(f"[PA Verify] {basename}: FAILED\n{error}")
            except Exception as e:
                results.append(f"[PA Verify] {os.path.basename(script)}: ERROR - {str(e)[:100]}")
        
        all_success = all("SUCCESS" in r for r in results)
        return FunctionResult(
            name=FunctionName.VERIFY_PRODUCT,
            success=all_success,
            output="\n".join(results),
            metadata={"type": "script", "scripts_tested": len(results)}
        )
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _resolve_path(self, filepath: str) -> str:
        """Resolve a filepath to absolute path relative to working_dir."""
        if os.path.isabs(filepath):
            return filepath
        return os.path.join(self.working_dir, filepath)
