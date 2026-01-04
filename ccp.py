"""
CCP - Code Custodian Persona
============================

Unified AI agent that supervises Claude Code with three capabilities:
- Thinking: Reasoning about Claude's actions and deciding next steps
- QA: Generating and running tests
- Verification: Executing and validating created code

Three-tier memory system:
1. Best Practices (Tier 1): Static rules from prompts/*.md
2. Session Context (Tier 2): High-level mission, persists hours/days
3. Interaction History (Tier 3): Rolling window of Claude interactions

Usage:
    ccp = CCP(working_dir="./myproject", user_mission="Build a REST API")
    for event in ccp.run_task("Create the user endpoints"):
        print(event)
"""

import base64
import json
import os
import re
import subprocess
import time
import webbrowser
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv

from ccp_memory import CCPMemory
from models import OutputEvent, EventType, ControllerState
from process_manager import ClaudeProcessManager
from display import RealtimeDisplay, create_display

# Load environment variables
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class ThinkingResult:
    """Result from CCP-Thinking analysis."""
    action: str  # CONTINUE, VERIFY, DONE
    reasoning: str
    follow_up: str = ""  # If CONTINUE, what to tell Claude
    
    @property
    def should_continue(self) -> bool:
        return self.action == "CONTINUE"
    
    @property
    def should_verify(self) -> bool:
        return self.action == "VERIFY"
    
    @property
    def is_done(self) -> bool:
        return self.action == "DONE"


@dataclass
class VerificationResult:
    """Result from CCP-Verify."""
    success: bool
    execution_output: str
    analysis: str
    files_tested: List[str] = field(default_factory=list)


@dataclass
class QAResult:
    """Result from CCP-QA review."""
    prompt: str
    claude_response: str = ""
    issues_found: List[str] = field(default_factory=list)
    passed: bool = False


@dataclass
class CCPConfig:
    """Configuration for CCP agent."""
    working_dir: str = "."
    display_mode: str = "rich"
    task_timeout: float = 300.0
    max_conversation_turns: int = 10
    auto_verify: bool = True
    auto_qa: bool = True


# =============================================================================
# Gemini API Client (for CCP-Thinking)
# =============================================================================

class GeminiClient:
    """
    Client for Gemini API - powers CCP's reasoning capabilities.
    Supports text and image inputs.
    """
    
    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    
    # Supported image formats for Gemini
    SUPPORTED_IMAGE_TYPES = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
    
    def _encode_image(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Encode image file to base64 for Gemini API."""
        path = Path(image_path)
        if not path.exists():
            return None
        
        ext = path.suffix.lower()
        mime_type = self.SUPPORTED_IMAGE_TYPES.get(ext)
        if not mime_type:
            return None
        
        try:
            with open(path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": image_data
                }
            }
        except Exception:
            return None
    
    def call(self, system_prompt: str, user_prompt: str, image_paths: Optional[List[str]] = None) -> str:
        """Make API call to Gemini with optional images."""
        url = f"{self.API_URL}?key={self.api_key}"
        
        # Build parts: system prompt, then images, then user prompt
        parts = [{"text": system_prompt}]
        
        # Add images if provided
        if image_paths:
            for img_path in image_paths:
                img_part = self._encode_image(img_path)
                if img_part:
                    parts.append(img_part)
        
        parts.append({"text": user_prompt})
        
        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 8192,
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                candidates = result.get("candidates", [])
                if candidates:
                    content = candidates[0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                
                return "[No response from Gemini]"
                
        except urllib.error.URLError:
            return "[Network error]"
        except json.JSONDecodeError:
            return "[Invalid response]"
        except Exception as e:
            return f"[Error: {str(e)[:50]}]"


# =============================================================================
# CCP Agent
# =============================================================================

class CCP:
    """
    Unified CCP Agent - Long-running supervisor for Claude Code.
    
    Combines:
    - CCP-Thinking: Skeptical analysis of Claude's output
    - CCP-QA: Test generation and execution
    - CCP-Verify: Code execution and validation
    
    With three-tier memory:
    - Tier 1: Best practices from prompts/*.md
    - Tier 2: Session context (mission, constraints, criteria)
    - Tier 3: Interaction history (rolling window)
    """
    
    # System prompt for CCP-Thinking
    THINKING_PROMPT = """You are CCP-thinking, a SKEPTICAL human supervisor overseeing Claude Code.

NEVER be easily satisfied. Assume Claude's work:
- Is incomplete until VERIFIED with actual execution
- May have bugs, edge cases, missing error handling
- Claims "done" but hasn't actually tested it

Your mindset: "Trust but verify. Show me it works, don't tell me."

OUTPUT FORMAT (MUST follow exactly):
[ACTION:CONTINUE|VERIFY|DONE] Your skeptical observation

ACTION meanings:
- CONTINUE = Claude needs to do more work, send follow-up instruction
- VERIFY = Claude claims done, but needs testing/verification
- DONE = Actually verified and working (rare - be skeptical!)

Examples:
- "[ACTION:VERIFY] Claude says calculator works - test divide by zero and edge cases."
- "[ACTION:CONTINUE] File created but no error handling - tell Claude to add try/catch."
- "[ACTION:VERIFY] Claims app running - run actual tests before trusting."
- "[ACTION:CONTINUE] Only happy path tested - tell Claude to handle empty inputs."
- "[ACTION:DONE] Code tested, edge cases handled, errors caught - actually verified."

Default to VERIFY or CONTINUE. DONE is rare. Max 30 words after action tag.
"""
    
    # System prompt for CCP-NextStep: comprehensive review after Claude exits
    NEXT_STEP_PROMPT = """You are CCP-NextStep, a critical checkpoint that activates EVERY TIME Claude exits.

Your job: Review ALL context and make a CRITICAL DECISION about whether the job is truly done.

## REVIEW CHECKLIST
1. ORIGINAL MISSION: What was the user's full intent?
2. CLAUDE'S WORK: What did Claude actually accomplish across all steps?
3. THINKING HISTORY: What did CCP-thinking observe/flag during execution?
4. VERIFICATION STATUS: Was the code actually run and tested?
5. REMAINING ITEMS: What parts of the mission are still incomplete?

## DECISION CRITERIA
- DONE: ALL aspects of mission verified working with actual output/proof
- CONTINUE: Any part of mission incomplete, untested, or needs more work
- VERIFY: Claims made but not proven with actual execution

## OUTPUT FORMAT
[ACTION:DONE|CONTINUE|VERIFY]
<If CONTINUE: Specific prompt for next Claude session summarizing remaining work>
<If VERIFY: Specific test/verification to run>
<If DONE: Brief summary of what was accomplished>

## BE SKEPTICAL
Claude often exits after:
- Creating files but not running them
- Testing partial functionality, not full scope
- Claiming success without actual output proof
- Completing one step but missing the bigger picture

Default to CONTINUE or VERIFY. DONE is rare and requires actual proof.
"""
    
    # System prompt for QA review
    QA_PROMPT = """You are CCP-QA-Reviewer, a quality assurance subagent.

Your job is to review completed code and draft a prompt for Claude to test it.

Based on the context provided, create a SHORT, ACTIONABLE prompt that tells Claude to:
1. Start/run the application
2. Test the main features
3. Report any issues found

Output ONLY the prompt text (1-3 sentences). No explanations.

Examples:
- "Run calculator.py and test add(2,3), subtract(5,2), multiply(3,4), divide(10,2). Report results."
- "Start the Flask app with 'python app.py' and test the /hello endpoint. Verify response."
- "Execute test_utils.py and confirm all tests pass. Fix any failures."
"""
    
    def __init__(
        self,
        working_dir: str = ".",
        session_id: Optional[str] = None,
        user_mission: Optional[str] = None,
        display_mode: str = "rich",
        auto_verify: bool = True,
        auto_qa: bool = True,
    ):
        """
        Initialize CCP agent.
        
        Args:
            working_dir: Working directory for Claude operations
            session_id: Resume existing session (or create new)
            user_mission: High-level mission description
            display_mode: "rich", "simple", "json", or "quiet"
            auto_verify: Auto-verify after Claude claims done
            auto_qa: Auto-run QA review after verification
        """
        self.config = CCPConfig(
            working_dir=os.path.abspath(working_dir),
            display_mode=display_mode,
            auto_verify=auto_verify,
            auto_qa=auto_qa,
        )
        
        # Initialize memory system
        self.memory = CCPMemory(
            working_dir=working_dir,
            session_id=session_id,
        )
        
        if user_mission:
            self.memory.session.set_mission(user_mission)
        
        # Initialize components
        self._display: Optional[RealtimeDisplay] = None
        self._claude: Optional[ClaudeProcessManager] = None
        self._gemini: Optional[GeminiClient] = None
        self._state = ControllerState.IDLE
        
        # Try to initialize Gemini client
        try:
            self._gemini = GeminiClient()
        except ValueError:
            pass  # Will work without Gemini, just no thinking
    
    @property
    def state(self) -> ControllerState:
        return self._state
    
    @property
    def session_id(self) -> str:
        return self.memory.session.session_id
    
    def _get_screenshot_paths(self) -> List[str]:
        """Get list of reference screenshot paths from session."""
        return [s.path for s in self.memory.session.reference_screenshots if Path(s.path).exists()]
    
    # =========================================================================
    # CCP-Thinking: Analyze Claude's output
    # =========================================================================
    
    def think(self, content: str, event_type: str, task: str) -> ThinkingResult:
        """
        CCP-Thinking: Analyze Claude's output and decide action.
        
        Uses all three memory tiers:
        - Tier 1: Rules for what to check
        - Tier 2: Mission/criteria to evaluate against
        - Tier 3: Recent history for context
        
        Returns:
            ThinkingResult with action (CONTINUE/VERIFY/DONE) and reasoning
        """
        if not self._gemini:
            return ThinkingResult(action="VERIFY", reasoning="[Gemini not available]")
        
        if not content or len(content.strip()) < 5:
            return ThinkingResult(action="CONTINUE", reasoning="[Awaiting substantive output]")
        
        # Build context from all memory tiers
        context_parts = []
        
        # Tier 2: Session context
        context_parts.append(f"MISSION: {self.memory.session.user_mission or task}")
        
        if self.memory.session.acceptance_criteria:
            context_parts.append(f"CRITERIA: {', '.join(self.memory.session.acceptance_criteria)}")
        
        # Reference screenshots context
        screenshots = self._get_screenshot_paths()
        if screenshots:
            screenshot_descriptions = [f"- {s.path}: {s.description}" for s in self.memory.session.reference_screenshots]
            context_parts.append("REFERENCE SCREENSHOTS (attached as images):\n" + "\n".join(screenshot_descriptions))
        
        # Tier 3: Recent history
        history = self.memory.history.get_history_for_llm(max_chars=1000)
        if history:
            history_str = "\n".join([
                f"- {h.get('type', 'event')}: {h.get('content', '')[:100]}"
                for h in history[-5:]
            ])
            context_parts.append(f"RECENT HISTORY:\n{history_str}")
        
        # Current event
        context_parts.append(f"CURRENT EVENT ({event_type}): {content[:400]}")
        context_parts.append("\nAs human supervisor's representative, what should they know?")
        
        prompt = "\n\n".join(context_parts)
        
        try:
            # Include reference screenshots in the Gemini call
            response = self._gemini.call(self.THINKING_PROMPT, prompt, image_paths=screenshots)
            action, reasoning = self._parse_thinking_action(response)
            
            # Record decision in Tier 3
            self.memory.history.add_decision(action, reasoning)
            
            return ThinkingResult(action=action, reasoning=reasoning)
            
        except Exception as e:
            return ThinkingResult(action="VERIFY", reasoning=f"[Error: {str(e)[:30]}]")
    
    def _parse_thinking_action(self, response: str) -> Tuple[str, str]:
        """Parse action tag from thinking response."""
        match = re.search(r'\[ACTION:(CONTINUE|VERIFY|DONE)\]', response)
        
        if match:
            action = match.group(1)
            message = response[match.end():].strip()
            return action, message
        
        return "VERIFY", response
    
    def next_step(self, task: str, iteration: int) -> Tuple[str, str]:
        """
        CCP-NextStep: Comprehensive review after Claude exits.
        
        Reviews:
        - Original mission and full scope
        - All past thinking observations
        - All Claude execution steps
        - Memory context
        
        Returns:
            Tuple of (action, instruction) where action is DONE/CONTINUE/VERIFY
            and instruction is the next prompt or summary.
        """
        if not self._gemini:
            # Never default to DONE - require verification when Gemini unavailable
            return ("VERIFY", "Gemini unavailable - manual verification required")
        
        # Build comprehensive context from all sources
        context_parts = []
        
        # 1. Original mission and task
        context_parts.append(f"## ORIGINAL MISSION\n{self.memory.session.user_mission or 'Not specified'}")
        context_parts.append(f"## CURRENT TASK\n{task}")
        
        # 2. Acceptance criteria if any
        if self.memory.session.acceptance_criteria:
            context_parts.append("## ACCEPTANCE CRITERIA\n" + "\n".join(
                f"- {c}" for c in self.memory.session.acceptance_criteria
            ))
        
        # 2b. Reference screenshots
        screenshots = self._get_screenshot_paths()
        if screenshots:
            screenshot_descriptions = [f"- {s.path}: {s.description}" for s in self.memory.session.reference_screenshots]
            context_parts.append("## REFERENCE SCREENSHOTS (attached as images)\n" + "\n".join(screenshot_descriptions))
        
        # 3. Full history of Claude steps and CCP thinking
        history = self.memory.history.get_history_for_llm(max_chars=4000)
        if history:
            # Separate Claude actions from CCP thinking
            claude_steps = []
            thinking_observations = []
            
            for h in history:
                h_type = h.get('type', 'event')
                h_content = h.get('content', '')[:200]
                
                if 'decision' in h_type or 'thinking' in h_type.lower():
                    thinking_observations.append(f"- {h_content}")
                else:
                    claude_steps.append(f"- {h_content}")
            
            if claude_steps:
                context_parts.append("## CLAUDE'S WORK (all steps)\n" + "\n".join(claude_steps[-15:]))
            
            if thinking_observations:
                context_parts.append("## CCP-THINKING OBSERVATIONS\n" + "\n".join(thinking_observations[-10:]))
        
        # 4. Files created/modified
        if self.memory.history.files_created:
            context_parts.append("## FILES CREATED\n" + "\n".join(
                f"- {f}" for f in self.memory.history.files_created
            ))
        
        # 5. Verification history if any
        verifications = [e for e in self.memory.history.events if 'verification' in e.event_type.lower()]
        if verifications:
            context_parts.append("## VERIFICATION RESULTS\n" + "\n".join(
                f"- {v.content[:150]}" for v in verifications[-5:]
            ))
        
        # 6. Iteration context
        context_parts.append(f"## ITERATION\nThis is iteration {iteration}. Claude just exited.")
        
        prompt = "\n\n".join(context_parts)
        prompt += "\n\n## YOUR DECISION\nBased on the above, what is the critical next step?"
        
        try:
            response = self._gemini.call(self.NEXT_STEP_PROMPT, prompt, image_paths=screenshots)
            action, instruction = self._parse_thinking_action(response)
            
            # Record the decision in history
            self.memory.history.add_decision(f"NextStep-{action}", instruction)
            
            return (action, instruction)
        except Exception as e:
            return ("VERIFY", f"Run the created files to verify: {str(e)[:30]}")
    
    def supervise(self, task: str) -> Tuple[str, str]:
        """Deprecated: Use next_step() instead. Kept for compatibility."""
        return self.next_step(task, iteration=1)
    
    # =========================================================================
    # CCP-Verify: Execute and validate
    # =========================================================================
    
    def verify(self, task: str) -> VerificationResult:
        """
        CCP-Verify: Execute created files and validate results.
        
        Uses:
        - Tier 2: Acceptance criteria
        - Tier 3: Files created in history
        
        Returns:
            VerificationResult with success status and analysis
        """
        # Get files from Tier 3 history
        files = self.memory.history.files_created.copy()
        
        # Also scan history for file creation mentions
        for event in self.memory.history.events:
            content = event.content
            patterns = [
                r"(?:created|wrote|saved)\s+(?:file\s+)?[`'\"]?([^\s`'\"]+\.py)[`'\"]?",
                r"([^\s]+\.py)\s+(?:created|saved)",
            ]
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for m in matches:
                    if m not in files:
                        files.append(m)
        
        if not files:
            return VerificationResult(
                success=True,
                execution_output="No executable files found to verify.",
                analysis="Task completed without creating executable files.",
            )
        
        # Execute Python files
        execution_results = []
        for filepath in files:
            if filepath.endswith('.py'):
                result = self._execute_python(filepath)
                execution_results.append(f"{filepath}: {result}")
        
        execution_output = "\n".join(execution_results) if execution_results else "No Python files to execute."
        
        # Get analysis from Gemini
        analysis = self._get_verification_analysis(task, files, execution_output)
        
        # Determine success
        success = "error" not in execution_output.lower() and "traceback" not in execution_output.lower()
        
        # Record in Tier 3
        self.memory.history.add_verification(task, success, execution_output, analysis)
        
        return VerificationResult(
            success=success,
            execution_output=execution_output,
            analysis=analysis,
            files_tested=files,
        )
    
    def _execute_python(self, filepath: str) -> str:
        """Execute a Python file and return result."""
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.config.working_dir, filepath)
        
        if not os.path.exists(filepath):
            # Try common locations
            for base in [self.config.working_dir, os.getcwd()]:
                test_path = os.path.join(base, os.path.basename(filepath))
                if os.path.exists(test_path):
                    filepath = test_path
                    break
            else:
                return f"File not found: {filepath}"
        
        try:
            result = subprocess.run(
                ["python3", filepath],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=self.config.working_dir,
                input=""
            )
            
            output = result.stdout[:500] if result.stdout else ""
            error = result.stderr[:500] if result.stderr else ""
            
            if result.returncode == 0:
                return f"OK - {output[:200]}" if output else "OK - ran successfully"
            else:
                return f"Error (code {result.returncode}): {error[:200]}"
                
        except subprocess.TimeoutExpired:
            return "Timeout - script took too long"
        except Exception as e:
            return f"Execution failed: {str(e)[:100]}"
    
    def _get_verification_analysis(self, task: str, files: List[str], output: str) -> str:
        """Get Gemini's verification analysis."""
        if not self._gemini:
            return "[Verification analysis unavailable]"
        
        prompt = f"""ORIGINAL TASK: {task}

FILES CREATED: {', '.join(files) if files else 'None'}

EXECUTION RESULTS:
{output}

As supervisor, verify: Does this satisfy the original task? 
Answer in ONE sentence: what was achieved and any issues found."""

        try:
            return self._gemini.call("You are a code reviewer.", prompt)
        except Exception as e:
            return f"[Analysis error: {str(e)[:30]}]"
    
    # =========================================================================
    # CCP-QA: Generate and run tests
    # =========================================================================
    
    def draft_qa_prompt(self, task: str) -> str:
        """
        Draft a QA prompt for Claude to test the implementation.
        
        Uses:
        - Tier 1: QA patterns
        - Tier 2: Acceptance criteria
        - Tier 3: Files created
        """
        # Gather context
        readme = self._find_readme()
        files = self.memory.history.files_created or self._find_project_files()
        main_content = self._read_main_file(files)
        
        context = f"""ORIGINAL TASK: {task}

PROJECT FILES: {', '.join(files) if files else 'None found'}

README CONTENT:
{readme[:500] if readme else 'No README found'}

MAIN FILE CONTENT:
{main_content[:800] if main_content else 'No main file found'}

ACCEPTANCE CRITERIA:
{chr(10).join(self.memory.session.acceptance_criteria) if self.memory.session.acceptance_criteria else 'None specified'}

Draft a QA prompt for Claude to test this application."""

        if self._gemini:
            try:
                return self._gemini.call(self.QA_PROMPT, context)
            except Exception:
                pass
        
        return f"Test the created files ({', '.join(files)}) and verify they work correctly."
    
    def run_qa_review(self, task: str) -> Generator[OutputEvent, None, QAResult]:
        """
        Run QA review: draft prompt via Gemini, execute via Claude.
        
        Yields events during execution.
        Returns QAResult.
        """
        # Draft QA prompt
        qa_prompt = self.draft_qa_prompt(task)
        
        # Announce
        yield self._emit_event(EventType.THINKING, f"QA Review: {qa_prompt}", source="ccp")
        
        # Execute via Claude
        claude_response = ""
        for raw_event in self._claude.run_task(qa_prompt):
            output_event = self._convert_claude_event(raw_event)
            output_event.metadata["source"] = "claude"
            output_event.metadata["qa_phase"] = True
            
            self._display.render_event(output_event)
            self.memory.history.add_event("claude_output", output_event.content)
            
            if output_event.content:
                claude_response += output_event.content + " "
            
            # CCP-Thinking on every QA Claude output (always show)
            thinking = self.think(output_event.content or "[empty]", output_event.event_type.name, qa_prompt)
            thinking_event = self._emit_event(
                EventType.THINKING,
                f"[{thinking.action}] {thinking.reasoning}",
                source="thinking",
                action=thinking.action
            )
            self._display.render_event(thinking_event)
            yield thinking_event
            
            yield output_event
        
        # Analyze for issues (simple heuristic)
        issues = []
        if "error" in claude_response.lower():
            issues.append("Errors detected in output")
        if "fail" in claude_response.lower():
            issues.append("Test failures detected")
        
        result = QAResult(
            prompt=qa_prompt,
            claude_response=claude_response.strip(),
            issues_found=issues,
            passed=len(issues) == 0,
        )
        
        # Record in Tier 3
        self.memory.history.add_qa_review(qa_prompt, claude_response, issues)
        
        yield self._emit_event(
            EventType.COMPLETED,
            f"QA Review {'passed' if result.passed else 'found issues: ' + ', '.join(issues)}",
            source="ccp"
        )
        
        return result
    
    def _find_readme(self) -> Optional[str]:
        """Find and read README file."""
        for name in ["README.md", "README.txt", "README", "readme.md"]:
            path = os.path.join(self.config.working_dir, name)
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return f.read()[:2000]
                except Exception:
                    pass
        return None
    
    def _find_project_files(self) -> List[str]:
        """Find relevant project files."""
        files = []
        extensions = ['.py', '.js', '.ts', '.html', '.json']
        
        try:
            for item in os.listdir(self.config.working_dir):
                if any(item.endswith(ext) for ext in extensions):
                    files.append(item)
        except Exception:
            pass
        
        return files[:10]
    
    def _read_main_file(self, files: List[str]) -> Optional[str]:
        """Read the main/entry file content."""
        priority = ['main.py', 'app.py', 'index.py', 'run.py']
        
        for pfile in priority:
            if pfile in files:
                path = os.path.join(self.config.working_dir, pfile)
                try:
                    with open(path, 'r') as f:
                        return f.read()[:1500]
                except Exception:
                    pass
        
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(self.config.working_dir, f)
                try:
                    with open(path, 'r') as fp:
                        return fp.read()[:1500]
                except Exception:
                    pass
        
        return None
    
    # =========================================================================
    # Main Execution
    # =========================================================================
    
    def run_task(self, task: str, max_iterations: int = 5, max_turns_per_iteration: int = 10) -> Generator[OutputEvent, None, None]:
        """
        Execute a task with full CCP supervision as a LONG-RUNNING process.
        
        UNIFIED MODE: Combines supervisor loop + auto-reply to Claude questions.
        
        Flow:
        1. Send task to Claude
        2. Monitor Claude output with CCP-thinking
        3. If Claude asks a question → CCP auto-replies (within same iteration)
        4. When Claude exits → CCP-Supervisor decides: DONE, CONTINUE, VERIFY
        5. If CONTINUE/VERIFY → spawn new Claude with follow-up
        6. Loop until DONE or max_iterations
        
        Yields OutputEvent objects during execution.
        """
        # Initialize components
        self._display = create_display(self.config.display_mode)
        
        # Update Tier 2 session
        self.memory.session.user_prompt = task
        self.memory.session.increment_task()
        
        # Display header
        self._display.render_header(
            f"CCP Session: {self.session_id}",
            f"Mission: {self.memory.session.user_mission or task[:50]}"
        )
        
        # CCP: Started
        yield self._emit_event(EventType.STARTED, f"Task: {task[:60]}...", source="ccp")
        yield self._emit_event(EventType.STARTED, "CCP starting - auto-replies to Claude, supervises on exit", source="ccp")
        
        self._state = ControllerState.PROCESSING
        
        current_prompt = task
        iteration = 0
        
        # =====================================================================
        # SUPERVISOR LOOP - CCP persists and controls Claude
        # =====================================================================
        while iteration < max_iterations:
            iteration += 1
            
            self._display.render_separator()
            yield self._emit_event(
                EventType.STARTED, 
                f"[Iteration {iteration}/{max_iterations}] Launching Claude...", 
                source="ccp"
            )
            
            # =================================================================
            # INNER LOOP: Run Claude with auto-reply to questions
            # CCP-NextStep is ALWAYS invoked after this loop (via try/finally)
            # =================================================================
            claude_session_id = None
            turn = 0
            inner_loop_error = None
            
            # Run Claude inner loop - CCP-NextStep is called in finally block
            try:
                while turn < max_turns_per_iteration:
                    turn += 1
                    
                    # Build Claude command
                    cmd = [
                        "claude",
                        "--dangerously-skip-permissions",
                        "--print",
                        "--verbose",
                        "--output-format", "stream-json",
                    ]
                    
                    # Resume session if we have one (for auto-reply continuity)
                    if claude_session_id:
                        cmd.extend(["--resume", claude_session_id])
                    
                    cmd.append(current_prompt)
                    
                    claude_response = ""
                    needs_response = False
                    claude_completed = False
                    server_detected = False
                    
                    try:
                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            cwd=self.config.working_dir,
                            text=True,
                        )
                        
                        for line in process.stdout:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                raw_event = json.loads(line)
                                output_event = self._convert_claude_event(raw_event)
                                
                                self._display.render_event(output_event)
                                self.memory.history.add_event("claude_output", output_event.content)
                                
                                # Track session ID for resume
                                if raw_event.get("type") == "system" and "session_id" in raw_event:
                                    claude_session_id = raw_event["session_id"]
                                
                                # Track file creation
                                self._track_files_from_content(output_event.content)
                                
                                # SERVER DETECTION: Check if Claude started a web server
                                if not server_detected and output_event.content:
                                    detected_url = self._detect_server_start(output_event.content)
                                    if detected_url:
                                        server_detected = True
                                        
                                        # Handle the server - verify and open browser
                                        for event in self._handle_server_detected(detected_url, process, task):
                                            self._display.render_event(event)
                                            yield event
                                        
                                        # After handling, check server status and break out
                                        is_up, status = self._health_check_server(detected_url, max_retries=1)
                                        self.memory.history.add_event(
                                            "server_detected", 
                                            f"Server at {detected_url}: {'UP' if is_up else 'DOWN'} - {status}"
                                        )
                                        break  # Exit the stdout reading loop
                                
                                # CCP-Thinking on every Claude output
                                thinking = self.think(output_event.content or "[empty]", output_event.event_type.name, task)
                                thinking_event = self._emit_event(
                                    EventType.THINKING,
                                    f"[{thinking.action}] {thinking.reasoning}",
                                    source="thinking",
                                    action=thinking.action
                                )
                                self._display.render_event(thinking_event)
                                yield thinking_event
                                
                                # Check if Claude is asking a question
                                if output_event.content and self._is_asking_question(output_event.content):
                                    needs_response = True
                                    claude_response = output_event.content
                                
                                # Check for completion
                                if raw_event.get("type") == "result" and raw_event.get("subtype") == "success":
                                    claude_completed = True
                                
                                yield output_event
                                
                            except json.JSONDecodeError:
                                yield self._emit_event(EventType.RAW, line, source="claude")
                        
                        # Only wait if we didn't break due to server detection
                        if not server_detected:
                            process.wait()
                        
                    except Exception as e:
                        inner_loop_error = e
                        yield self._emit_event(EventType.ERROR, str(e), source="ccp")
                        break
                    
                    # If server was detected, skip auto-reply logic and go to nextstep
                    if server_detected:
                        break
                    
                    # Handle Claude asking a question - auto-reply and continue
                    if needs_response and self._gemini and not claude_completed:
                        response = self._generate_auto_response(claude_response, task)
                        if response:
                            yield self._emit_event(EventType.PROMPT, f"[Auto-reply] {response}", source="ccp")
                            current_prompt = response
                            continue  # Continue inner loop with auto-reply
                    
                    # Claude completed or no question - exit inner loop
                    break
                    
            except Exception as inner_error:
                # Capture any uncaught exception
                inner_loop_error = inner_error
                yield self._emit_event(
                    EventType.ERROR,
                    f"Claude process error: {str(inner_error)[:100]}",
                    source="ccp"
                )
            
            finally:
                # ===============================================================
                # FINALLY BLOCK: CCP-NextStep ALWAYS executes after Claude exits
                # This is GUARANTEED to run no matter how the try block exits
                # ===============================================================
                
                # Render separator to visually mark CCP-NextStep section
                self._display.render_separator()
                
                # Emit start message - use direct print as backup in case yield fails
                try:
                    nextstep_start = self._emit_event(
                        EventType.STARTED, 
                        "══════ CCP-NextStep: Reviewing all context to decide next action ══════", 
                        source="nextstep"
                    )
                    self._display.render_event(nextstep_start)
                except Exception:
                    print("\033[93m\033[1m══════ CCP-NextStep: Reviewing all context ══════\033[0m")
                
                # Add error context if inner loop had issues
                if inner_loop_error:
                    self.memory.history.add_event(
                        "error", 
                        f"Claude process error: {str(inner_loop_error)[:200]}"
                    )
                
                # CCP-NextStep: comprehensive review of mission, history, and thinking
                # This is the ONLY place that decides DONE/CONTINUE/VERIFY
                try:
                    action, instruction = self.next_step(task, iteration)
                except Exception as nextstep_error:
                    # If nextstep fails, default to VERIFY (never silently DONE)
                    action = "VERIFY"
                    instruction = f"NextStep error: {str(nextstep_error)[:50]}. Verify manually."
                    print(f"\033[91mCCP-NextStep error: {str(nextstep_error)[:100]}\033[0m")
                
                # Display CCP-NextStep decision (always visible, even if yield fails)
                next_step_event = self._emit_event(
                    EventType.THINKING,
                    f"[{action}] {instruction}",
                    source="nextstep",
                    action=action,
                    next_step=True
                )
                self._display.render_event(next_step_event)
                
                # Store for use after finally block
                self._nextstep_action = action
                self._nextstep_instruction = instruction
            
            # Yield the nextstep event (now outside finally, but action is stored)
            yield self._emit_event(
                EventType.THINKING,
                f"[{self._nextstep_action}] {self._nextstep_instruction}",
                source="nextstep",
                action=self._nextstep_action,
                next_step=True
            )
            
            # Use stored values for decision
            action = self._nextstep_action
            instruction = self._nextstep_instruction
            
            # Act on supervisor decision
            if action == "DONE":
                yield self._emit_event(
                    EventType.COMPLETED,
                    f"✓ CCP: Task verified complete after {iteration} iteration(s)",
                    source="ccp"
                )
                break
                
            elif action == "VERIFY":
                # Run verification and use result to decide
                yield self._emit_event(EventType.THINKING, f"Verifying: {instruction}", source="thinking")
                
                verification = self.verify(task)
                status = "✓ VERIFIED" if verification.success else "⚠ NEEDS MORE WORK"
                
                yield self._emit_event(
                    EventType.THINKING,
                    f"{status}: {verification.analysis}",
                    source="thinking",
                    verification=True,
                    success=verification.success
                )
                
                if verification.success:
                    yield self._emit_event(
                        EventType.COMPLETED,
                        f"✓ CCP: Task verified complete after {iteration} iteration(s)",
                        source="ccp"
                    )
                    break
                else:
                    # Continue with verification analysis as next prompt
                    current_prompt = f"Previous work needs fixes. {verification.analysis}. Fix these issues."
                    
            elif action == "CONTINUE":
                # CCP-NextStep generated a follow-up prompt - launch new Claude
                current_prompt = instruction
                yield self._emit_event(
                    EventType.STARTED,
                    "CCP-NextStep: Launching new Claude with remaining work...",
                    source="nextstep"
                )
        
        else:
            # Max iterations reached
            yield self._emit_event(
                EventType.COMPLETED,
                f"⚠ CCP: Max iterations ({max_iterations}) reached. Review manually.",
                source="ccp"
            )
        
        # Final QA review if enabled
        if self.config.auto_qa:
            yield self._emit_event(EventType.STARTED, "Running final QA review...", source="ccp")
            self._claude = ClaudeProcessManager(working_dir=self.config.working_dir)
            for qa_event in self.run_qa_review(task):
                yield qa_event
        
        # Save session
        self._state = ControllerState.IDLE
        self.memory.save_session()
        
        yield self._emit_event(
            EventType.COMPLETED,
            f"CCP Session ended. Total iterations: {iteration}",
            source="ccp"
        )
    
    def _is_asking_question(self, content: str) -> bool:
        """Check if Claude is asking a question."""
        indicators = [
            "?",
            "would you like",
            "do you want",
            "should I",
            "please provide",
            "let me know",
            "what would you prefer",
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in indicators)
    
    def _generate_auto_response(self, question: str, original_task: str) -> str:
        """Generate auto-response to Claude's question."""
        if not self._gemini:
            return ""
        
        prompt = f"""ORIGINAL TASK: {original_task}

CLAUDE'S QUESTION: {question}

SESSION CONTEXT:
{self.memory.session.get_summary()}

As the user's proxy, provide a SHORT, DECISIVE answer (1-2 sentences).
- Prefer the simplest working solution
- Say "yes proceed" if unclear
- Don't ask counter-questions"""

        system = "You are a user proxy. Give brief, decisive answers."
        
        try:
            return self._gemini.call(system, prompt)
        except Exception:
            return "Yes, proceed with the default approach."
    
    # =========================================================================
    # Server Detection & Verification
    # =========================================================================
    
    # System prompt for server detection
    SERVER_DETECT_PROMPT = """You detect if output indicates a web server has started.

Analyze the text and determine:
1. Is a web server starting/running? (Flask, Django, Node, Vite, etc.)
2. What is the URL to access it?

OUTPUT FORMAT (exactly):
SERVER:YES|NO
URL:<url or NONE>

Examples:
Input: "* Serving Flask app 'app'\n* Running on http://127.0.0.1:5000"
Output:
SERVER:YES
URL:http://127.0.0.1:5000

Input: "Created file app.py"
Output:
SERVER:NO
URL:NONE

Input: "Starting development server at http://localhost:8000"
Output:
SERVER:YES
URL:http://localhost:8000

Be concise. Output ONLY the two lines."""
    
    def _detect_server_start(self, content: str) -> Optional[str]:
        """
        Use LLM to detect if content indicates a web server has started.
        
        Returns:
            URL of the server if detected, None otherwise.
        """
        if not content or len(content) < 10:
            return None
        
        # Quick pre-filter to avoid unnecessary LLM calls
        quick_indicators = ['serving', 'running on', 'localhost:', '127.0.0.1:', 
                           'development server', 'listening', ':5000', ':3000', 
                           ':8000', ':8080', 'server running']
        if not any(ind in content.lower() for ind in quick_indicators):
            return None
        
        if not self._gemini:
            # Fallback to simple regex if no Gemini
            return self._detect_server_start_fallback(content)
        
        try:
            response = self._gemini.call(self.SERVER_DETECT_PROMPT, content[:500])
            
            # Parse response
            lines = response.strip().split('\n')
            server_detected = False
            url = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('SERVER:YES'):
                    server_detected = True
                elif line.startswith('URL:') and 'NONE' not in line.upper():
                    url = line[4:].strip()
            
            if server_detected and url:
                # Validate URL format
                if url.startswith('http://') or url.startswith('https://'):
                    return url
                elif ':' in url:  # Just host:port
                    return f"http://{url}"
            
            return None
            
        except Exception:
            return self._detect_server_start_fallback(content)
    
    def _detect_server_start_fallback(self, content: str) -> Optional[str]:
        """Fallback regex-based server detection if LLM unavailable."""
        patterns = [
            r"Running on (https?://[^\s]+)",
            r"Starting development server at (https?://[^\s]+)",
            r"Local:\s+(https?://[^\s]+)",
            r"Server running at (https?://[^\s]+)",
            r"listening on (https?://[^\s]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Check for port patterns
        port_match = re.search(r'(?:localhost|127\.0\.0\.1):(\d+)', content)
        if port_match:
            return f"http://localhost:{port_match.group(1)}"
        
        return None
    
    def _health_check_server(self, url: str, max_retries: int = 3, delay: float = 1.0) -> Tuple[bool, str]:
        """
        Check if a server is responding at the given URL.
        
        Returns:
            Tuple of (is_up, response_info)
        """
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, method='GET')
                req.add_header('User-Agent', 'CCP-HealthCheck/1.0')
                
                with urllib.request.urlopen(req, timeout=5) as response:
                    status = response.status
                    content_length = len(response.read())
                    return True, f"Status {status}, {content_length} bytes"
                    
            except urllib.error.HTTPError as e:
                # Server is up but returned an error (still counts as "up")
                return True, f"Status {e.code} ({e.reason})"
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                return False, f"Connection failed: {str(e.reason)[:50]}"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                return False, f"Error: {str(e)[:50]}"
        
        return False, "Max retries exceeded"
    
    def _open_browser_and_verify(self, url: str) -> Tuple[bool, str]:
        """
        Open browser to verify server and perform health check.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        # First do a health check
        is_up, health_info = self._health_check_server(url)
        
        if is_up:
            # Open in default browser for user to see
            try:
                webbrowser.open(url)
            except Exception:
                pass  # Browser open is best-effort
            
            return True, f"Server UP at {url} - {health_info}"
        else:
            return False, f"Server DOWN at {url} - {health_info}"
    
    def _handle_server_detected(
        self, 
        url: str, 
        process: subprocess.Popen,
        task: str
    ) -> Generator[OutputEvent, None, Tuple[bool, str]]:
        """
        Handle detected server: verify, open browser, decide next steps.
        
        Yields events during the process.
        Returns (server_ok, analysis) tuple.
        """
        yield self._emit_event(
            EventType.STARTED,
            f"🌐 Server detected! Verifying {url}...",
            source="ccp"
        )
        
        # Give server a moment to fully start
        time.sleep(1.5)
        
        # Verify server is up
        is_up, status = self._open_browser_and_verify(url)
        
        if is_up:
            yield self._emit_event(
                EventType.COMPLETED,
                f"✓ {status} - Opening browser for verification",
                source="ccp"
            )
            
            # Server is up - terminate Claude process so we can proceed
            yield self._emit_event(
                EventType.STARTED,
                "Server running. Terminating Claude to proceed with CCP-nextstep...",
                source="ccp"
            )
            
            # Terminate the Claude process (server will keep running)
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            return (True, f"Server verified running at {url}")
        else:
            yield self._emit_event(
                EventType.ERROR,
                f"✗ {status}",
                source="ccp"
            )
            
            # Server not responding - kill everything
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            
            return (False, f"Server failed to respond at {url}: {status}")
    
    # =========================================================================
    # Helpers
    # =========================================================================
    
    def _emit_event(
        self,
        event_type: EventType,
        content: str,
        source: str = "ccp",
        **metadata
    ) -> OutputEvent:
        """Create and return an OutputEvent."""
        return OutputEvent(
            event_type=event_type,
            content=content,
            metadata={"source": source, **metadata},
        )
    
    def _convert_claude_event(self, raw: Dict[str, Any]) -> OutputEvent:
        """Convert raw Claude JSON to OutputEvent."""
        event_type = raw.get("type", "raw")
        
        type_map = {
            "assistant": EventType.TEXT,
            "user": EventType.TEXT,
            "system": EventType.TEXT,
            "tool_use": EventType.TOOL_CALL,
            "tool_result": EventType.TOOL_RESULT,
            "error": EventType.ERROR,
            "result": EventType.COMPLETED,
            "raw": EventType.RAW,
            "stderr": EventType.ERROR,
        }
        
        mapped_type = type_map.get(event_type, EventType.RAW)
        
        content = ""
        if "message" in raw:
            msg = raw["message"]
            if isinstance(msg, dict) and "content" in msg:
                content_items = msg["content"]
                if isinstance(content_items, list):
                    for item in content_items:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                content += item.get("text", "")
                            elif item.get("type") == "tool_use":
                                content = f"Tool: {item.get('name', 'unknown')}"
                                mapped_type = EventType.TOOL_CALL
                            elif item.get("type") == "tool_result":
                                content = str(item.get("content", ""))[:200]
                                mapped_type = EventType.TOOL_RESULT
                else:
                    content = str(content_items)
        elif "content" in raw:
            content = str(raw["content"])
        elif "error" in raw:
            content = str(raw["error"])
        elif "exit_code" in raw:
            content = f"Exit code: {raw['exit_code']}"
        else:
            content = str(raw)
        
        return OutputEvent(
            event_type=mapped_type,
            content=content,
            metadata={"source": "claude", **raw},
        )
    
    def _track_files_from_content(self, content: str) -> None:
        """Extract and track file paths from content."""
        patterns = [
            r"(?:created|wrote|saved)\s+(?:file\s+)?[`'\"]?([^\s`'\"]+\.[a-z]+)[`'\"]?",
            r"([^\s]+\.py)\s+(?:created|saved)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for m in matches:
                self.memory.history.track_file(m)
                self.memory.session.track_file(m)
    
    def stop(self) -> None:
        """Stop current task."""
        if self._claude:
            self._claude.stop()
        self._state = ControllerState.IDLE


# =============================================================================
# Convenience Functions
# =============================================================================

def create_ccp(
    working_dir: str = ".",
    session_id: Optional[str] = None,
    user_mission: Optional[str] = None,
    **kwargs
) -> CCP:
    """Factory function to create a CCP instance."""
    return CCP(
        working_dir=working_dir,
        session_id=session_id,
        user_mission=user_mission,
        **kwargs
    )


def list_sessions() -> List[Dict[str, Any]]:
    """List all available sessions."""
    module_dir = Path(__file__).parent
    sessions_dir = module_dir / "sessions"
    return CCPMemory.list_sessions(sessions_dir)
