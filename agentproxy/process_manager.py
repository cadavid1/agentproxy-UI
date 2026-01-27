"""Claude Process Manager
======================

Manages Claude CLI execution using subprocess with streaming JSON output.
Designed for programmatic control with real-time event streaming.
"""

import json
import os
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Generator
from queue import Queue, Empty

from .models import ControllerState


@dataclass
class ProcessConfig:
    """Configuration for Claude process."""
    working_dir: str = "."
    command: str = "claude"
    timeout: float = 300.0
    env: dict = field(default_factory=lambda: os.environ.copy())


class ClaudeProcessManager:
    """
    Manages Claude CLI using --print mode with streaming JSON.
    
    This approach uses Claude's native streaming JSON output format
    which is designed for programmatic consumption.
    
    Usage:
        manager = ClaudeProcessManager(working_dir=".")
        for event in manager.run_task("Create hello.py"):
            print(event)
    """
    
    def __init__(
        self,
        working_dir: str = ".",
        on_output: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ):
        self.config = ProcessConfig(working_dir=os.path.abspath(working_dir))
        self.on_output = on_output
        self.on_exit = on_exit
        
        self._process: Optional[subprocess.Popen] = None
        self._state = ControllerState.IDLE
        self._stop_event = threading.Event()
        self._output_queue: Queue = Queue()
        self._reader_thread: Optional[threading.Thread] = None
    
    @property
    def state(self) -> ControllerState:
        return self._state
    
    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None
    
    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process else None
    
    def run_task(self, prompt: str) -> Generator[Dict[str, Any], None, None]:
        """
        Run a task and yield streaming JSON events.
        
        Args:
            prompt: The task/prompt to send to Claude
            
        Yields:
            Dict events from Claude's streaming JSON output
        """
        self._state = ControllerState.PROCESSING
        self._stop_event.clear()
        
        cmd = [
            self.config.command,
            "--dangerously-skip-permissions",
            "--print",
            "--verbose",
            "--output-format", "stream-json",
            prompt
        ]
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.config.working_dir,
                env=self.config.env,
                text=True,
                bufsize=1,
            )
            
            # Start stderr reader thread
            self._reader_thread = threading.Thread(
                target=self._read_stderr,
                daemon=True
            )
            self._reader_thread.start()
            
            # Stream stdout line by line
            for line in self._process.stdout:
                if self._stop_event.is_set():
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = json.loads(line)
                    if self.on_output:
                        self.on_output(event)
                    yield event
                except json.JSONDecodeError:
                    # Non-JSON output, wrap it
                    event = {"type": "raw", "content": line}
                    yield event
            
            # Wait for process to complete
            self._process.wait()
            exit_code = self._process.returncode
            
            if self.on_exit:
                self.on_exit(exit_code)
            
            # Yield completion event
            yield {"type": "result", "exit_code": exit_code}
            
        except Exception as e:
            yield {"type": "error", "error": str(e)}
        finally:
            self._cleanup()
            self._state = ControllerState.IDLE
    
    def _read_stderr(self) -> None:
        """Read stderr in background thread."""
        if not self._process or not self._process.stderr:
            return
        
        for line in self._process.stderr:
            if self._stop_event.is_set():
                break
            self._output_queue.put({"type": "stderr", "content": line.strip()})
    
    def get_stderr(self) -> List[str]:
        """Get accumulated stderr output."""
        messages = []
        while True:
            try:
                msg = self._output_queue.get_nowait()
                messages.append(msg.get("content", ""))
            except Empty:
                break
        return messages
    
    def stop(self) -> None:
        """Stop the current task."""
        self._stop_event.set()
        
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        
        self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self._reader_thread and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=1.0)
        self._process = None
        self._state = ControllerState.IDLE


# Convenience function for simple usage
def run_claude_task(
    prompt: str,
    working_dir: str = ".",
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """
    Run a single Claude task and collect all events.
    
    Args:
        prompt: Task to execute
        working_dir: Working directory
        on_event: Optional callback for each event
        
    Returns:
        List of all events from the task
    """
    manager = ClaudeProcessManager(working_dir=working_dir)
    events = []
    
    for event in manager.run_task(prompt):
        events.append(event)
        if on_event:
            on_event(event)
    
    return events
