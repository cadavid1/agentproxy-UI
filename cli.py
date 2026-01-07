#!/usr/bin/env python3
"""
CCP - Code Custodian Persona CLI
================================

Entry point for running the unified CCP agent.

Usage:
    python -m interactive_labs "Create a hello world app"
    python -m interactive_labs --session abc123 "Continue working"
    python -m interactive_labs --list-sessions
    python -m interactive_labs --help
"""

import argparse
import sys
import signal
import os
from typing import Optional

from ccp import CCP, list_sessions
from models import EventType
from pathlib import Path


class CLI:
    """
    Command-line interface for CCP agent.
    """
    
    def __init__(self):
        self.ccp: Optional[CCP] = None
        self._shutdown_requested = False
    
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog="ccp",
            description="CCP (Code Custodian Persona) - AI supervisor for Claude Code",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s "Create a hello world Flask app"
  %(prog)s --mission "Build REST API" "Start with user endpoints"
  %(prog)s --session abc123 "Continue the work"
  %(prog)s --list-sessions
  %(prog)s -d ./myproject "Fix the bug in app.py"
            """
        )
        
        parser.add_argument(
            "task",
            nargs="*",
            help="Task to execute"
        )
        
        parser.add_argument(
            "-d", "--working-dir",
            default="./sandbox",
            help="Working directory for Claude (default: ./sandbox)"
        )
        
        parser.add_argument(
            "-m", "--mission",
            help="High-level mission for this session (persists across tasks)"
        )
        
        parser.add_argument(
            "-s", "--session",
            help="Resume an existing session by ID"
        )
        
        parser.add_argument(
            "--list-sessions",
            action="store_true",
            help="List all available sessions"
        )
        
        parser.add_argument(
            "--display",
            choices=["rich", "simple", "json", "quiet"],
            default="rich",
            help="Display mode (default: rich)"
        )
        
        parser.add_argument(
            "--no-verify",
            action="store_true",
            help="Disable auto-verification after task completion"
        )
        
        parser.add_argument(
            "--no-qa",
            action="store_true",
            help="Disable auto-QA review after task completion"
        )
        
        parser.add_argument(
            "--screenshot",
            nargs=2,
            metavar=("PATH", "DESCRIPTION"),
            action="append",
            help="Attach reference screenshot: --screenshot /path/to/img.png 'description'"
        )
        
        parser.add_argument(
            "--add-screenshot",
            action="append",
            metavar="PATH",
            help="Add screenshot to current session (repeatable): --add-screenshot img1.png --add-screenshot img2.jpg"
        )
        
        return parser.parse_args()
    
    def run(self) -> int:
        """Main entry point."""
        args = self.parse_args()
        
        # Handle --list-sessions
        if args.list_sessions:
            return self._list_sessions()
        
        # Handle --add-screenshot - attach to task, not standalone
        # Screenshots will be attached when task runs
        
        # Require task if not listing sessions
        if not args.task:
            print("Error: task is required", file=sys.stderr)
            print("Usage: ccp [OPTIONS] TASK", file=sys.stderr)
            return 1
        
        # Setup signal handler
        def signal_handler(sig, frame):
            self._shutdown_requested = True
            print("\nInterrupted...")
            if self.ccp:
                self.ccp.stop()
            sys.exit(130)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        task = " ".join(args.task)
        return self._run_task(args, task)
    
    def _attach_screenshots(self, ccp: 'CCP', paths: list) -> int:
        """Attach screenshots to the current session."""
        SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        attached = 0
        
        for path in paths:
            p = Path(path)
            if not p.exists():
                print(f"⚠ Screenshot not found: {path}", file=sys.stderr)
                continue
            
            ext = p.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                print(f"⚠ Unsupported image format: {path} (supported: {', '.join(SUPPORTED_EXTENSIONS)})", file=sys.stderr)
                continue
            
            # Use filename as description
            description = p.stem.replace('_', ' ').replace('-', ' ')
            ccp.memory.session.add_screenshot(str(p.absolute()), description)
            print(f"📸 Attached: {path}")
            attached += 1
        
        return attached
    
    def _list_sessions(self) -> int:
        """List all available sessions."""
        sessions = list_sessions()
        
        if not sessions:
            print("No sessions found.")
            return 0
        
        print(f"{'ID':<10} {'Mission':<40} {'Tasks':<6} {'Last Active'}")
        print("-" * 80)
        
        for s in sessions:
            session_id = s.get("session_id", "?")[:8]
            mission = s.get("mission", "")[:38] or "(no mission)"
            tasks = s.get("task_count", 0)
            last_active = s.get("last_active", "")[:19]
            print(f"{session_id:<10} {mission:<40} {tasks:<6} {last_active}")
        
        return 0
    
    def _run_task(self, args: argparse.Namespace, task: str) -> int:
        """Execute a single task."""
        # Create working directory if it doesn't exist
        if not os.path.exists(args.working_dir):
            os.makedirs(args.working_dir)
            print(f"Created working directory: {args.working_dir}")
        
        try:
            self.ccp = CCP(
                working_dir=args.working_dir,
                session_id=args.session,
                user_mission=args.mission,
                display_mode=args.display,
                auto_verify=not args.no_verify,
                auto_qa=not args.no_qa,
            )
            
            # Attach screenshots if provided (--screenshot with description)
            if args.screenshot:
                for path, description in args.screenshot:
                    if Path(path).exists():
                        self.ccp.memory.session.add_screenshot(path, description)
                        print(f"📸 Attached: {path}")
                    else:
                        print(f"⚠ Screenshot not found: {path}", file=sys.stderr)
            
            # Attach screenshots from --add-screenshot (just paths, auto-description)
            if args.add_screenshot:
                self._attach_screenshots(self.ccp, args.add_screenshot)
            
            print(f"Session: {self.ccp.session_id}")
            print()
            
            # Color legend - each line shown in its actual color
            print("\033[1m COLOR LEGEND \033[0m")
            print("\033[35m┃ Claude ┃ Claude subprocess output\033[0m")
            print("\033[36m│ CCP    │ CCP orchestration & reasoning\033[0m")
            print("\033[93m■ Tool calls - function invocations\033[0m")
            print("\033[95m■ Tool results - execution output\033[0m")
            print("\033[96m■ CCP thinking - analysis & decisions\033[0m")
            print("\033[94m■ CCP tasks - task breakdown & tracking\033[0m")
            print("\033[92m■ Completed - success messages\033[0m")
            print("\033[91m■ Errors - failures & issues\033[0m")
            print()
            
            exit_code = 0
            for event in self.ccp.run_task(task):
                if self._shutdown_requested:
                    break
                
                # Display each event in real-time
                source = event.metadata.get("source", "ccp") if event.metadata else "ccp"
                is_claude = source == "claude"
                
                # Colored prefix based on source
                if is_claude:
                    prefix = "\033[35m┃ Claude ┃\033[0m"  # Magenta for Claude
                else:
                    prefix = "\033[36m│ CCP    │\033[0m"  # Cyan for CCP
                
                # Color content based on event type
                if event.event_type == EventType.ERROR:
                    print(f"{prefix} \033[91m{event.content}\033[0m")  # Red
                    exit_code = 1
                elif event.event_type == EventType.COMPLETED:
                    print(f"{prefix} \033[92m{event.content}\033[0m")  # Green
                elif event.event_type == EventType.THINKING:
                    print(f"{prefix} \033[96m{event.content}\033[0m")  # Bright cyan
                elif event.event_type == EventType.TOOL_CALL:
                    print(f"{prefix} \033[93m{event.content}\033[0m")  # Yellow
                elif event.event_type == EventType.TOOL_RESULT:
                    print(f"{prefix} \033[95m{event.content}\033[0m")  # Magenta
                else:
                    print(f"{prefix} {event.content}")  # Default
                
                sys.stdout.flush()
            
            return exit_code
            
        except KeyboardInterrupt:
            print("\nInterrupted")
            return 130
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1


def main() -> int:
    """Main entry point."""
    cli = CLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
