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
        
        return parser.parse_args()
    
    def run(self) -> int:
        """Main entry point."""
        args = self.parse_args()
        
        # Handle --list-sessions
        if args.list_sessions:
            return self._list_sessions()
        
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
            
            print(f"Session: {self.ccp.session_id}")
            
            exit_code = 0
            for event in self.ccp.run_task(task):
                if self._shutdown_requested:
                    break
                
                if event.event_type == EventType.ERROR:
                    exit_code = 1
            
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
