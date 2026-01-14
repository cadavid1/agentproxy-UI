#!/usr/bin/env python3
"""
PA - Proxy Agent CLI
================================

Entry point for running the unified PA agent.

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

from pa import PA, list_sessions
from models import EventType
from pathlib import Path

CONFIG_FILE = Path.home() / ".pa_config"


class CLI:
    """
    Command-line interface for PA agent.
    """
    
    def __init__(self):
        self.pa: Optional[PA] = None
        self._shutdown_requested = False
    
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(
            prog="pa",
            description="PA (Proxy Agent) - AI supervisor for Claude Code",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  %(prog)s "Create a hello world Flask app"
  %(prog)s -t "Create a hello world Flask app"
  %(prog)s --session abc123 "Continue the work"
  %(prog)s --list-sessions
  %(prog)s -d ./myproject "Fix the bug in app.py"
  %(prog)s --set-workdir ./myproject   # Set default working directory
  %(prog)s --show-workdir               # Show current default
            """
        )
        
        parser.add_argument(
            "task",
            nargs="*",
            help="Task to execute (positional)"
        )
        
        parser.add_argument(
            "-t", "--task",
            dest="task_flag",
            help="Task to execute (alternative to positional)"
        )
        
        parser.add_argument(
            "-d", "--working-dir",
            default="./sandbox",
            help="Working directory for Claude (default: ./sandbox)"
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
            "--set-workdir",
            metavar="PATH",
            help="Set default working directory (persists to ~/.pa_config)"
        )
        
        parser.add_argument(
            "--show-workdir",
            action="store_true",
            help="Show current default working directory"
        )
        
        parser.add_argument(
            "--set-contextdir",
            metavar="PATH",
            help="Set project context directory (files here inform PA's decisions)"
        )
        
        parser.add_argument(
            "--show-contextdir",
            action="store_true",
            help="Show current project context directory"
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
        
        # Handle --set-workdir (saves config, then continues if task provided)
        if args.set_workdir:
            self._set_workdir(args.set_workdir)
        
        # Handle --show-workdir
        if args.show_workdir:
            return self._show_workdir()
        
        # Handle --set-contextdir (saves config, then continues if task provided)
        if args.set_contextdir:
            self._set_contextdir(args.set_contextdir)
        
        # Handle --show-contextdir
        if args.show_contextdir:
            return self._show_contextdir()
        
        # Handle --list-sessions
        if args.list_sessions:
            return self._list_sessions()
        
        # Handle --add-screenshot - attach to task, not standalone
        # Screenshots will be attached when task runs
        
        # Get task from either positional or -t/--task flag
        if args.task_flag:
            task = args.task_flag
        elif args.task:
            task = " ".join(args.task)
        else:
            self._print_usage()
            return 1
        
        # Setup signal handler
        def signal_handler(sig, frame):
            self._shutdown_requested = True
            print("\nInterrupted...")
            if self.pa:
                self.pa.stop()
            sys.exit(130)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        return self._run_task(args, task)
    
    def _attach_screenshots(self, pa: 'PA', paths: list) -> int:
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
            
            # Use full filename so user can reference it in task
            filename = p.name
            pa.memory.session.add_screenshot(str(p.absolute()), filename)
            print(f"📸 Attached: {filename}")
            attached += 1
        
        return attached
    
    def _print_usage(self) -> None:
        """Print full usage help with examples."""
        print("""
\033[1mPA (Proxy Agent)\033[0m - AI supervisor for Claude Code

\033[1mUSAGE:\033[0m
    pa [OPTIONS] "TASK"
    pa [OPTIONS] -t "TASK"

\033[1mOPTIONS:\033[0m
    -t, --task TEXT         Task to execute
    -d, --dir PATH          Working directory (default: current)
    -s, --session ID        Resume existing session
    --display MODE          Output mode: rich|simple|json|quiet (default: rich)
    --no-verify             Disable auto-verification
    --no-qa                 Disable auto QA
    --add-screenshot PATH   Attach screenshot(s) to task
    --screenshot PATH DESC  Attach screenshot with description
    --list-sessions         List all sessions
    --set-workdir PATH      Set default working directory
    --show-workdir          Show current default directory
    --set-contextdir PATH   Set project context directory (PA reads these files)
    --show-contextdir       Show current context directory and files

\033[1mEXAMPLES:\033[0m
    \033[36m# Simple task (positional)\033[0m
    pa "Create a hello world Flask app"

    \033[36m# Task with -t flag\033[0m
    pa -t "Create a hello world Flask app"

    \033[36m# Full setup in one command (recommended first run)\033[0m
    pa --set-workdir ./myproject --set-contextdir ./docs -t "Create user endpoints"

    \033[36m# With working directory\033[0m
    pa -d ./myproject "Fix the bug in app.py"

    \033[36m# Resume a session\033[0m
    pa -s abc123 "Continue where we left off"

    \033[36m# With screenshot reference\033[0m
    pa --add-screenshot design.png "Build UI matching this design"

\033[1mMORE INFO:\033[0m
    pa --help               Full argument details
""")
    
    def _get_config(self, key: str) -> Optional[str]:
        """Get a config value from ~/.pa_config."""
        if CONFIG_FILE.exists():
            try:
                content = CONFIG_FILE.read_text().strip()
                for line in content.split('\n'):
                    if line.startswith(f'{key}='):
                        return line.split('=', 1)[1]
            except Exception:
                pass
        return None
    
    def _set_config(self, key: str, value: str) -> None:
        """Set a config value in ~/.pa_config."""
        config = {}
        if CONFIG_FILE.exists():
            try:
                for line in CONFIG_FILE.read_text().strip().split('\n'):
                    if '=' in line:
                        k, v = line.split('=', 1)
                        config[k] = v
            except Exception:
                pass
        config[key] = value
        CONFIG_FILE.write_text('\n'.join(f'{k}={v}' for k, v in config.items()) + '\n')
    
    def _get_saved_workdir(self) -> Optional[str]:
        """Get saved working directory from config."""
        return self._get_config('workdir')
    
    def _set_workdir(self, path: str) -> int:
        """Set default working directory."""
        abs_path = Path(path).resolve()
        if not abs_path.exists():
            abs_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {abs_path}")
        
        self._set_config('workdir', str(abs_path))
        print(f"✓ Default working directory set to: {abs_path}")
        return 0
    
    def _show_workdir(self) -> int:
        """Show current default working directory."""
        saved = self._get_saved_workdir()
        if saved:
            print(f"Default working directory: {saved}")
        else:
            print("No default working directory set (using ./sandbox)")
        return 0
    
    def _get_saved_contextdir(self) -> Optional[str]:
        """Get saved project context directory from config."""
        return self._get_config('contextdir')
    
    def _set_contextdir(self, path: str) -> int:
        """Set project context directory."""
        abs_path = Path(path).resolve()
        if not abs_path.exists():
            print(f"⚠ Directory does not exist: {abs_path}", file=sys.stderr)
            return 1
        
        # Count text files
        text_files = list(abs_path.glob('*.md')) + list(abs_path.glob('*.txt'))
        
        # Count images (recursively including subdirs)
        image_exts = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.bmp', '*.svg']
        image_files = []
        for ext in image_exts:
            image_files.extend(abs_path.rglob(ext))  # rglob for recursive
        
        self._set_config('contextdir', str(abs_path))
        print(f"✓ Project context directory set to: {abs_path}")
        print(f"  Found {len(text_files)} text files (.md, .txt)")
        if image_files:
            print(f"  Found {len(image_files)} images (will be sent to Gemini)")
            for img in sorted(image_files)[:5]:
                print(f"    - {img.name} (ref: '{img.stem}')")
            if len(image_files) > 5:
                print(f"    ... and {len(image_files) - 5} more")
        return 0
    
    def _show_contextdir(self) -> int:
        """Show current project context directory."""
        saved = self._get_saved_contextdir()
        if saved:
            print(f"Project context directory: {saved}")
            context_path = Path(saved)
            if context_path.exists():
                # Show text files
                text_files = list(context_path.glob('*.md')) + list(context_path.glob('*.txt'))
                if text_files:
                    print(f"\nText files ({len(text_files)}):")
                    for f in sorted(text_files)[:10]:
                        print(f"  - {f.name}")
                    if len(text_files) > 10:
                        print(f"  ... and {len(text_files) - 10} more")
                
                # Show images (recursively)
                image_exts = ['*.png', '*.jpg', '*.jpeg', '*.gif', '*.webp', '*.bmp', '*.svg']
                image_files = []
                for ext in image_exts:
                    image_files.extend(context_path.rglob(ext))  # rglob for recursive
                
                if image_files:
                    print(f"\nImages ({len(image_files)}) - can be referenced by name:")
                    for f in sorted(image_files)[:10]:
                        print(f"  - {f.name} (ref: '{f.stem}')")
                    if len(image_files) > 10:
                        print(f"  ... and {len(image_files) - 10} more")
                    print("\n  Tip: Reference images in your task like 'match the architecture image'")
                    print("       or 'follow the design in sequence.jpg'")
        else:
            print("No project context directory set")
        return 0
    
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
        # Use saved workdir if -d not explicitly provided
        working_dir = args.working_dir
        if working_dir == "./sandbox":  # default value, check for saved
            saved = self._get_saved_workdir()
            if saved:
                working_dir = saved
        
        # Get saved context directory
        context_dir = self._get_saved_contextdir()
        
        # Create working directory if it doesn't exist
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
            print(f"Created working directory: {working_dir}")
        
        try:
            self.pa = PA(
                working_dir=working_dir,
                session_id=args.session,
                display_mode=args.display,
                auto_verify=not args.no_verify,
                auto_qa=not args.no_qa,
                context_dir=context_dir,
            )
            
            # Attach screenshots if provided (--screenshot with description)
            if args.screenshot:
                for path, description in args.screenshot:
                    if Path(path).exists():
                        self.pa.memory.session.add_screenshot(path, description)
                        print(f"📸 Attached: {path}")
                    else:
                        print(f"⚠ Screenshot not found: {path}", file=sys.stderr)
            
            # Attach screenshots from --add-screenshot (just paths, auto-description)
            if args.add_screenshot:
                self._attach_screenshots(self.pa, args.add_screenshot)
            
            print(f"Session: {self.pa.session_id}")
            print()
            
            # Color legend - each line shown in its actual color
            print("\033[1m COLOR LEGEND \033[0m")
            print("\033[35m┃ Claude ┃ Claude subprocess output\033[0m")
            print("\033[36m│ PA     │ PA orchestration & reasoning\033[0m")
            print("\033[93m■ Tool calls - function invocations\033[0m")
            print("\033[95m■ Tool results - execution output\033[0m")
            print("\033[96m■ PA thinking - analysis & decisions\033[0m")
            print("\033[94m■ PA tasks - task breakdown & tracking\033[0m")
            print("\033[92m■ Completed - success messages\033[0m")
            print("\033[91m■ Errors - failures & issues\033[0m")
            print()
            
            exit_code = 0
            for event in self.pa.run_task(task):
                if self._shutdown_requested:
                    break
                
                # Display each event in real-time
                source = event.metadata.get("source", "pa") if event.metadata else "pa"
                is_claude = source == "claude"
                
                # Colored prefix based on source
                if is_claude:
                    prefix = "\033[35m┃ Claude ┃\033[0m"  # Magenta for Claude
                else:
                    prefix = "\033[36m│ PA     │\033[0m"  # Cyan for PA
                
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
