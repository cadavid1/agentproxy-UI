"""
Real-time Display for Claude Output
===================================

Renders Claude CLI events to the terminal with rich formatting.
Provides clear visual distinction between different event types.
"""

import sys
from typing import Optional, TextIO
from enum import Enum

from models import OutputEvent, EventType


class DisplayMode(Enum):
    """Display output modes."""
    RICH = "rich"       # Full colors and formatting
    SIMPLE = "simple"   # Basic formatting, no colors
    JSON = "json"       # JSON output for programmatic use
    QUIET = "quiet"     # Minimal output


class Colors:
    """ANSI color codes for terminal output."""
    
    # Basic colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    
    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright foreground
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    
    # Background colors
    BG_BLACK = "\033[40m"
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


class RealtimeDisplay:
    """
    Renders events with clear visual distinction between sources.
    
    Sources (color-coded prefixes):
        PA      (CYAN)     │ PA     │  PA agent (orchestration, thinking, QA)
        Claude  (MAGENTA)  ┃ Claude ┃  Claude Code subprocess (execution)
    """
    
    # Source-to-prefix mapping - PA, PA-thinking, PA-NextStep, and Claude
    SOURCE_PREFIXES = {
        "pa":          (Colors.CYAN,    "│ PA          │"),
        "pa-thinking": (Colors.BLUE,    "│ PA-thinking │"),
        "pa-nextstep": (Colors.BRIGHT_YELLOW + Colors.BOLD, "│ PA-nextstep │"),
        "claude":      (Colors.MAGENTA, "┃ Claude      ┃"),
    }
    
    # Event type styling configuration
    EVENT_STYLES = {
        EventType.TEXT: {
            "prefix": "",
            "color": Colors.RESET,
            "show_timestamp": False,
        },
        EventType.THINKING: {
            "prefix": "",
            "color": Colors.DIM + Colors.ITALIC,
            "show_timestamp": False,
        },
        EventType.TOOL_CALL: {
            "prefix": "🔧 ",
            "color": Colors.YELLOW,
            "show_timestamp": True,
        },
        EventType.TOOL_RESULT: {
            "prefix": "   ",
            "color": Colors.DIM,
            "show_timestamp": False,
        },
        EventType.PROMPT: {
            "prefix": "❯ ",
            "color": Colors.CYAN + Colors.BOLD,
            "show_timestamp": False,
        },
        EventType.CONFIRMATION: {
            "prefix": "⚠️  ",
            "color": Colors.BRIGHT_YELLOW + Colors.BOLD,
            "show_timestamp": False,
        },
        EventType.STARTED: {
            "prefix": "🚀 ",
            "color": Colors.GREEN,
            "show_timestamp": True,
        },
        EventType.COMPLETED: {
            "prefix": "✨ ",
            "color": Colors.GREEN + Colors.BOLD,
            "show_timestamp": True,
        },
        EventType.ERROR: {
            "prefix": "❌ ",
            "color": Colors.RED + Colors.BOLD,
            "show_timestamp": True,
        },
        EventType.RAW: {
            "prefix": "",
            "color": Colors.DIM,
            "show_timestamp": False,
        },
    }
    
    def __init__(
        self,
        mode: DisplayMode = DisplayMode.RICH,
        output: TextIO = None,
        show_timestamps: bool = False,
        max_line_length: int = 120,
    ):
        """
        Initialize the display.
        
        Args:
            mode: Display mode (rich, simple, json, quiet)
            output: Output stream (default: sys.stdout)
            show_timestamps: Override to show timestamps for all events
            max_line_length: Truncate lines longer than this
        """
        self.mode = mode
        self.output = output or sys.stdout
        self.show_timestamps = show_timestamps
        self.max_line_length = max_line_length
        
        # Track state for smart rendering
        self._last_event_type: Optional[EventType] = None
        self._tool_depth = 0
    
    # =========================================================================
    # Public Methods
    # =========================================================================
    
    def render_event(self, event: OutputEvent) -> None:
        """
        Render an event to the terminal.
        
        Args:
            event: The event to display
        """
        if self.mode == DisplayMode.QUIET:
            return
        
        if self.mode == DisplayMode.JSON:
            self._render_json(event)
            return
        
        if self.mode == DisplayMode.SIMPLE:
            self._render_simple(event)
            return
        
        # Rich mode
        self._render_rich(event)
        self._last_event_type = event.event_type
    
    def render_header(self, title: str, subtitle: str = "") -> None:
        """Render a section header."""
        if self.mode == DisplayMode.QUIET:
            return
        
        width = 60
        line = "═" * width
        
        self._write(f"\n{Colors.CYAN}{Colors.BOLD}{line}{Colors.RESET}")
        self._write(f"{Colors.CYAN}{Colors.BOLD}  {title}{Colors.RESET}")
        if subtitle:
            self._write(f"{Colors.DIM}  {subtitle}{Colors.RESET}")
        self._write(f"{Colors.CYAN}{Colors.BOLD}{line}{Colors.RESET}")
        self._write("")
        self._write(f"{Colors.CYAN}│ PA          │{Colors.RESET} = PA orchestration")
        self._write(f"{Colors.BLUE}│ PA-thinking │{Colors.RESET} = PA reasoning about Claude's output")
        self._write(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}│ PA-nextstep │{Colors.RESET} = PA decision after Claude exits")
        self._write(f"{Colors.MAGENTA}┃ Claude      ┃{Colors.RESET} = Claude Code subprocess")
        self._write("")
    
    def render_status(self, message: str, status_type: str = "info") -> None:
        """
        Render a status message.
        
        Args:
            message: Status message
            status_type: One of "info", "success", "warning", "error"
        """
        if self.mode == DisplayMode.QUIET:
            return
        
        colors = {
            "info": Colors.CYAN,
            "success": Colors.GREEN,
            "warning": Colors.YELLOW,
            "error": Colors.RED,
        }
        icons = {
            "info": "ℹ️ ",
            "success": "✓ ",
            "warning": "⚠ ",
            "error": "✗ ",
        }
        
        color = colors.get(status_type, Colors.RESET)
        icon = icons.get(status_type, "• ")
        
        self._write(f"{color}{icon}{message}{Colors.RESET}")
    
    def render_separator(self, char: str = "─", width: int = 40) -> None:
        """Render a visual separator line."""
        if self.mode not in (DisplayMode.QUIET, DisplayMode.JSON):
            self._write(f"{Colors.DIM}{char * width}{Colors.RESET}")
    
    def render_tool_call(self, name: str, params: dict) -> None:
        """Render a tool invocation with parameters."""
        if self.mode == DisplayMode.QUIET:
            return
        
        self._write(f"\n{Colors.YELLOW}▶ Tool:{Colors.RESET} {Colors.BOLD}{name}{Colors.RESET}")
        
        for key, value in params.items():
            value_str = self._truncate(str(value), 80)
            self._write(f"  {Colors.DIM}{key}:{Colors.RESET} {value_str}")
    
    def render_tool_result(self, result: str, is_error: bool = False) -> None:
        """Render tool execution result."""
        if self.mode == DisplayMode.QUIET:
            return
        
        color = Colors.RED if is_error else Colors.GREEN
        icon = "✗" if is_error else "✓"
        
        self._write(f"  {color}{icon} Result:{Colors.RESET}")
        
        # Truncate long results
        lines = result.split("\n")
        if len(lines) > 10:
            for line in lines[:10]:
                self._write(f"    {Colors.DIM}{self._truncate(line, 100)}{Colors.RESET}")
            self._write(f"    {Colors.DIM}... ({len(lines) - 10} more lines){Colors.RESET}")
        else:
            for line in lines:
                self._write(f"    {Colors.DIM}{self._truncate(line, 100)}{Colors.RESET}")
    
    def render_thinking(self, content: str) -> None:
        """Render thinking/reasoning content."""
        if self.mode == DisplayMode.QUIET:
            return
        
        self._write(f"{Colors.DIM}{Colors.ITALIC}{self._truncate(content, 200)}{Colors.RESET}")
    
    def render_confirmation_prompt(self, message: str) -> None:
        """Render a confirmation prompt prominently."""
        if self.mode == DisplayMode.QUIET:
            return
        
        self._write(f"\n{Colors.YELLOW}{'─' * 50}{Colors.RESET}")
        self._write(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}⚠️  CONFIRMATION REQUIRED{Colors.RESET}")
        self._write(f"  {message}")
        self._write(f"{Colors.YELLOW}{'─' * 50}{Colors.RESET}")
    
    def clear_line(self) -> None:
        """Clear the current line (for progress updates)."""
        self._write_raw("\r\033[K")
    
    def render_progress(self, message: str) -> None:
        """Render an inline progress message (updates same line)."""
        if self.mode not in (DisplayMode.QUIET, DisplayMode.JSON):
            self._write_raw(f"\r{Colors.DIM}{message}{Colors.RESET}")
    
    # =========================================================================
    # Private Methods
    # =========================================================================
    
    # Action tag colors for PA thinking output
    ACTION_COLORS = {
        "CONTINUE": Colors.YELLOW,
        "VERIFY": Colors.BRIGHT_CYAN,
        "DONE": Colors.GREEN + Colors.BOLD,
    }
    
    def _colorize_action_tags(self, content: str) -> str:
        """Colorize [CONTINUE], [VERIFY], [DONE] tags in content."""
        import re
        
        def replace_tag(match):
            tag = match.group(1)
            color = self.ACTION_COLORS.get(tag, Colors.RESET)
            return f"{color}[{tag}]{Colors.RESET}"
        
        return re.sub(r'\[(CONTINUE|VERIFY|DONE)\]', replace_tag, content)
    
    def _render_rich(self, event: OutputEvent) -> None:
        """Render event with full formatting and source distinction."""
        style = self.EVENT_STYLES.get(event.event_type, {})
        
        prefix = style.get("prefix", "")
        color = style.get("color", Colors.RESET)
        show_ts = self.show_timestamps or style.get("show_timestamp", False)
        
        # Get source prefix from mapping (default to Claude)
        source = event.metadata.get("source", "claude")
        source_color, source_label = self.SOURCE_PREFIXES.get(
            source, 
            self.SOURCE_PREFIXES["claude"]
        )
        proc_prefix = f"{source_color}{source_label}{Colors.RESET} "
        
        # PA-nextstep: entire content is yellow bold (high visibility)
        if source == "pa-nextstep":
            color = Colors.BRIGHT_YELLOW + Colors.BOLD
        
        # Build output line
        parts = [proc_prefix]
        
        if show_ts:
            ts = event.timestamp.strftime("%H:%M:%S")
            parts.append(f"{Colors.DIM}[{ts}]{Colors.RESET} ")
        
        # Colorize action tags in content (skip for nextstep - already fully colored)
        if source == "pa-nextstep":
            content = event.content
        else:
            content = self._colorize_action_tags(event.content)
        
        parts.append(f"{color}{prefix}{content}{Colors.RESET}")
        
        self._write("".join(parts))
    
    def _render_simple(self, event: OutputEvent) -> None:
        """Render event with minimal formatting."""
        type_labels = {
            EventType.TEXT: "",
            EventType.THINKING: "[THINK] ",
            EventType.TOOL_CALL: "[TOOL] ",
            EventType.TOOL_RESULT: "[RESULT] ",
            EventType.PROMPT: "[PROMPT] ",
            EventType.CONFIRMATION: "[CONFIRM] ",
            EventType.STARTED: "[START] ",
            EventType.COMPLETED: "[DONE] ",
            EventType.ERROR: "[ERROR] ",
            EventType.RAW: "",
        }
        
        label = type_labels.get(event.event_type, "")
        self._write(f"{label}{event.content}")
    
    def _render_json(self, event: OutputEvent) -> None:
        """Render event as JSON."""
        import json
        self._write(json.dumps(event.to_dict()))
    
    def _write(self, text: str) -> None:
        """Write a line to output."""
        print(text, file=self.output, flush=True)
    
    def _write_raw(self, text: str) -> None:
        """Write raw text without newline."""
        print(text, end="", file=self.output, flush=True)
    
    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."


def create_display(mode: str = "rich", **kwargs) -> RealtimeDisplay:
    """
    Factory function to create a display instance.
    
    Args:
        mode: "rich", "simple", "json", or "quiet"
        **kwargs: Additional arguments for RealtimeDisplay
        
    Returns:
        Configured RealtimeDisplay instance
    """
    mode_map = {
        "rich": DisplayMode.RICH,
        "simple": DisplayMode.SIMPLE,
        "json": DisplayMode.JSON,
        "quiet": DisplayMode.QUIET,
    }
    
    display_mode = mode_map.get(mode.lower(), DisplayMode.RICH)
    return RealtimeDisplay(mode=display_mode, **kwargs)
