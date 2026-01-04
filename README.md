# Interactive Labs - Claude Code Process Controller

A Python framework for running Claude CLI as a subprocess with programmatic control over input/output.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CCP (Controller Process)                   │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   Input     │───▶│   Claude    │───▶│   Output Parser     │  │
│  │   Writer    │    │   Process   │    │   & Event Stream    │  │
│  │   (stdin)   │◀───│   (PTY)     │◀───│   (stdout/stderr)   │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `models.py` | Data classes for events, state, and sessions |
| `process_manager.py` | PTY-based process spawning and lifecycle |
| `output_parser.py` | Parse CLI output into structured events |
| `io_handler.py` | Async bidirectional I/O communication |
| `display.py` | Real-time terminal rendering |
| `controller.py` | High-level orchestration API |
| `cli.py` | Command-line interface |

## Usage

### Command Line

```bash
# Single task
python -m interactive_labs "Create a hello world Flask app"

# With a mission
python -m interactive_labs --mission "Build REST API" "Start with user endpoints"

# Resume existing session
python -m interactive_labs --session abc123 "Continue the work"

# With options
python -m interactive_labs -d ./myproject --display simple "Fix bug in app.py"
```

### Reference Screenshots

Attach screenshots to show CCP what you want to build. Supports PNG, JPG, JPEG, GIF, WebP, BMP.

```bash
# Simple: just pass image paths (uses filename as description)
python -m interactive_labs \
  --add-screenshot /path/to/mockup.png \
  --add-screenshot /path/to/login.jpg \
  "Build a web app that looks exactly like these screenshots"

# With custom descriptions (--screenshot takes path + description)
python -m interactive_labs \
  --screenshot /path/to/mockup.png "Main dashboard - must look like this" \
  --screenshot /path/to/login.png "Login page design" \
  "Build a web app matching these designs"
```

**Images are sent directly to Gemini** for visual context during CCP's thinking and supervision.

Screenshots are stored in the session JSON:

```json
{
  "session_id": "abc123",
  "reference_screenshots": [
    {
      "path": "/path/to/mockup.png",
      "description": "Main dashboard - must look like this",
      "added_at": "2026-01-03T21:35:29"
    }
  ]
}
```

### Programmatic

```python
import asyncio
from interactive_labs import ClaudeCodeController

async def main():
    async with ClaudeCodeController(working_dir=".") as controller:
        async for event in controller.execute_task("Create hello.py"):
            print(event)

asyncio.run(main())
```

### Event Types

- `TEXT` - Regular text response
- `THINKING` - Extended thinking content
- `TOOL_CALL` - Claude invoking a tool
- `TOOL_RESULT` - Result from tool execution
- `PROMPT` - Waiting for user input
- `CONFIRMATION` - Asking y/n confirmation
- `STARTED` / `COMPLETED` / `ERROR` - Status events

## Display Modes

- `rich` - Full colors and formatting (default)
- `simple` - Basic formatting, no colors
- `json` - JSON output for programmatic use
- `quiet` - Minimal output

## Requirements

- Python 3.10+
- Claude CLI installed (`claude` command available)
- Unix-like OS (macOS, Linux) - uses PTY

## Note

This runs Claude CLI with `--dangerously-skip-permissions` flag for autonomous operation.
Use with caution and appropriate safeguards.
