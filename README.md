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

# Interactive mode
python -m interactive_labs --interactive

# With options
python -m interactive_labs -d ./myproject --display simple "Fix bug in app.py"
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
