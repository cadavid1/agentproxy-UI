# Agent Proxy

An AI supervisor layer that orchestrates coding agents (Claude Code, Cursor, etc.) to complete complex development tasks autonomously.

## How It Works

Agent Proxy sits between human developers and AI coding agents, acting as an intelligent supervisor that:

1. **Maintains Context** - Stores project context, coding standards, and session history
2. **Supervises & Assigns** - Breaks down tasks and delegates to coding agents
3. **Unblocks** - Auto-responds to agent clarification requests
4. **Verifies** - Reviews agent output for correctness and quality
5. **Aligns & Refocuses** - Keeps agents on track toward the goal

### Architecture

![Architecture](docs/images/architecture.jpg)

The proxy maintains three types of context:
- **Project Context**: Company mission, OKRs, competitive analysis, mockups
- **Guidance & Practice**: Coding standards, QA checklists, team preferences
- **Sprint Session Context**: Current tasks and their status

### Execution Flow

![Sequence](docs/images/sequence.jpg)

1. Human provides one-time project context setup
2. Human assigns high-level tasks
3. Proxy Agent thinks, plans, and invokes the Coding Agent
4. Coding Agent executes steps, may ask for clarification
5. Proxy Agent auto-replies or redirects as needed
6. Proxy verifies results and updates task status
7. Final delivery presented to human

## Setup
```bash
pip install fastapi uvicorn pydantic
echo "GEMINI_API_KEY=your_key" > .env
```

## Mode 1: CLI
```bash
python cli.py "Create hello.py"
python cli.py -d ./myproject "Fix bug"
python cli.py --add-screenshot design.png "Match this UI"
```

```bash
# Add alias to ~/.zshrc or ~/.bashrc
alias ccp='python /Users/ethw/Desktop/GIT-Aertoria/agentproxy/cli.py'

# Set working directory with task (saves to ~/.ccp_config for future tasks)
ccp --set-workdir ./myproject "Fix bug"
ccp --show-workdir  # View current default

# Example with screenshot
ccp --add-screenshot design.png "Build a web app matching this UI"
```

## Mode 2: Server
```bash
python server.py --port 8080
```

```bash
# Start task
curl -N -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Create hello.py", "working_dir": "./sandbox"}'

# With screenshot
curl -N -X POST http://localhost:8080/task \
  -H "Content-Type: application/json" \
  -d '{"task": "Match this UI", "screenshots": [{"path": "/path/to/design.png"}]}'

# Other endpoints
curl http://localhost:8080/health
curl http://localhost:8080/sessions
```

## Requirements
- Python 3.9+
- Claude CLI (`claude` command)
- Gemini API key
