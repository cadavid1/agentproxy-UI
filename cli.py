#!/usr/bin/env python3
"""
Backward compatibility shim for 'python cli.py'.

This file maintains backward compatibility with the old entry point.
New installations should use 'pa' command or 'python -m agentproxy' instead.
"""
import sys

if __name__ == "__main__":
    from agentproxy.cli import main
    sys.exit(main())
