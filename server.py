#!/usr/bin/env python3
"""
Backward compatibility shim for 'python server.py'.

This file maintains backward compatibility with the old entry point.
New installations should use 'pa-server' command or 'python -m agentproxy.server' instead.
"""
import sys

if __name__ == "__main__":
    from agentproxy.server import main
    sys.exit(main())
