"""
Baseline tests for agentproxy package structure compatibility.

These tests verify that both old and new entry points work correctly
after the package restructuring (PR1).

Old entry points (backward compatibility):
- python cli.py --help
- python server.py --help

New entry points:
- pa --help (requires pip install -e .)
- pa-server --help (requires pip install -e .)
- python -m agentproxy --help
- python -m agentproxy.server --help
"""

import subprocess
import sys
import os
from pathlib import Path


# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestOldEntryPoints:
    """Test backward compatibility with old entry points."""

    def test_old_cli_help(self):
        """Old entry point: python cli.py --help should work"""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"cli.py --help failed: {result.stderr}"
        assert "PA (Proxy Agent)" in result.stdout, "Expected PA description in help output"

    def test_old_cli_list_sessions(self):
        """Old entry point: python cli.py --list-sessions should work"""
        result = subprocess.run(
            [sys.executable, "cli.py", "--list-sessions"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        # Should succeed or exit with 0 (may have no sessions)
        assert result.returncode == 0, f"cli.py --list-sessions failed: {result.stderr}"

    def test_old_server_help(self):
        """Old entry point: python server.py --help should work"""
        result = subprocess.run(
            [sys.executable, "server.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"server.py --help failed: {result.stderr}"
        assert "PA Web Server" in result.stdout, "Expected server description in help output"


class TestNewEntryPoints:
    """Test new package entry points."""

    def test_python_module_cli_help(self):
        """New entry point: python -m agentproxy --help should work"""
        result = subprocess.run(
            [sys.executable, "-m", "agentproxy", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"python -m agentproxy --help failed: {result.stderr}"
        assert "PA (Proxy Agent)" in result.stdout, "Expected PA description in help output"

    def test_python_module_server_help(self):
        """New entry point: python -m agentproxy.server --help should work"""
        result = subprocess.run(
            [sys.executable, "-m", "agentproxy.server", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0, f"python -m agentproxy.server --help failed: {result.stderr}"
        assert "PA Web Server" in result.stdout, "Expected server description in help output"


class TestPackageImports:
    """Test that package can be imported correctly."""

    def test_import_pa(self):
        """Can import PA from agentproxy package"""
        from agentproxy import PA
        assert PA is not None

    def test_import_create_pa(self):
        """Can import create_pa helper"""
        from agentproxy import create_pa
        assert create_pa is not None

    def test_import_list_sessions(self):
        """Can import list_sessions helper"""
        from agentproxy import list_sessions
        assert list_sessions is not None

    def test_import_models(self):
        """Can import core models"""
        from agentproxy import OutputEvent, EventType, ControllerState
        assert OutputEvent is not None
        assert EventType is not None
        assert ControllerState is not None

    def test_import_memory(self):
        """Can import memory system components"""
        from agentproxy import PAMemory, BestPractices, SessionContext, InteractionHistory
        assert PAMemory is not None
        assert BestPractices is not None
        assert SessionContext is not None
        assert InteractionHistory is not None

    def test_import_display(self):
        """Can import display component"""
        from agentproxy import RealtimeDisplay
        assert RealtimeDisplay is not None

    def test_import_process_manager(self):
        """Can import process manager"""
        from agentproxy import ClaudeProcessManager
        assert ClaudeProcessManager is not None

    def test_package_version(self):
        """Package has version defined"""
        from agentproxy import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)


class TestPackageStructure:
    """Test that package structure is correct."""

    def test_package_directory_exists(self):
        """agentproxy package directory exists"""
        pkg_dir = PROJECT_ROOT / "agentproxy"
        assert pkg_dir.exists(), "agentproxy directory should exist"
        assert pkg_dir.is_dir(), "agentproxy should be a directory"

    def test_init_file_exists(self):
        """agentproxy/__init__.py exists"""
        init_file = PROJECT_ROOT / "agentproxy" / "__init__.py"
        assert init_file.exists(), "__init__.py should exist"

    def test_main_file_exists(self):
        """agentproxy/__main__.py exists"""
        main_file = PROJECT_ROOT / "agentproxy" / "__main__.py"
        assert main_file.exists(), "__main__.py should exist"

    def test_pyproject_toml_exists(self):
        """pyproject.toml exists at root"""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist"

    def test_backward_compat_shims_exist(self):
        """Backward compatibility shims exist at root"""
        cli_shim = PROJECT_ROOT / "cli.py"
        server_shim = PROJECT_ROOT / "server.py"
        assert cli_shim.exists(), "cli.py shim should exist at root"
        assert server_shim.exists(), "server.py shim should exist at root"


class TestCoreComponents:
    """Test that core components are present and importable."""

    def test_cli_module_exists(self):
        """CLI module can be imported"""
        from agentproxy import cli
        assert hasattr(cli, 'main'), "CLI should have main function"

    def test_server_module_exists(self):
        """Server module can be imported"""
        from agentproxy import server
        assert hasattr(server, 'main'), "Server should have main function"

    def test_pa_module_exists(self):
        """PA module can be imported"""
        from agentproxy import pa
        assert hasattr(pa, 'PA'), "PA module should have PA class"

    def test_pa_agent_module_exists(self):
        """PA agent module can be imported"""
        from agentproxy import pa_agent
        assert pa_agent is not None

    def test_gemini_client_module_exists(self):
        """Gemini client module can be imported"""
        from agentproxy import gemini_client
        assert gemini_client is not None
