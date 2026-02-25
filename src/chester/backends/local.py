"""Local execution backend."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from .base import Backend, BackendConfig
from ..utils import build_cli_args


class LocalBackend(Backend):
    """Backend for local execution (direct subprocess or singularity)."""

    def generate_command(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate the shell command string for a task.

        Args:
            task: Task dict with a ``params`` sub-dict.
            script: Python script to run.
            python_command: Base python command (default ``python``).
            env: Optional env vars to prepend (``KEY=VAL``).

        Returns:
            The full command string.
        """
        params = task.get("params", {})
        wrapped = self.get_python_command(python_command)
        command = f"{wrapped} {script}"

        # Prepend environment variables
        if env:
            for k, v in env.items():
                command = f"{k}={v} " + command

        # Append CLI args from params
        cli_args = build_cli_args(params)
        if cli_args:
            command += "  " + cli_args

        # Singularity wrapping (on the full command)
        if self.config.singularity:
            command = self.wrap_with_singularity([command])

        return command

    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a full bash script for local execution.

        This is used when singularity or prepare.sh is involved.

        Args:
            task: Task dict with a ``params`` sub-dict.
            script: Python script to run.
            python_command: Base python command.
            env: Optional env vars.

        Returns:
            Full bash script as a string.
        """
        lines: List[str] = []
        lines.append("#!/usr/bin/env bash")
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")

        # Source prepare.sh if configured
        prepare_cmds = self.get_prepare_commands()
        lines.extend(prepare_cmds)

        # Build the inner python command (without singularity wrapping)
        params = task.get("params", {})
        wrapped = self.get_python_command(python_command)
        command = f"{wrapped} {script}"

        if env:
            for k, v in env.items():
                command = f"{k}={v} " + command

        cli_args = build_cli_args(params)
        if cli_args:
            command += "  " + cli_args

        # Singularity wrapping
        if self.config.singularity:
            command = self.wrap_with_singularity([command])

        lines.append(command)

        return "\n".join(lines) + "\n"

    def submit(
        self,
        task: Dict[str, Any],
        script_content: str,
        dry: bool = False,
    ) -> Optional[subprocess.Popen]:
        """Run a task locally via subprocess.

        Args:
            task: Task dict (used for logging context).
            script_content: The bash script content to execute.
            dry: If True, do nothing and return None.

        Returns:
            A ``subprocess.Popen`` object, or None if dry run.
        """
        if dry:
            return None

        proc = subprocess.Popen(
            ["bash", "-c", script_content],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc
