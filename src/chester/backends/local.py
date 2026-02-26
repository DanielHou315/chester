"""Local execution backend."""
from __future__ import annotations

import subprocess
from typing import Any, Dict, List, Optional

from .base import Backend, BackendConfig


class LocalBackend(Backend):
    """Backend for local execution (direct subprocess or singularity)."""

    def generate_command(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate the shell command string for a task.

        Args:
            task: Task dict with a ``params`` sub-dict.
            script: Python script to run.
            python_command: Base python command (default ``python``).
            env: Optional env vars to prepend (``KEY=VAL``).
            hydra_enabled: Use Hydra override format for args.
            hydra_flags: Hydra flags (e.g. ``{'multirun': True}``).

        Returns:
            The full command string.
        """
        params = task.get("params", {})
        command = self.build_python_command(
            params, script, python_command, env, hydra_enabled, hydra_flags,
        )

        # Host prepare always runs on the host (e.g. direnv, module loads).
        host_parts: List[str] = list(self.get_prepare_commands())

        if self.config.singularity:
            inner: List[str] = []
            inner.extend(self.get_singularity_prepare_commands())
            inner.append(command)
            host_parts.append(self.wrap_with_singularity(inner))
        else:
            host_parts.append(command)

        return " && ".join(host_parts)

    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a full bash script for local execution.

        When singularity is configured, the backend's own prepare.sh runs
        on the host and the singularity prepare.sh runs inside the container.

        Args:
            task: Task dict with a ``params`` sub-dict.
            script: Python script to run.
            python_command: Base python command.
            env: Optional env vars.
            hydra_enabled: Use Hydra override format for args.
            hydra_flags: Hydra flags (e.g. ``{'multirun': True}``).

        Returns:
            Full bash script as a string.
        """
        lines: List[str] = []
        lines.append("#!/usr/bin/env bash")
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")

        # Host prepare always runs on the host (e.g. direnv, module loads).
        lines.extend(self.get_prepare_commands())

        params = task.get("params", {})
        command = self.build_python_command(
            params, script, python_command, env, hydra_enabled, hydra_flags,
        )

        if self.config.singularity:
            inner: List[str] = []
            inner.extend(self.get_singularity_prepare_commands())
            inner.append(command)
            lines.append(self.wrap_with_singularity(inner))
        else:
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
