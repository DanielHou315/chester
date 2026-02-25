"""SSH execution backend."""
from __future__ import annotations

import os
import shlex
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from .base import Backend, BackendConfig
from ..utils import build_cli_args


class SSHBackend(Backend):
    """Backend for remote execution via SSH + nohup."""

    # ------------------------------------------------------------------
    # prepare.sh handling
    # ------------------------------------------------------------------

    def get_prepare_commands(self) -> List[str]:
        """Return source command for prepare.sh, relative to remote_dir."""
        return self._get_remote_prepare_commands()

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a full bash script for SSH-based remote execution.

        The generated script follows this structure:
        1. ``#!/usr/bin/env bash`` header
        2. ``set -x``, ``set -u``, ``set -e``
        3. ``cd {remote_dir}``
        4. Source prepare.sh if configured
        5. Python command with params (wrapped for package manager)
        6. ``touch {log_dir}/.done`` marker on success

        For singularity mode the inner commands (prepare + python + done
        marker) are wrapped with ``singularity exec``.

        Args:
            task: Task dict with ``params`` sub-dict. ``params`` must contain
                  ``log_dir`` for the ``.done`` marker.
            script: Python script to run.
            python_command: Base python command.
            env: Optional env vars to prepend.

        Returns:
            Full bash script as a string.
        """
        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        remote_dir = self.config.remote_dir or "./"

        lines: List[str] = []
        lines.append("#!/usr/bin/env bash")
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")
        lines.append(f"cd {remote_dir}")

        # Build inner commands (may be wrapped by singularity)
        inner: List[str] = []

        # Source prepare.sh
        prepare_cmds = self.get_prepare_commands()
        inner.extend(prepare_cmds)

        # Python command
        wrapped = self.get_python_command(python_command)
        command = f"{wrapped} {script}"

        if env:
            for k, v in env.items():
                command = f"{k}={v} " + command

        cli_args = build_cli_args(params)
        if cli_args:
            command += "  " + cli_args

        inner.append(command)

        # .done marker
        inner.append(f"touch {log_dir}/.done")

        # Singularity wrapping or plain join
        if self.config.singularity:
            lines.append(self.wrap_with_singularity(inner))
        else:
            lines.extend(inner)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        task: Dict[str, Any],
        script_content: str,
        dry: bool = False,
    ) -> Optional[int]:
        """Submit a task for execution on a remote host via SSH.

        Steps:
        1. Create remote log dir via ``ssh host mkdir -p {log_dir}``
        2. Write script to a temp file
        3. Copy script to remote via ``scp``
        4. Execute via ``ssh host nohup bash script > output 2>&1 &``
        5. Save PID to ``.chester_pid``

        Args:
            task: Task dict.
            script_content: The bash script content.
            dry: If True, do nothing and return None.

        Returns:
            Remote PID, or None if dry run.
        """
        if dry:
            return None

        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        host = self.config.host
        exp_name = params.get("exp_name", "chester_job")

        # 1. Create remote log dir
        subprocess.run(
            ["ssh", host, f"mkdir -p {shlex.quote(log_dir)}"],
            check=True,
        )

        # 2. Write script to temp file
        with NamedTemporaryFile(
            mode="w", suffix=".sh", delete=False, prefix="chester_"
        ) as f:
            f.write(script_content)
            local_script = f.name

        try:
            # 3. Copy to remote
            remote_script = f"{log_dir}/chester_run.sh"
            subprocess.run(
                ["scp", local_script, f"{host}:{remote_script}"],
                check=True,
            )

            # 4. Execute via nohup and capture PID
            q_script = shlex.quote(remote_script)
            q_log = shlex.quote(f"{log_dir}/output.log")
            result = subprocess.run(
                [
                    "ssh", host,
                    f"nohup bash {q_script} > {q_log} 2>&1 & echo $!",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            pid = int(result.stdout.strip())

            # 5. Save PID to .chester_pid
            q_pid_file = shlex.quote(f"{log_dir}/.chester_pid")
            subprocess.run(
                ["ssh", host, f"echo {pid} > {q_pid_file}"],
                check=True,
            )

            return pid
        finally:
            os.unlink(local_script)
