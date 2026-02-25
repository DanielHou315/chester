"""SLURM batch execution backend."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional

from .base import Backend, BackendConfig, SlurmConfig
from ..utils import build_cli_args


class SlurmBackend(Backend):
    """Backend for SLURM cluster execution."""

    # ------------------------------------------------------------------
    # prepare.sh handling (same as SSH -- relative to remote_dir)
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
        slurm_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a SLURM batch script.

        Structure:
        1. SBATCH header from ``SlurmConfig.to_sbatch_header()``
        2. ``#SBATCH -o / -e / --job-name`` directives
        3. ``set -x``, ``set -u``, ``set -e``
        4. ``srun hostname``
        5. ``cd {remote_dir}``
        6. ``module load`` for each module and cuda_module
        7. Inner commands (prepare.sh + python + .done marker), optionally
           wrapped with singularity

        Args:
            task: Task dict with ``params`` sub-dict.  ``params`` must contain
                  ``log_dir`` for SBATCH output directives and ``.done`` marker.
            script: Python script to run.
            python_command: Base python command (default ``python``; for SLURM
                  this is often ``srun python``).
            env: Optional env vars.
            slurm_overrides: Optional dict to override SLURM params per-experiment.

        Returns:
            Full SLURM batch script as a string.
        """
        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        exp_name = params.get("exp_name", task.get("exp_name", "chester_job"))
        remote_dir = self.config.remote_dir or "./"

        # ---- SBATCH header ----
        slurm_cfg = self.config.slurm or SlurmConfig()
        if slurm_overrides:
            slurm_cfg = slurm_cfg.with_overrides(slurm_overrides)

        header = slurm_cfg.to_sbatch_header()

        lines: List[str] = []
        lines.append(header)

        # Per-job SBATCH directives
        lines.append(f"#SBATCH -o {log_dir}/slurm.out")
        lines.append(f"#SBATCH -e {log_dir}/slurm.err")
        lines.append(f"#SBATCH --job-name={exp_name}")

        # ---- Bash preamble ----
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")
        lines.append("srun hostname")
        lines.append(f"cd {remote_dir}")

        # ---- Module loads ----
        for mod in self.config.modules:
            lines.append(f"module load {mod}")
        if self.config.cuda_module:
            lines.append(f"module load {self.config.cuda_module}")

        # ---- Inner commands (may be wrapped by singularity) ----
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

        # Singularity wrapping or plain append
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
        """Submit a SLURM batch job.

        Steps:
        1. Write script locally to ``{local_log_dir}/chester_slurm.sh``
        2. Create remote log directory via SSH
        3. SCP the batch script to remote
        4. SSH into the host and run ``sbatch``

        Args:
            task: Task dict.  ``params`` must contain ``log_dir`` (the
                  *remote* log dir where the script will be placed).
            script_content: The full SLURM batch script content.
            dry: If True, print the script but do not submit.
        """
        if dry:
            print(script_content)
            return None

        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        exp_name = params.get("exp_name", task.get("exp_name", "chester_job"))
        host = self.config.host

        # Determine local log dir for writing the script before SCP
        local_log_dir = task.get("_local_log_dir", log_dir)

        # 1. Write script locally
        os.makedirs(local_log_dir, exist_ok=True)
        local_script = os.path.join(local_log_dir, "chester_slurm.sh")
        with open(local_script, "w") as f:
            f.write(script_content)

        # 2. Create remote log dir
        subprocess.run(
            ["ssh", host, f"mkdir -p {shlex.quote(log_dir)}"],
            check=True,
        )

        # 3. SCP script to remote
        remote_script = os.path.join(log_dir, "chester_slurm.sh")
        subprocess.run(
            ["scp", local_script, f"{host}:{remote_script}"],
            check=True,
        )

        # 4. Submit via sbatch
        print(f"[chester] Submitting SLURM job on {host}: {exp_name}")
        print(f"[chester] Remote script: {remote_script}")
        result = subprocess.run(
            ["ssh", host, f"sbatch {shlex.quote(remote_script)}"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[chester] sbatch failed: {result.stderr.strip()}")
            raise RuntimeError(
                f"sbatch failed on {host}: {result.stderr.strip()}"
            )
        print(f"[chester] {result.stdout.strip()}")

        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if match:
            job_id = int(match.group(1))
            print(f"[chester] SLURM job ID: {job_id}")
            return job_id
        else:
            print(f"[chester] Warning: could not parse SLURM job ID from: {result.stdout.strip()}")
            return None
