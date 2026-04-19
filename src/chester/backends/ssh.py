"""SSH execution backend."""
from __future__ import annotations

import os
import shlex
import subprocess
import textwrap
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

from .base import (
    Backend,
    BackendConfig,
    _build_worktree_setup_commands,
    _rewrite_mounts_for_worktrees,
)


class SSHBackend(Backend):
    """Backend for remote execution via SSH + nohup.

    When ``batch_gpu`` is set in the backend config, submit() accumulates
    scripts instead of firing them immediately.  Call flush_batch() after
    all variants are submitted to dispatch a single meta-job that runs K
    GPU workers sharing a flock-based queue.
    """

    def __init__(self, config: BackendConfig, project_config: Dict[str, Any]):
        super().__init__(config, project_config)
        self._pending: List[Tuple[Dict[str, Any], str]] = []

    # ------------------------------------------------------------------
    # prepare.sh handling
    # ------------------------------------------------------------------

    def get_prepare_commands(self) -> List[str]:
        """Return source command for prepare.sh, relative to remote_dir."""
        return self._get_remote_prepare_commands()

    # ------------------------------------------------------------------
    # GPU detection
    # ------------------------------------------------------------------

    def _resolve_gpu_ids(self) -> List[str]:
        """Return the list of GPU device IDs to use for batch workers.

        Resolution order:
        1. ``cuda_visible_devices`` config string (e.g. "0,1,2")
        2. ``$CUDA_VISIBLE_DEVICES`` on the remote host
        3. ``nvidia-smi`` on the remote host
        4. ``range(batch_gpu)`` fallback if remote query fails
        """
        cap = self.config.batch_gpu  # optional upper bound

        if self.config.cuda_visible_devices:
            ids = [g.strip() for g in self.config.cuda_visible_devices.split(",") if g.strip()]
            return ids[:cap] if cap else ids

        host = self.config.host
        cmd = (
            'if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then '
            '  printf "%s" "$CUDA_VISIBLE_DEVICES" | tr "," "\\n"; '
            'else '
            '  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null; '
            'fi'
        )
        result = subprocess.run(["ssh", host, cmd], capture_output=True, text=True)
        if result.returncode == 0:
            ids = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            if ids:
                return ids[:cap] if cap else ids

        if cap:
            return [str(i) for i in range(cap)]

        raise RuntimeError(
            f"[chester] Could not detect GPUs on {host}. "
            "Set cuda_visible_devices or batch_gpu in the backend config."
        )

    # ------------------------------------------------------------------
    # Script generation
    # ------------------------------------------------------------------

    def generate_script(
        self,
        task: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
        serial_steps: Optional[List[Tuple[str, list]]] = None,
        submodule_worktrees: Optional[Dict[str, str]] = None,
        submodule_resolved_commits: Optional[Dict[str, str]] = None,
    ) -> str:
        """Generate a full bash script for SSH-based remote execution.

        The generated script follows this structure:
        1. ``#!/usr/bin/env bash`` header
        2. ``set -x``, ``set -u``, ``set -e``
        3. ``cd {remote_dir}``
        4. Source prepare.sh if configured
        5. Python command with params (wrapped for package manager)

        For singularity mode the inner commands (prepare + python) are
        wrapped with ``singularity exec``.

        For ``serial_steps``, multiple python commands are generated in the
        same script (one per step), each wrapped independently with
        singularity if configured.

        Args:
            task: Task dict with ``params`` sub-dict. ``params`` must contain
                  ``log_dir`` for the output log path.
            script: Python script to run.
            python_command: Base python command.
            env: Optional env vars to prepend.
            hydra_enabled: Use Hydra override format for args.
            hydra_flags: Hydra flags (e.g. ``{'multirun': True}``).
            serial_steps: List of (key, [val1, val2, ...]) for order='serial'.
                  Generates one command per step value in the same script.
            submodule_worktrees: Optional mapping of submodule path to absolute
                remote worktree path. When set, mounts for these submodules are
                redirected to the worktree. Requires ``submodule_resolved_commits``.
            submodule_resolved_commits: Optional mapping of submodule path to full
                40-char SHA. Required when ``submodule_worktrees`` is set.

        Returns:
            Full bash script as a string.
        """
        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        remote_dir = self.config.remote_dir or "./"

        lines: List[str] = []
        lines.append("#!/usr/bin/env bash")
        # Redirect xtrace to a separate file so output.log stays clean
        lines.append(f"exec 19>{log_dir}/chester_xtrace.log")
        lines.append("BASH_XTRACEFD=19")
        lines.append("set -x")
        lines.append("set -u")
        lines.append("set -e")
        lines.append(f"cd {remote_dir}")

        # Backend prepare.sh always runs on the host.
        prepare_cmds = self.get_prepare_commands()
        lines.extend(prepare_cmds)

        # ---- Submodule worktree setup (before overlay and singularity) ----
        if submodule_worktrees:
            lines.extend(_build_worktree_setup_commands(
                submodule_worktrees, submodule_resolved_commits, remote_dir
            ))
            rewritten_mounts = _rewrite_mounts_for_worktrees(
                self.config.singularity.mounts if self.config.singularity else [],
                submodule_worktrees,
                remote_dir,
            )
        else:
            rewritten_mounts = None

        # Create overlay image if needed (before singularity exec).
        lines.extend(self.get_overlay_setup_commands())

        if serial_steps:
            # Generate one command per serial step
            from ..run_exp import _iter_serial_overrides
            for step_overrides in _iter_serial_overrides(serial_steps):
                command = self.build_python_command(
                    params, script, python_command, env,
                    hydra_enabled, hydra_flags,
                    extra_overrides=step_overrides,
                )
                if self.config.singularity:
                    inner: List[str] = list(self.get_singularity_prepare_commands())
                    inner.append(command)
                    lines.append(self.wrap_with_singularity(inner, mounts_override=rewritten_mounts))
                else:
                    lines.append(command)
        else:
            command = self.build_python_command(
                params, script, python_command, env,
                hydra_enabled, hydra_flags,
            )
            if self.config.singularity:
                inner: List[str] = list(self.get_singularity_prepare_commands())
                inner.append(command)
                lines.append(self.wrap_with_singularity(inner, mounts_override=rewritten_mounts))
            else:
                lines.append(command)

        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        task: Dict[str, Any],
        script_content: str,
        dry: bool = False,
    ) -> None:
        """Submit a task for execution on a remote host via SSH.

        In normal mode (no batch_gpu), submits immediately:
        1. Create remote log dir via ``ssh host mkdir -p {log_dir}``
        2. Copy script to remote via ``scp``
        3. Execute via ``ssh host nohup bash script > output 2>&1 &``

        In batch mode (batch_gpu set), accumulates the script instead.
        Call :meth:`flush_batch` after all variants are submitted.

        Args:
            task: Task dict.
            script_content: The bash script content.
            dry: If True, do nothing and return None.

        Returns:
            None.
        """
        if self.config.batch_gpu:
            if not dry:
                self._pending.append((task, script_content))
            return None

        return self._submit_single(task, script_content, dry)

    def _submit_single(
        self,
        task: Dict[str, Any],
        script_content: str,
        dry: bool = False,
    ) -> None:
        if dry:
            return None

        params = task.get("params", {})
        log_dir = params.get("log_dir", "")
        host = self.config.host

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

            # 4. Execute via nohup
            q_script = shlex.quote(remote_script)
            q_log = shlex.quote(f"{log_dir}/output.log")
            subprocess.run(
                ["ssh", host, f"nohup bash {q_script} > {q_log} 2>&1 &"],
                check=True,
            )
        finally:
            os.unlink(local_script)

    # ------------------------------------------------------------------
    # Batch flush
    # ------------------------------------------------------------------

    def flush_batch(self) -> None:
        """Upload all pending scripts and dispatch a single GPU-worker meta-job.

        GPU IDs are resolved via :meth:`_resolve_gpu_ids` (config string →
        remote env → nvidia-smi → range(batch_gpu) fallback).  One worker
        process is spawned per GPU; workers atomically pop scripts from a
        shared queue file using ``flock``.
        """
        if not self._pending:
            return

        host = self.config.host
        gpu_ids = self._resolve_gpu_ids()
        n_gpus = len(gpu_ids)

        # Use the parent of the first log_dir as the batch coordination dir
        first_log_dir = self._pending[0][0].get("params", {}).get("log_dir", "")
        batch_dir = os.path.dirname(first_log_dir)

        # 1. Create all remote log dirs + batch dir
        all_dirs = " ".join(
            shlex.quote(task.get("params", {}).get("log_dir", ""))
            for task, _ in self._pending
        )
        subprocess.run(
            ["ssh", host, f"mkdir -p {shlex.quote(batch_dir)} {all_dirs}"],
            check=True,
        )

        # 2. Upload all scripts, collect remote paths for the queue
        queue_entries: List[str] = []
        for task, script_content in self._pending:
            log_dir = task.get("params", {}).get("log_dir", "")
            remote_script = f"{log_dir}/chester_run.sh"
            subprocess.run(
                ["ssh", host, f"cat > {shlex.quote(remote_script)}"],
                input=script_content, text=True, check=True,
            )
            queue_entries.append(remote_script)

        # 3. Write queue file (one script path per line)
        queue_file = f"{batch_dir}/chester_queue.txt"
        subprocess.run(
            ["ssh", host, f"cat > {shlex.quote(queue_file)}"],
            input="\n".join(queue_entries) + "\n", text=True, check=True,
        )

        # 4. Upload gpu_worker.sh
        worker_file = f"{batch_dir}/chester_gpu_worker.sh"
        subprocess.run(
            ["ssh", host, f"cat > {shlex.quote(worker_file)}"],
            input=self._gpu_worker_script(), text=True, check=True,
        )

        # 5. Upload meta_job.sh
        meta_file = f"{batch_dir}/chester_meta_job.sh"
        subprocess.run(
            ["ssh", host, f"cat > {shlex.quote(meta_file)}"],
            input=self._meta_job_script(batch_dir, queue_file, worker_file, gpu_ids),
            text=True, check=True,
        )

        # 6. Fire meta-job via nohup
        q_meta = shlex.quote(meta_file)
        q_log = shlex.quote(f"{batch_dir}/chester_batch_output.log")
        result = subprocess.run(
            ["ssh", host, f"nohup bash {q_meta} > {q_log} 2>&1 & echo $!"],
            capture_output=True, text=True, check=True,
        )
        pid = int(result.stdout.strip())

        # Save batch PID alongside queue for manual inspection
        subprocess.run(
            ["ssh", host,
             f"echo {pid} > {shlex.quote(batch_dir + '/.chester_batch_pid')}"],
            check=True,
        )

        print(
            f"[chester] Batch submitted: {len(self._pending)} scripts "
            f"across {n_gpus} GPUs {gpu_ids} on {host} (PID={pid})\n"
            f"[chester] Tail batch log: ssh {host} tail -f {q_log}"
        )
        self._pending.clear()

    # ------------------------------------------------------------------
    # Batch script templates
    # ------------------------------------------------------------------

    def _gpu_worker_script(self) -> str:
        return textwrap.dedent("""\
            #!/usr/bin/env bash
            # Chester GPU batch worker — pops scripts from a shared queue using flock.
            # Usage: bash chester_gpu_worker.sh <gpu_id> <queue_file>
            GPU=$1
            QUEUE=$2
            LOCK="${QUEUE}.lock"

            echo "[chester-worker gpu=$GPU] Starting"
            while true; do
                script=$(
                    flock -x "$LOCK" bash -c '
                        line=$(head -1 "$1" 2>/dev/null)
                        [ -z "$line" ] && exit 0
                        tail -n +2 "$1" > "${1}.tmp" && mv "${1}.tmp" "$1"
                        printf "%s" "$line"
                    ' _ "$QUEUE"
                )
                [ -z "$script" ] && break
                echo "[chester-worker gpu=$GPU] Running: $script"
                CUDA_VISIBLE_DEVICES=$GPU bash "$script" \
                    || echo "[chester-worker gpu=$GPU] Script failed (continuing): $script"
            done
            echo "[chester-worker gpu=$GPU] Queue empty, done."
        """)

    def _meta_job_script(
        self,
        batch_dir: str,
        queue_file: str,
        worker_file: str,
        gpu_ids: List[str],
    ) -> str:
        lines = [
            "#!/usr/bin/env bash",
            f"QUEUE={shlex.quote(queue_file)}",
            f"WORKER={shlex.quote(worker_file)}",
            "",
            "pids=()",
        ]
        for gpu_id in gpu_ids:
            log_file = shlex.quote(f"{batch_dir}/chester_worker_{gpu_id}.log")
            lines.append(f'bash "$WORKER" {shlex.quote(str(gpu_id))} "$QUEUE" > {log_file} 2>&1 &')
            lines.append("pids+=($!)")
        lines += [
            "",
            'for pid in "${pids[@]}"; do',
            '    wait "$pid"',
            "done",
            'echo "[chester] All GPU workers finished."',
        ]
        return "\n".join(lines) + "\n"
