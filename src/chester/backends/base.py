"""Backend configuration and base classes."""
from __future__ import annotations

import copy
import os
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional


VALID_BACKEND_TYPES = ("local", "ssh", "slurm")


@dataclass
class SingularityConfig:
    """Singularity container configuration."""
    image: str
    mounts: List[str] = field(default_factory=list)
    gpu: bool = False
    build_script: Optional[str] = None
    workdir: Optional[str] = None
    prepare: Optional[str] = None  # prepare script to source *inside* the container
    enabled: bool = True  # when False, config is present but not used by default
    writable_tmpfs: bool = False  # --writable-tmpfs: allow writes via in-memory overlay
    overlay: Optional[str] = None  # path to persistent ext3 overlay image
    overlay_size: int = 10240  # overlay size in MB (default 10 GB)
    fakeroot: bool = True  # --fakeroot: run as fake root inside the container
    pid_namespace: bool = False  # --pid: isolate PID namespace so all child processes are killed on exit


@dataclass
class SlurmConfig:
    """SLURM job parameters with per-experiment override support.

    Known fields (partition, time, gpus, etc.) are emitted as their
    corresponding ``#SBATCH`` directives.  Any additional key-value pairs
    — passed via ``slurm_overrides`` in the launcher or the YAML config —
    are stored in ``extras`` and emitted as ``#SBATCH --key=value``
    directives, allowing arbitrary sbatch options without code changes.
    """
    partition: Optional[str] = None
    time: Optional[str] = None
    nodes: int = 1
    gpus: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    mem_per_gpu: Optional[str] = None
    ntasks_per_node: Optional[int] = None
    email_begin: Optional[str] = None  # email address for BEGIN notifications
    email_end: Optional[str] = None    # email address for END/FAIL notifications
    extras: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, overrides: Dict[str, Any]) -> SlurmConfig:
        """Return a new SlurmConfig with specified fields overridden.

        Known dataclass fields are set directly.  Unknown keys are stored
        in ``extras`` and emitted as arbitrary ``#SBATCH`` directives.
        """
        new = copy.deepcopy(self)
        known_fields = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key in known_fields:
                setattr(new, key, value)
            else:
                new.extras[key] = value
        return new

    def to_sbatch_header(self) -> str:
        """Generate #SBATCH header from structured config."""
        lines = ["#!/usr/bin/env bash"]
        if self.nodes is not None:
            lines.append(f"#SBATCH --nodes={self.nodes}")
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.gpus is not None:
            lines.append(f"#SBATCH --gpus={self.gpus}")
        if self.cpus_per_gpu is not None:
            lines.append(f"#SBATCH --cpus-per-gpu={self.cpus_per_gpu}")
        if self.mem_per_gpu:
            lines.append(f"#SBATCH --mem-per-gpu={self.mem_per_gpu}")
        if self.ntasks_per_node is not None:
            lines.append(f"#SBATCH --ntasks-per-node={self.ntasks_per_node}")
        # Email notifications
        mail_types: List[str] = []
        mail_user = None
        if self.email_begin:
            mail_types.append("BEGIN")
            mail_user = self.email_begin
        if self.email_end:
            mail_types.extend(["END", "FAIL"])
            mail_user = self.email_end
        if mail_types and mail_user:
            lines.append(f"#SBATCH --mail-user={mail_user}")
            lines.append(f"#SBATCH --mail-type={','.join(mail_types)}")
        # Arbitrary extra directives
        for key, value in self.extras.items():
            k = key if key.startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    lines.append(f"#SBATCH {k}")
            else:
                lines.append(f"#SBATCH {k}={value}")
        return "\n".join(lines)


@dataclass
class BackendConfig:
    """Configuration for a single backend."""
    name: str
    type: str  # "local", "ssh", "slurm"
    host: Optional[str] = None
    remote_dir: Optional[str] = None
    prepare: Optional[str] = None  # Path to prepare.sh
    modules: List[str] = field(default_factory=list)
    cuda_module: Optional[str] = None
    slurm: Optional[SlurmConfig] = None
    singularity: Optional[SingularityConfig] = None


def parse_backend_config(name: str, raw: Dict[str, Any]) -> BackendConfig:
    """Parse a raw dict from config.yaml into a BackendConfig."""
    backend_type = raw.get("type")
    if backend_type not in VALID_BACKEND_TYPES:
        raise ValueError(
            f"Unknown backend type '{backend_type}' for backend '{name}'. "
            f"Valid types: {VALID_BACKEND_TYPES}"
        )

    # Validate required fields for remote backends
    if backend_type in ("ssh", "slurm"):
        if not raw.get("host"):
            raise ValueError(
                f"Backend '{name}' (type={backend_type}) requires 'host'"
            )
        if not raw.get("remote_dir"):
            raise ValueError(
                f"Backend '{name}' (type={backend_type}) requires 'remote_dir'"
            )

    # Parse singularity config
    sing_raw = raw.get("singularity")
    singularity = None
    if sing_raw:
        singularity = SingularityConfig(
            image=sing_raw["image"],
            mounts=sing_raw.get("mounts", []),
            gpu=sing_raw.get("gpu", False),
            build_script=sing_raw.get("build_script"),
            workdir=sing_raw.get("workdir"),
            prepare=sing_raw.get("prepare"),
            enabled=sing_raw.get("enabled", True),
            writable_tmpfs=sing_raw.get("writable_tmpfs", False),
            overlay=sing_raw.get("overlay"),
            overlay_size=sing_raw.get("overlay_size", 10240),
            fakeroot=sing_raw.get("fakeroot", True),
            pid_namespace=sing_raw.get("pid_namespace", False),
        )

    # Parse SLURM config
    slurm_raw = raw.get("slurm")
    slurm = None
    if slurm_raw:
        known_slurm_keys = {f.name for f in fields(SlurmConfig)}
        known_kwargs = {}
        extras = {}
        for k, v in slurm_raw.items():
            if k in known_slurm_keys:
                known_kwargs[k] = v
            else:
                extras[k] = v
        slurm = SlurmConfig(**known_kwargs, extras=extras)

    return BackendConfig(
        name=name,
        type=backend_type,
        host=raw.get("host"),
        remote_dir=raw.get("remote_dir"),
        prepare=raw.get("prepare"),
        modules=raw.get("modules", []),
        cuda_module=raw.get("cuda_module"),
        slurm=slurm,
        singularity=singularity,
    )


class Backend(ABC):
    """Abstract base class for execution backends."""

    def __init__(self, config: BackendConfig, project_config: Dict[str, Any]):
        self.config = config
        self.project_config = project_config

    @abstractmethod
    def generate_script(self, task: Dict[str, Any], **kwargs) -> str:
        """Generate the execution script for a task.

        Returns the full script content as a string.
        """
        ...

    @abstractmethod
    def submit(self, task: Dict[str, Any], script: str, dry: bool = False) -> None:
        """Submit a task for execution."""
        ...

    def get_prepare_commands(self) -> List[str]:
        """Read prepare.sh and return a source command for it."""
        if not self.config.prepare:
            return []
        prepare_path = self.config.prepare
        project_path = self.project_config.get("project_path", "")
        if not os.path.isabs(prepare_path):
            prepare_path = os.path.join(project_path, prepare_path)
        if not os.path.exists(prepare_path):
            raise FileNotFoundError(
                f"prepare script not found: {prepare_path} "
                f"(backend '{self.config.name}')"
            )
        return [f"source {prepare_path}"]

    def get_python_command(self, base: str = "python") -> str:
        """Wrap python command based on package manager.

        When singularity is active the container's own venv is on PATH,
        so we skip the ``uv run`` wrapper.
        """
        if self.config.singularity:
            return base
        pkg = self.project_config.get("package_manager", "python")
        if pkg == "uv":
            return f"uv run {base}"
        else:
            return base

    def _get_remote_prepare_commands(self) -> List[str]:
        """Resolve prepare.sh relative to remote_dir (for remote backends)."""
        if not self.config.prepare:
            return []
        prepare_path = self.config.prepare
        remote_dir = self.config.remote_dir or ""
        if not os.path.isabs(prepare_path):
            prepare_path = os.path.join(remote_dir, prepare_path)
        return [f"source {prepare_path}"]

    def build_python_command(
        self,
        params: Dict[str, Any],
        script: str,
        python_command: str = "python",
        env: Optional[Dict[str, str]] = None,
        hydra_enabled: bool = False,
        hydra_flags: Optional[Dict[str, Any]] = None,
        extra_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the inner python command string with arguments.

        This is the command *before* any backend wrapping (singularity,
        prepare.sh, sbatch header, ssh, etc.).

        When ``hydra_enabled=False``, args use ``--key value`` format
        (via :func:`~chester.utils.build_cli_args`).

        When ``hydra_enabled=True``, args use ``key=value`` Hydra override
        format (via :func:`~chester.hydra_utils.build_hydra_args`).

        Args:
            params: Task parameter dict.
            script: Python script/module to run.
            python_command: Base python command (e.g. ``"python"``).
            env: Optional env vars to prepend as ``KEY=VAL``.
            hydra_enabled: Use Hydra override format instead of CLI args.
            hydra_flags: Hydra flags (e.g. ``{'multirun': True}``).
                         Only used when ``hydra_enabled=True``.
            extra_overrides: Optional dict of key/value pairs to merge on top
                             of the variant before formatting Hydra overrides.
                             Only used when ``hydra_enabled=True``.

        Returns:
            The full command string.
        """
        wrapped = self.get_python_command(python_command)
        command = f"{wrapped} {script}"

        if env:
            for k, v in env.items():
                command = f"{k}={v} " + command

        if hydra_enabled:
            from ..hydra_utils import build_hydra_args
            args = build_hydra_args(params, hydra_flags, extra_overrides=extra_overrides)
        else:
            from ..utils import build_cli_args
            args = build_cli_args(params)

        if args:
            command += " " + args

        return command

    def get_singularity_prepare_commands(self) -> List[str]:
        """Return source command for the singularity-specific prepare script.

        This runs *inside* the container, before the python command.
        The path is resolved against ``workdir`` (container-side) if relative.
        """
        sing = self.config.singularity
        if not sing or not sing.prepare:
            return []
        prepare_path = sing.prepare
        if not os.path.isabs(prepare_path) and sing.workdir:
            prepare_path = os.path.join(sing.workdir, prepare_path)
        return [f"source {prepare_path}"]

    def get_overlay_setup_commands(self) -> List[str]:
        """Return shell commands to create the overlay image if it doesn't exist.

        The overlay file is created lazily on first use.  Commands are
        returned as separate lines for inclusion in bash scripts.
        """
        sing = self.config.singularity
        if not sing or not sing.overlay:
            return []
        overlay = sing.overlay
        project_path = self.project_config.get("project_path", "")
        if not os.path.isabs(overlay):
            overlay = os.path.join(project_path, overlay)
        size = sing.overlay_size
        return [
            f"if [ ! -f {overlay} ]; then",
            f'  echo "[chester] Creating singularity overlay ({size} MB): {overlay}"',
            f'  mkdir -p "$(dirname {overlay})"',
            f"  singularity overlay create --size {size} {overlay}",
            f"fi",
        ]

    def wrap_with_singularity(
        self,
        commands: List[str],
        mounts_override: Optional[List[str]] = None,
    ) -> str:
        """Wrap a list of commands with singularity exec if configured."""
        sing = self.config.singularity
        if not sing:
            return " && ".join(commands)

        project_path = self.project_config.get("project_path", "")

        parts = ["singularity", "exec"]
        effective_mounts = mounts_override if mounts_override is not None else sing.mounts
        for m in effective_mounts:
            # Resolve relative mount sources against project root
            if ":" in m:
                src, dst = m.split(":", 1)
                # Leave ~ and $ prefixed paths for bash expansion at runtime
                if not os.path.isabs(src) and not src.startswith(("~", "$")):
                    src = os.path.join(project_path, src)
                parts.extend(["-B", f"{src}:{dst}"])
            else:
                parts.extend(["-B", m])
        if sing.fakeroot:
            parts.append("--fakeroot")
        if sing.gpu:
            parts.append("--nv")
        if sing.writable_tmpfs:
            parts.append("--writable-tmpfs")
        if sing.pid_namespace:
            parts.append("--pid")
        # Persistent ext3 overlay
        if sing.overlay:
            overlay = sing.overlay
            if not os.path.isabs(overlay):
                overlay = os.path.join(project_path, overlay)
            parts.extend(["--overlay", overlay])
        if sing.workdir:
            parts.extend(["--pwd", sing.workdir])

        # Resolve relative image path against project root
        image = sing.image
        if not os.path.isabs(image):
            image = os.path.join(project_path, image)
        parts.append(image)

        # Filter out comment-only lines — they break && joining because
        # bash treats everything after # as a comment to end-of-line.
        executable = [c for c in commands if not c.lstrip().startswith("#")]
        inner = " && ".join(executable)
        parts.extend(["/bin/bash", "-c", shlex.quote(inner)])
        return " ".join(parts)


def _rewrite_mounts_for_worktrees(
    mounts: List[str],
    submodule_worktrees: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Rewrite mount sources that fall under a pinned submodule worktree.

    For each mount whose host-side source resolves to a path under a pinned
    submodule, replaces the submodule root prefix with the worktree path.

    The ``/`` suffix guard in the prefix check prevents false matches
    (e.g. submodule ``IsaacLabTactile`` does NOT match ``IsaacLabTactile_v2``).

    Mounts whose source starts with ``~`` or ``$`` are left untouched
    (shell-expanded at runtime, cannot be statically resolved).

    NOTE: Returns absolute worktree paths (not $CHESTER_WT_N bash variable
    references). The spec describes bash variable refs in mounts, but using
    absolute paths is functionally equivalent and simpler — the path is fully
    known at generation time. The CHESTER_WT_N variables are still emitted
    for git worktree add and cleanup; they just aren't needed in the -B flags.

    Args:
        mounts: List of mount strings in ``src:dst`` or bare ``src`` format.
        submodule_worktrees: Mapping of submodule path (relative to project
            root) to absolute remote worktree path.
        remote_dir: Absolute path of the remote project root.

    Returns:
        New list of mount strings with sources rewritten where applicable.
        The input list is not mutated.
    """
    result = []
    # Pre-compute absolute submodule paths once
    abs_submodules = {
        sub: os.path.normpath(os.path.join(remote_dir, sub))
        for sub in submodule_worktrees
    }

    for mount in mounts:
        if ":" in mount:
            src, dst = mount.split(":", 1)
        else:
            src, dst = mount, None

        # Leave shell-expanded paths untouched
        if src.startswith(("~", "$")):
            result.append(mount)
            continue

        # Resolve relative src to absolute remote path
        if not os.path.isabs(src):
            abs_src = os.path.normpath(os.path.join(remote_dir, src))
        else:
            abs_src = os.path.normpath(src)

        new_src = None
        for sub_path, wt_path in submodule_worktrees.items():
            abs_sub = abs_submodules[sub_path]
            if abs_src == abs_sub:
                new_src = wt_path
                break
            # Explicit /sep guard prevents IsaacLabTactile_v2 matching IsaacLabTactile
            if abs_src.startswith(abs_sub + os.sep):
                suffix = abs_src[len(abs_sub):]  # includes leading sep
                new_src = wt_path + suffix
                break

        if new_src is None:
            # No submodule matched — return the original mount string unchanged
            result.append(mount)
        elif dst is not None:
            result.append(f"{new_src}:{dst}")
        else:
            result.append(new_src)

    return result


def _build_worktree_setup_commands(
    submodule_worktrees: Dict[str, str],
    resolved_commits: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Return bash lines for worktree variable assignments, trap, and git worktree add.

    Emits CHESTER_WT_0, CHESTER_WT_1, ... in dict insertion order.
    The same ordering is relied upon by _rewrite_mounts_for_worktrees()
    and _build_worktree_cleanup_commands().

    Args:
        submodule_worktrees: {submodule_path: abs_remote_worktree_path}
        resolved_commits: {submodule_path: full_40char_sha}
        remote_dir: Absolute remote project root path.

    Returns:
        List of bash lines to inject into the host-side script.
    """
    lines = ["# --- chester: submodule worktree setup ---"]

    # Validate that all submodule worktrees have a corresponding commit SHA
    missing = [sub for sub in submodule_worktrees if sub not in resolved_commits]
    if missing:
        raise ValueError(
            f"[chester] _build_worktree_setup_commands: missing resolved commits for: {missing}"
        )

    # Variable assignments
    for i, (sub, wt_path) in enumerate(submodule_worktrees.items()):
        lines.append(f'CHESTER_WT_{i}="{wt_path}"')

    # Cleanup function
    lines.append("")
    lines.append("_chester_wt_cleanup() {")
    for i, (sub, _wt_path) in enumerate(submodule_worktrees.items()):
        abs_sub = os.path.normpath(os.path.join(remote_dir, sub))
        lines.append(
            f'    git -C "{abs_sub}" worktree remove --force "$CHESTER_WT_{i}" 2>/dev/null || true'
        )
    lines.append("}")

    # Traps: EXIT always fires; INT/TERM suppress EXIT re-fire then call cleanup
    lines.append("trap '_chester_wt_cleanup' EXIT")
    lines.append("trap 'trap - EXIT; _chester_wt_cleanup; exit 130' INT")
    lines.append("trap 'trap - EXIT; _chester_wt_cleanup; exit 143' TERM")
    lines.append("")

    # Worktree creation
    for i, (sub, wt_path) in enumerate(submodule_worktrees.items()):
        sha = resolved_commits[sub]
        abs_sub = os.path.normpath(os.path.join(remote_dir, sub))
        lines.append(
            f'git -C "{abs_sub}" worktree add "$CHESTER_WT_{i}" "{sha}"'
        )

    return lines


def _build_worktree_cleanup_commands(
    submodule_worktrees: Dict[str, str],
    remote_dir: str,
) -> List[str]:
    """Return the cleanup body as bash lines (without the function wrapper).

    Useful for inspection and testing of the cleanup logic in isolation.
    Each line uses '|| true' so cleanup of a non-existent worktree
    (e.g. due to partial creation failure under set -e) does not itself fail.

    Args:
        submodule_worktrees: {submodule_path: abs_remote_worktree_path}
        remote_dir: Absolute remote project root path.

    Returns:
        List of bash lines for the cleanup body.
    """
    lines = []
    for i, (sub, _wt_path) in enumerate(submodule_worktrees.items()):
        abs_sub = os.path.normpath(os.path.join(remote_dir, sub))
        lines.append(
            f'git -C "{abs_sub}" worktree remove --force "$CHESTER_WT_{i}" 2>/dev/null || true'
        )
    return lines
