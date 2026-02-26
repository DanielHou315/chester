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


@dataclass
class SlurmConfig:
    """SLURM job parameters with per-experiment override support."""
    partition: Optional[str] = None
    time: Optional[str] = None
    nodes: int = 1
    gpus: Optional[int] = None
    cpus_per_gpu: Optional[int] = None
    mem_per_gpu: Optional[str] = None
    ntasks_per_node: Optional[int] = None
    email_begin: Optional[str] = None  # email address for BEGIN notifications
    email_end: Optional[str] = None    # email address for END/FAIL notifications
    extra_directives: List[str] = field(default_factory=list)

    def with_overrides(self, overrides: Dict[str, Any]) -> SlurmConfig:
        """Return a new SlurmConfig with specified fields overridden."""
        new = copy.deepcopy(self)
        valid_fields = {f.name for f in fields(self)}
        for key, value in overrides.items():
            if key not in valid_fields:
                raise ValueError(
                    f"Unknown SLURM override '{key}'. "
                    f"Valid fields: {sorted(valid_fields)}"
                )
            setattr(new, key, value)
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
        for directive in self.extra_directives:
            d = directive if directive.startswith("--") else f"--{directive}"
            lines.append(f"#SBATCH {d}")
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
        )

    # Parse SLURM config
    slurm_raw = raw.get("slurm")
    slurm = None
    if slurm_raw:
        slurm = SlurmConfig(
            partition=slurm_raw.get("partition"),
            time=slurm_raw.get("time"),
            nodes=slurm_raw.get("nodes", 1),
            gpus=slurm_raw.get("gpus"),
            cpus_per_gpu=slurm_raw.get("cpus_per_gpu"),
            mem_per_gpu=slurm_raw.get("mem_per_gpu"),
            ntasks_per_node=slurm_raw.get("ntasks_per_node"),
            email_begin=slurm_raw.get("email_begin"),
            email_end=slurm_raw.get("email_end"),
            extra_directives=slurm_raw.get("extra_directives", []),
        )

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
            args = build_hydra_args(params, hydra_flags)
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

    def wrap_with_singularity(self, commands: List[str]) -> str:
        """Wrap a list of commands with singularity exec if configured."""
        sing = self.config.singularity
        if not sing:
            return " && ".join(commands)

        project_path = self.project_config.get("project_path", "")

        parts = ["singularity", "exec"]
        for m in sing.mounts:
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

        inner = " && ".join(commands)
        parts.extend(["/bin/bash", "-c", shlex.quote(inner)])
        return " ".join(parts)
