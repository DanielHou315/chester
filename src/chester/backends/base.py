"""Backend configuration and base classes."""
from __future__ import annotations

import copy
import os
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
        """Wrap python command based on package manager."""
        pkg = self.project_config.get("package_manager", "python")
        if pkg == "uv":
            return f"uv run {base}"
        else:
            return base

    def wrap_with_singularity(self, commands: List[str]) -> str:
        """Wrap a list of commands with singularity exec if configured."""
        sing = self.config.singularity
        if not sing:
            # No singularity -- join commands with &&
            return " && ".join(commands)

        mount_args = ""
        if sing.mounts:
            mount_args = " ".join(f"-B {m}" for m in sing.mounts)

        gpu_flag = "--nv" if sing.gpu else ""

        inner = " && ".join(commands)
        return (
            f"singularity exec {mount_args} {gpu_flag} "
            f"{sing.image} /bin/bash -c '{inner}'"
        ).strip()
