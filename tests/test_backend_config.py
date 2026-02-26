# tests/test_backend_config.py
import pytest
from chester.backends.base import (
    BackendConfig,
    SlurmConfig,
    SingularityConfig,
    parse_backend_config,
)


def test_parse_local_backend():
    raw = {"type": "local"}
    cfg = parse_backend_config("local", raw)
    assert cfg.name == "local"
    assert cfg.type == "local"
    assert cfg.host is None
    assert cfg.remote_dir is None
    assert cfg.singularity is None


def test_parse_ssh_backend():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "prepare": ".chester/backends/myserver/prepare.sh",
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.type == "ssh"
    assert cfg.host == "myserver"
    assert cfg.remote_dir == "/home/user/project"
    assert cfg.prepare == ".chester/backends/myserver/prepare.sh"


def test_parse_slurm_backend():
    raw = {
        "type": "slurm",
        "host": "gl",
        "remote_dir": "/home/user/project",
        "modules": ["singularity"],
        "cuda_module": "cuda/12.1.1",
        "slurm": {
            "partition": "spgpu",
            "time": "72:00:00",
            "gpus": 1,
            "cpus_per_gpu": 4,
            "mem_per_gpu": "80G",
        },
    }
    cfg = parse_backend_config("gl", raw)
    assert cfg.type == "slurm"
    assert cfg.slurm is not None
    assert cfg.slurm.partition == "spgpu"
    assert cfg.slurm.time == "72:00:00"
    assert cfg.slurm.gpus == 1


def test_parse_singularity_config():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "singularity": {
            "image": "/path/to/container.sif",
            "mounts": ["/data:/data", "/scratch"],
            "gpu": True,
        },
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.singularity is not None
    assert cfg.singularity.image == "/path/to/container.sif"
    assert cfg.singularity.gpu is True
    assert len(cfg.singularity.mounts) == 2


def test_slurm_override():
    slurm = SlurmConfig(partition="spgpu", time="72:00:00", gpus=1)
    overridden = slurm.with_overrides({"time": "24:00:00", "gpus": 4})
    assert overridden.time == "24:00:00"
    assert overridden.gpus == 4
    assert overridden.partition == "spgpu"  # unchanged


def test_invalid_backend_type():
    with pytest.raises(ValueError, match="Unknown backend type"):
        parse_backend_config("bad", {"type": "ec2"})


def test_slurm_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_backend_config("gl", {"type": "slurm"})


def test_ssh_missing_host():
    with pytest.raises(ValueError, match="host"):
        parse_backend_config("srv", {"type": "ssh"})


def test_parse_singularity_overlay():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "singularity": {
            "image": "/path/to/container.sif",
            "overlay": ".containers/overlay.img",
            "overlay_size": 5120,
        },
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.singularity.overlay == ".containers/overlay.img"
    assert cfg.singularity.overlay_size == 5120


def test_parse_singularity_overlay_defaults():
    raw = {
        "type": "ssh",
        "host": "myserver",
        "remote_dir": "/home/user/project",
        "singularity": {
            "image": "/path/to/container.sif",
        },
    }
    cfg = parse_backend_config("myserver", raw)
    assert cfg.singularity.overlay is None
    assert cfg.singularity.overlay_size == 10240


def test_slurm_generates_header():
    slurm = SlurmConfig(
        partition="spgpu",
        time="72:00:00",
        nodes=1,
        gpus=2,
        cpus_per_gpu=4,
        mem_per_gpu="80G",
        extra_directives=["--gpu_cmode=shared"],
    )
    header = slurm.to_sbatch_header()
    assert "#!/usr/bin/env bash" in header
    assert "#SBATCH --partition=spgpu" in header
    assert "#SBATCH --time=72:00:00" in header
    assert "#SBATCH --gpus=2" in header
    assert "#SBATCH --cpus-per-gpu=4" in header
    assert "#SBATCH --mem-per-gpu=80G" in header
    assert "#SBATCH --gpu_cmode=shared" in header
    assert "#SBATCH --nodes=1" in header
