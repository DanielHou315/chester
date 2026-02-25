# tests/test_backend_ssh.py
import pytest

from chester.backends.base import BackendConfig, SingularityConfig
from chester.backends.ssh import SSHBackend


def _make_backend(package_manager="python", singularity=None, prepare=None,
                  modules=None, host="myserver", remote_dir="/home/user/project"):
    config = BackendConfig(
        name="myserver",
        type="ssh",
        host=host,
        remote_dir=remote_dir,
        singularity=singularity,
        prepare=prepare,
        modules=modules or [],
    )
    project_config = {
        "project_path": "/local/project",
        "package_manager": package_manager,
    }
    return SSHBackend(config, project_config)


def test_ssh_script_has_bash_header():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "#!/usr/bin/env bash" in script
    assert "set -x" in script
    assert "set -u" in script
    assert "set -e" in script


def test_ssh_script_cds_to_remote_dir():
    backend = _make_backend(remote_dir="/home/user/myproject")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "cd /home/user/myproject" in script


def test_ssh_script_creates_done_marker():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "touch /remote/logs/exp1/.done" in script


def test_ssh_script_with_singularity():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        mounts=["/data:/data", "/scratch"],
        gpu=True,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "singularity exec" in script
    assert "-B /data:/data" in script
    assert "-B /scratch" in script
    assert "--nv" in script
    assert "/path/to/container.sif" in script


def test_ssh_script_wraps_python_for_uv():
    backend = _make_backend(package_manager="uv")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "uv run python" in script


def test_ssh_script_with_prepare():
    backend = _make_backend(prepare=".chester/backends/myserver/prepare.sh")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    # Prepare path should be relative to remote_dir
    assert "source /home/user/project/.chester/backends/myserver/prepare.sh" in script


def test_ssh_script_contains_command_with_params():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "batch_size": 64, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "--lr" in script
    assert "--batch_size" in script
    assert "train.py" in script
