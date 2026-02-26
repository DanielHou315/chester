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


def test_ssh_script_with_overlay():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        overlay="/data/overlay.img",
        overlay_size=5120,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    # Overlay creation guard should appear before singularity exec
    assert 'if [ ! -f /data/overlay.img ]' in script
    assert "singularity overlay create --size 5120 /data/overlay.img" in script
    # Singularity exec should use --overlay
    assert "--overlay /data/overlay.img" in script


def test_ssh_script_with_relative_overlay():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        overlay=".containers/overlay.img",
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    # Relative overlay path resolved against project_path
    assert "--overlay /local/project/.containers/overlay.img" in script
    assert "singularity overlay create --size 10240 /local/project/.containers/overlay.img" in script


def test_ssh_script_with_fakeroot():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        fakeroot=True,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "--fakeroot" in script
    # fakeroot should come before --nv
    fakeroot_pos = script.index("--fakeroot")
    nv_pos = script.index("--nv")
    assert fakeroot_pos < nv_pos


def test_ssh_script_no_overlay_without_config():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "--overlay" not in script
    assert "overlay create" not in script


def test_ssh_script_tilde_mount_not_resolved():
    """Tilde-prefixed mount sources should be left as-is for bash expansion."""
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        mounts=["~/.isaac-cache/kit-data:/opt/isaac-sim/kit/data"],
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "-B ~/.isaac-cache/kit-data:/opt/isaac-sim/kit/data" in script
    # Should NOT be resolved against project_path
    assert "/local/project/~" not in script


def test_ssh_script_env_var_mount_not_resolved():
    """$HOME-prefixed mount sources should be left as-is for bash expansion."""
    sing = SingularityConfig(
        image="/path/to/container.sif",
        mounts=["$HOME/.cache:/opt/cache"],
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "-B $HOME/.cache:/opt/cache" in script
    assert "/local/project/$HOME" not in script


def test_ssh_script_contains_command_with_params():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "batch_size": 64, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "--lr" in script
    assert "--batch_size" in script
    assert "train.py" in script
