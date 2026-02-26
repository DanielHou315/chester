# tests/test_backend_local.py
import os
import tempfile

import pytest

from chester.backends.base import BackendConfig, SingularityConfig
from chester.backends.local import LocalBackend


def _make_backend(package_manager="python", singularity=None, prepare=None,
                  project_path="/tmp/myproject"):
    config = BackendConfig(
        name="local",
        type="local",
        singularity=singularity,
        prepare=prepare,
    )
    project_config = {
        "project_path": project_path,
        "package_manager": package_manager,
    }
    return LocalBackend(config, project_config)


def test_generate_command_basic():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "batch_size": 32}}
    cmd = backend.generate_command(task, script="train.py")
    assert "python" in cmd
    assert "train.py" in cmd
    assert "--lr" in cmd
    assert "--batch_size" in cmd


def test_generate_command_uv():
    backend = _make_backend(package_manager="uv")
    task = {"params": {"lr": 0.01}}
    cmd = backend.generate_command(task, script="train.py")
    assert cmd.startswith("uv run python")


def test_generate_command_conda():
    backend = _make_backend(package_manager="conda")
    task = {"params": {"lr": 0.01}}
    cmd = backend.generate_command(task, script="train.py")
    # conda doesn't wrap the command, just plain python
    assert cmd.startswith("python")
    assert "uv" not in cmd


def test_generate_command_dict_params():
    """Test that dict params with _name key are handled correctly."""
    backend = _make_backend()
    task = {"params": {"algo": {"_name": "ppo", "clip": 0.2}}}
    cmd = backend.generate_command(task, script="train.py")
    assert "--algo ppo" in cmd or "--algo 'ppo'" in cmd
    assert "--algo_clip" in cmd


def test_generate_command_env_vars():
    backend = _make_backend()
    task = {"params": {"lr": 0.01}}
    cmd = backend.generate_command(task, script="train.py", env={"CUDA_VISIBLE_DEVICES": "0"})
    assert "CUDA_VISIBLE_DEVICES=0" in cmd


def test_local_with_singularity():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        mounts=["/data:/data", "/scratch"],
        gpu=True,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01}}
    script_content = backend.generate_script(task, script="train.py")
    assert "singularity exec" in script_content
    assert "-B /data:/data" in script_content
    assert "--nv" in script_content
    assert "/path/to/container.sif" in script_content


def test_local_with_prepare_script():
    """Test that prepare.sh is sourced in the generated script."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False,
                                      dir='/tmp') as f:
        f.write("#!/bin/bash\nexport FOO=bar\n")
        prepare_path = f.name

    try:
        # Use absolute path for prepare
        backend = _make_backend(prepare=prepare_path)
        task = {"params": {"lr": 0.01}}
        script_content = backend.generate_script(task, script="train.py")
        assert f"source {prepare_path}" in script_content
    finally:
        os.unlink(prepare_path)


def test_generate_script_has_shebang():
    backend = _make_backend()
    task = {"params": {"lr": 0.01}}
    script = backend.generate_script(task, script="train.py")
    assert script.startswith("#!/usr/bin/env bash")


def test_local_script_with_overlay():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        overlay=".containers/overlay.img",
        overlay_size=5120,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01}}
    script = backend.generate_script(task, script="train.py")
    # Overlay creation guard
    assert "if [ ! -f /tmp/myproject/.containers/overlay.img ]" in script
    assert "singularity overlay create --size 5120 /tmp/myproject/.containers/overlay.img" in script
    # Singularity exec includes --overlay
    assert "--overlay /tmp/myproject/.containers/overlay.img" in script


def test_local_command_with_overlay():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        overlay="/data/overlay.img",
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01}}
    cmd = backend.generate_command(task, script="train.py")
    # Overlay creation guard in command string
    assert "if [ ! -f /data/overlay.img ]" in cmd
    assert "--overlay /data/overlay.img" in cmd


def test_submit_dry_returns_none():
    backend = _make_backend()
    task = {"params": {"lr": 0.01}}
    result = backend.submit(task, script_content="echo hello", dry=True)
    assert result is None
