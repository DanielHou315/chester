"""Integration tests for the new v2 backend system.

These tests exercise the full pipeline: config loading -> backend creation ->
script generation, verifying that all layers work together correctly.
"""
import os

import pytest

from chester.config_v2 import load_config, get_backend
from chester.backends import create_backend


def test_end_to_end_local_command(tmp_path):
    """Full local execution: config + backend + command generation."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  local:\n"
        "    type: local\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("local", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": str(tmp_path / "data" / "exp1"),
            "exp_name": "test_exp",
            "learning_rate": 0.001,
        },
    }

    cmd = backend.generate_command(task, script="train.py")
    assert "python" in cmd
    assert "train.py" in cmd
    assert "--learning_rate" in cmd
    assert "--exp_name" in cmd


def test_end_to_end_local_script(tmp_path):
    """Full local script generation: config + backend + script."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  local:\n"
        "    type: local\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("local", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": str(tmp_path / "data" / "exp1"),
            "exp_name": "test_exp",
        },
    }

    script = backend.generate_script(task, script="train.py")
    assert script.startswith("#!/usr/bin/env bash\n")
    assert "set -e" in script
    assert "python train.py" in script


def test_end_to_end_slurm_script_generation(tmp_path):
    """Full SLURM script gen: config + backend + script."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: uv\n"
        "\n"
        "backends:\n"
        "  gl:\n"
        "    type: slurm\n"
        "    host: gl\n"
        "    remote_dir: /remote/project\n"
        "    modules: [singularity]\n"
        "    cuda_module: cuda/12.8.1\n"
        "    slurm:\n"
        '      partition: spgpu\n'
        '      time: "72:00:00"\n'
        "      nodes: 1\n"
        "      gpus: 1\n"
        "      cpus_per_gpu: 8\n"
        "      mem_per_gpu: 64G\n"
        "    singularity:\n"
        "      image: /opt/ml.sif\n"
        "      mounts: [/usr/share/glvnd]\n"
        "      gpu: true\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("gl", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/exp1",
            "exp_name": "test_slurm",
            "learning_rate": 0.001,
        },
    }

    script = backend.generate_script(task, script="train.py")

    # Verify SBATCH header
    assert "#SBATCH --partition=spgpu" in script
    assert "#SBATCH --time=72:00:00" in script
    assert "#SBATCH --gpus=1" in script
    assert "#SBATCH --cpus-per-gpu=8" in script
    assert "#SBATCH --mem-per-gpu=64G" in script

    # Verify modules
    assert "module load singularity" in script
    assert "module load cuda/12.8.1" in script

    # Verify singularity
    assert "singularity exec" in script
    assert "/opt/ml.sif" in script
    assert "--nv" in script

    # Inside singularity, uv wrapping is skipped (container has its own venv)
    assert "uv run python" not in script
    assert "python train.py" in script

    # Verify .done marker
    assert "touch /remote/project/data/exp1/.done" in script


def test_end_to_end_ssh_script_generation(tmp_path):
    """Full SSH script gen: config + backend + script."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: uv\n"
        "\n"
        "backends:\n"
        "  armdual:\n"
        "    type: ssh\n"
        "    host: armdual\n"
        "    remote_dir: /home/user/project\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("armdual", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": "/home/user/project/data/exp1",
            "exp_name": "test_ssh",
            "batch_size": 32,
        },
    }

    script = backend.generate_script(task, script="train.py")

    assert "#!/usr/bin/env bash" in script
    assert "cd /home/user/project" in script
    assert "uv run python" in script
    assert "train.py" in script
    assert "--batch_size 32" in script
    assert "touch /home/user/project/data/exp1/.done" in script


def test_end_to_end_slurm_with_overrides(tmp_path):
    """SLURM with per-experiment overrides."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  gl:\n"
        "    type: slurm\n"
        "    host: gl\n"
        "    remote_dir: /remote/project\n"
        "    slurm:\n"
        "      partition: spgpu\n"
        '      time: "72:00:00"\n'
        "      gpus: 1\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("gl", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/exp1",
            "exp_name": "override_test",
        },
    }

    # Override time and gpus for this experiment
    script = backend.generate_script(
        task,
        script="train.py",
        slurm_overrides={"time": "6:00:00", "gpus": 4},
    )

    assert "#SBATCH --time=6:00:00" in script  # overridden
    assert "#SBATCH --gpus=4" in script  # overridden
    assert "#SBATCH --partition=spgpu" in script  # unchanged


def test_config_with_prepare_script(tmp_path):
    """Backend with prepare.sh script."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    prepare_dir = chester_dir / "backends" / "local"
    prepare_dir.mkdir(parents=True)
    (prepare_dir / "prepare.sh").write_text("#!/bin/bash\nexport MY_VAR=1\n")

    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  local:\n"
        "    type: local\n"
        "    prepare: .chester/backends/local/prepare.sh\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("local", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": str(tmp_path / "data" / "exp1"),
            "exp_name": "test",
        },
    }

    script = backend.generate_script(task, script="train.py")
    assert "source" in script
    assert "prepare.sh" in script


def test_slurm_with_prepare_script(tmp_path):
    """SLURM backend with prepare.sh resolves path relative to remote_dir."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()

    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  gl:\n"
        "    type: slurm\n"
        "    host: gl\n"
        "    remote_dir: /remote/project\n"
        "    prepare: .chester/backends/gl/prepare.sh\n"
        "    slurm:\n"
        "      partition: gpu\n"
        "      gpus: 1\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("gl", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/exp1",
            "exp_name": "test",
        },
    }

    script = backend.generate_script(task, script="train.py")

    # For remote backends, prepare.sh is resolved relative to remote_dir
    assert "source /remote/project/.chester/backends/gl/prepare.sh" in script


def test_ssh_with_prepare_script(tmp_path):
    """SSH backend with prepare.sh resolves path relative to remote_dir."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()

    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  myserver:\n"
        "    type: ssh\n"
        "    host: myserver\n"
        "    remote_dir: /remote/project\n"
        "    prepare: .chester/backends/ssh/prepare.sh\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend_config = get_backend("myserver", cfg)
    backend = create_backend(backend_config, cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/exp1",
            "exp_name": "test",
        },
    }

    script = backend.generate_script(task, script="train.py")
    assert "source /remote/project/.chester/backends/ssh/prepare.sh" in script


def test_multiple_backends_in_one_config(tmp_path):
    """Config with local, SSH, and SLURM backends all at once."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: uv\n"
        "\n"
        "backends:\n"
        "  local:\n"
        "    type: local\n"
        "  myserver:\n"
        "    type: ssh\n"
        "    host: myserver\n"
        "    remote_dir: /remote/project\n"
        "  cluster:\n"
        "    type: slurm\n"
        "    host: cluster\n"
        "    remote_dir: /cluster/project\n"
        "    slurm:\n"
        "      partition: gpu\n"
        "      gpus: 1\n"
    )

    cfg = load_config(search_from=tmp_path)

    # All three backends should be available
    local_cfg = get_backend("local", cfg)
    ssh_cfg = get_backend("myserver", cfg)
    slurm_cfg = get_backend("cluster", cfg)

    assert local_cfg.type == "local"
    assert ssh_cfg.type == "ssh"
    assert slurm_cfg.type == "slurm"

    # All three should create valid backends
    local = create_backend(local_cfg, cfg)
    ssh = create_backend(ssh_cfg, cfg)
    slurm = create_backend(slurm_cfg, cfg)

    task = {
        "params": {
            "log_dir": "/some/path",
            "exp_name": "test",
        },
    }

    # All three should generate scripts without error
    local_script = local.generate_script(task, script="train.py")
    ssh_script = ssh.generate_script(task, script="train.py")
    slurm_script = slurm.generate_script(task, script="train.py")

    # All use uv run python
    assert "uv run python" in local_script
    assert "uv run python" in ssh_script
    assert "uv run python" in slurm_script


def test_slurm_script_job_output_directives(tmp_path):
    """SLURM script sets per-job stdout/stderr paths and job name."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  gl:\n"
        "    type: slurm\n"
        "    host: gl\n"
        "    remote_dir: /remote/project\n"
        "    slurm:\n"
        "      partition: gpu\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend = create_backend(get_backend("gl", cfg), cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/my_exp",
            "exp_name": "cool_experiment",
        },
    }

    script = backend.generate_script(task, script="train.py")
    assert "#SBATCH -o /remote/project/data/my_exp/slurm.out" in script
    assert "#SBATCH -e /remote/project/data/my_exp/slurm.err" in script
    assert "#SBATCH --job-name=cool_experiment" in script


def test_singularity_wrapping_in_slurm(tmp_path):
    """Singularity wrapping bundles prepare + python + done marker."""
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        "log_dir: data\n"
        "package_manager: python\n"
        "\n"
        "backends:\n"
        "  gl:\n"
        "    type: slurm\n"
        "    host: gl\n"
        "    remote_dir: /remote/project\n"
        "    prepare: setup.sh\n"
        "    slurm:\n"
        "      partition: gpu\n"
        "    singularity:\n"
        "      image: /opt/container.sif\n"
        "      mounts: [/data, /scratch]\n"
        "      gpu: true\n"
    )

    cfg = load_config(search_from=tmp_path)
    backend = create_backend(get_backend("gl", cfg), cfg)

    task = {
        "params": {
            "log_dir": "/remote/project/data/exp1",
            "exp_name": "test",
        },
    }

    script = backend.generate_script(task, script="train.py")

    # Singularity exec line should contain all the important parts
    assert "singularity exec" in script
    assert "-B /data" in script
    assert "-B /scratch" in script
    assert "--nv" in script
    assert "/opt/container.sif" in script
    # Inner commands (prepare, python, done) should be inside singularity
    assert "source /remote/project/setup.sh" in script
    assert "train.py" in script
    assert ".done" in script
