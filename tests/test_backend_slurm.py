# tests/test_backend_slurm.py
import pytest

from chester.backends.base import BackendConfig, SlurmConfig, SingularityConfig
from chester.backends.slurm import SlurmBackend


def _make_backend(package_manager="python", singularity=None, prepare=None,
                  modules=None, cuda_module=None, host="gl",
                  remote_dir="/home/user/project", slurm=None):
    if slurm is None:
        slurm = SlurmConfig(partition="spgpu", time="72:00:00", gpus=1)
    config = BackendConfig(
        name="gl",
        type="slurm",
        host=host,
        remote_dir=remote_dir,
        singularity=singularity,
        prepare=prepare,
        modules=modules or [],
        cuda_module=cuda_module,
        slurm=slurm,
    )
    project_config = {
        "project_path": "/local/project",
        "package_manager": package_manager,
    }
    return SlurmBackend(config, project_config)


def test_slurm_script_has_sbatch_header():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "#SBATCH --partition=spgpu" in script
    assert "#SBATCH --time=72:00:00" in script
    assert "#SBATCH --gpus=1" in script


def test_slurm_script_with_overrides():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(
        task, script="train.py",
        slurm_overrides={"time": "24:00:00", "gpus": 4},
    )
    # Overridden values
    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --gpus=4" in script
    # Partition is NOT overridden, should keep default
    assert "#SBATCH --partition=spgpu" in script


def test_slurm_script_loads_modules():
    backend = _make_backend(modules=["singularity", "python/3.10"], cuda_module="cuda/12.1.1")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "module load singularity" in script
    assert "module load python/3.10" in script
    assert "module load cuda/12.1.1" in script


def test_slurm_script_creates_done_marker():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "touch /remote/logs/exp1/.done" in script


def test_slurm_script_with_singularity():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        mounts=["/data:/data"],
        gpu=True,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "singularity exec" in script
    assert "-B /data:/data" in script
    assert "--nv" in script
    assert "/path/to/container.sif" in script


def test_slurm_script_has_stdout_stderr_directives():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "#SBATCH -o /remote/logs/exp1/slurm.out" in script
    assert "#SBATCH -e /remote/logs/exp1/slurm.err" in script


def test_slurm_script_has_set_flags():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "set -x" in script
    assert "set -u" in script
    assert "set -e" in script


def test_slurm_script_has_srun_hostname():
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "srun hostname" in script


def test_slurm_script_cds_to_remote_dir():
    backend = _make_backend(remote_dir="/home/user/myproject")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "cd /home/user/myproject" in script


def test_slurm_submit_dry_prints_script(capsys):
    """Dry run prints the script content without executing."""
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    backend.submit(task, script_content="#!/bin/bash\necho hello", dry=True)
    captured = capsys.readouterr()
    assert "#!/bin/bash" in captured.out
    assert "echo hello" in captured.out


def test_slurm_submit_writes_local_script(tmp_path):
    """Submit writes the script to a local file before SCP (even if sbatch fails)."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }
    script_content = "#!/bin/bash\necho hello\n"

    # Mock subprocess.run to avoid real SSH/SCP
    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(returncode=0, stdout="Submitted batch job 12345\n", stderr="")
        backend.submit(task, script_content=script_content, dry=False)

    local_script = tmp_path / "local_logs" / "chester_slurm.sh"
    assert local_script.exists()
    assert local_script.read_text() == script_content


def test_slurm_submit_returns_job_id(tmp_path):
    """submit() parses and returns the SLURM job ID from sbatch output."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }
    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Submitted batch job 98765432\n", stderr=""
        )
        result = backend.submit(task, script_content="#!/bin/bash\necho hello\n", dry=False)
    assert result == 98765432


def test_slurm_submit_returns_none_on_dry_run():
    """Dry run returns None without executing."""
    backend = _make_backend()
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    result = backend.submit(task, script_content="#!/bin/bash\necho hello", dry=True)
    assert result is None


def test_slurm_submit_returns_none_when_job_id_unparseable(tmp_path):
    """submit() returns None when sbatch output cannot be parsed."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }
    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Some unexpected output\n", stderr=""
        )
        result = backend.submit(task, script_content="#!/bin/bash\necho hello\n", dry=False)
    assert result is None


def test_slurm_submit_writes_job_id_to_remote(tmp_path):
    """submit() writes the job ID to .chester_slurm_job_id on the remote host."""
    import unittest.mock as mock
    backend = _make_backend()
    task = {
        "params": {"lr": 0.01, "log_dir": "/remote/logs/exp1", "exp_name": "test"},
        "_local_log_dir": str(tmp_path / "local_logs"),
    }
    with mock.patch("chester.backends.slurm.subprocess.run") as mock_run:
        mock_run.return_value = mock.Mock(
            returncode=0, stdout="Submitted batch job 98765432\n", stderr=""
        )
        backend.submit(task, script_content="#!/bin/bash\necho hello\n", dry=False)

    # The 4th subprocess call (index 3) should write the job ID to remote
    assert mock_run.call_count == 4
    write_call = mock_run.call_args_list[3]
    write_cmd = write_call[0][0]  # positional arg: the command list
    assert "ssh" in write_cmd[0]
    assert "98765432" in write_cmd[2]
    assert ".chester_slurm_job_id" in write_cmd[2]


def test_slurm_script_with_overlay():
    sing = SingularityConfig(
        image="/path/to/container.sif",
        gpu=True,
        overlay="/data/overlay.img",
        overlay_size=8192,
    )
    backend = _make_backend(singularity=sing)
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    # Overlay creation guard before singularity exec
    assert 'if [ ! -f /data/overlay.img ]' in script
    assert "singularity overlay create --size 8192 /data/overlay.img" in script
    # Singularity exec includes --overlay
    assert "--overlay /data/overlay.img" in script


def test_slurm_script_wraps_python_for_uv():
    backend = _make_backend(package_manager="uv")
    task = {"params": {"lr": 0.01, "log_dir": "/remote/logs/exp1"}}
    script = backend.generate_script(task, script="train.py")
    assert "uv run python" in script
