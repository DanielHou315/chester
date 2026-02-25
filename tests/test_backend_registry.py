# tests/test_backend_registry.py
import pytest

from chester.backends import create_backend
from chester.backends.base import BackendConfig, SlurmConfig
from chester.backends.local import LocalBackend
from chester.backends.ssh import SSHBackend
from chester.backends.slurm import SlurmBackend


def _project_config():
    return {
        "project_path": "/tmp/project",
        "package_manager": "python",
    }


def test_create_local_backend():
    config = BackendConfig(name="local", type="local")
    backend = create_backend(config, _project_config())
    assert isinstance(backend, LocalBackend)


def test_create_ssh_backend():
    config = BackendConfig(
        name="myserver", type="ssh",
        host="myserver", remote_dir="/home/user/project",
    )
    backend = create_backend(config, _project_config())
    assert isinstance(backend, SSHBackend)


def test_create_slurm_backend():
    config = BackendConfig(
        name="gl", type="slurm",
        host="gl", remote_dir="/home/user/project",
        slurm=SlurmConfig(partition="spgpu", time="72:00:00", gpus=1),
    )
    backend = create_backend(config, _project_config())
    assert isinstance(backend, SlurmBackend)


def test_create_unknown_backend_raises():
    config = BackendConfig(name="bad", type="local")
    # Monkey-patch type to something invalid
    config.type = "ec2"
    with pytest.raises(ValueError, match="Unknown backend type"):
        create_backend(config, _project_config())
