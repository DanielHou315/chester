"""Chester backend system."""
from .base import Backend, BackendConfig, SlurmConfig, SingularityConfig, parse_backend_config

__all__ = [
    "Backend", "BackendConfig", "SlurmConfig", "SingularityConfig",
    "parse_backend_config", "create_backend",
]


def create_backend(config: BackendConfig, project_config: dict) -> Backend:
    """Create a Backend instance from config.

    Args:
        config: Parsed backend configuration.
        project_config: Project-level config dict (package_manager, paths, etc.).

    Returns:
        A concrete Backend instance.

    Raises:
        ValueError: If ``config.type`` is not a recognized backend type.
    """
    if config.type == "local":
        from .local import LocalBackend
        return LocalBackend(config, project_config)
    elif config.type == "ssh":
        from .ssh import SSHBackend
        return SSHBackend(config, project_config)
    elif config.type == "slurm":
        from .slurm import SlurmBackend
        return SlurmBackend(config, project_config)
    raise ValueError(f"Unknown backend type: {config.type}")
