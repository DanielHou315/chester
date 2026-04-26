"""
Chester configuration — loads .chester/config.yaml.

Usage:
    from chester.config import load_config, get_backend

    cfg = load_config()
    backend = get_backend("greatlakes", cfg)
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from chester.backends.base import BackendConfig, parse_backend_config


VALID_PACKAGE_MANAGERS = ("python", "conda", "uv")


def _find_config_file(search_from: Optional[Path] = None) -> Optional[Path]:
    """
    Search for chester configuration file.

    Search order:
    1. CHESTER_CONFIG_PATH environment variable
    2. .chester/config.yaml in search_from or parent directories

    Returns:
        Path to config file if found, None otherwise.
    """
    env_path = os.environ.get("CHESTER_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        warnings.warn(
            f"CHESTER_CONFIG_PATH set to {env_path} but file not found",
            stacklevel=2,
        )

    start = Path(search_from).resolve() if search_from else Path.cwd().resolve()

    current = start
    while True:
        new_config = current / ".chester" / "config.yaml"
        if new_config.exists():
            return new_config

        # Stop at git root to avoid searching too far
        if (current / ".git").exists():
            break

        parent = current.parent
        if parent == current:
            break
        current = parent

    return None


def _resolve_project_path(config: Dict[str, Any], config_file: Path) -> str:
    """
    Resolve project_path from config or auto-detect from .chester/ parent.
    """
    if "project_path" in config and config["project_path"]:
        return os.path.abspath(config["project_path"])

    return str(config_file.parent.parent)


def _parse_backends(
    raw_backends: Optional[Dict[str, Any]],
) -> Dict[str, BackendConfig]:
    """Parse backends section into BackendConfig objects."""
    if not raw_backends:
        return {}

    backends = {}
    for name, raw in raw_backends.items():
        backends[name] = parse_backend_config(name, raw)
    return backends


def load_config(search_from: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load chester configuration.

    Searches for config files and returns a fully resolved configuration dict.
    The 'backends' key contains BackendConfig objects (not raw dicts).

    Args:
        search_from: Directory to start searching from. Defaults to cwd.

    Returns:
        Configuration dictionary with resolved paths and parsed backends.

    Raises:
        FileNotFoundError: If no .chester/config.yaml is found.
        ValueError: If config contains invalid values (bad backend type, etc.).
    """
    config_file = _find_config_file(search_from)

    if config_file is None:
        raise FileNotFoundError(
            "Chester requires .chester/config.yaml in your project. "
            "See docs/legacy/migration-v1-to-v2.md for migration from "
            "chester 1.x (chester.yaml at project root)."
        )

    with open(config_file) as f:
        config = yaml.safe_load(f) or {}

    # Resolve project_path
    project_path = _resolve_project_path(config, config_file)
    config["project_path"] = project_path

    # Resolve log_dir relative to project_path
    log_dir = config.get("log_dir", "data")
    if not os.path.isabs(log_dir):
        config["log_dir"] = os.path.join(project_path, log_dir)
    else:
        config["log_dir"] = log_dir

    # Resolve hydra_config_path relative to project_path
    hydra_config_path = config.get("hydra_config_path", "configs")
    if not os.path.isabs(hydra_config_path):
        config["hydra_config_path"] = os.path.join(project_path, hydra_config_path)
    else:
        config["hydra_config_path"] = hydra_config_path

    # Validate package_manager
    pkg_manager = config.get("package_manager", "python")
    if pkg_manager not in VALID_PACKAGE_MANAGERS:
        raise ValueError(
            f"Invalid package_manager '{pkg_manager}'. "
            f"Must be one of: {VALID_PACKAGE_MANAGERS}"
        )
    config["package_manager"] = pkg_manager

    # Set defaults for optional fields
    config.setdefault("rsync_include", [])
    config.setdefault("rsync_exclude", [])

    # Parse shared singularity defaults (if any)
    shared_singularity = config.get("singularity")

    # Parse backends section, merging shared singularity into each backend
    raw_backends = config.get("backends")
    if shared_singularity and raw_backends:
        for _name, raw in raw_backends.items():
            if "singularity" not in raw:
                # Backend has no singularity section — inherit shared
                raw["singularity"] = dict(shared_singularity)
            else:
                # Backend has its own — merge shared as defaults
                merged = dict(shared_singularity)
                merged.update(raw["singularity"])
                raw["singularity"] = merged

    backends = _parse_backends(raw_backends)

    # Create a default local backend if none defined
    if not backends:
        backends["local"] = BackendConfig(name="local", type="local")

    config["backends"] = backends

    return config


def get_backend(name: str, config: Dict[str, Any]) -> BackendConfig:
    """
    Get a BackendConfig by name from a loaded config.

    Args:
        name: Backend name (must match a key in config['backends']).
        config: Configuration dict returned by load_config().

    Returns:
        BackendConfig for the named backend.

    Raises:
        KeyError: If no backend with the given name exists.
    """
    backends = config.get("backends", {})
    if name not in backends:
        available = sorted(backends.keys())
        raise KeyError(
            f"Backend '{name}' not found. "
            f"Available backends: {available}"
        )
    return backends[name]
