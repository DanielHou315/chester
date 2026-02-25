"""
Chester Configuration V2 — New config system loading from .chester/config.yaml.

This module provides the new configuration system for chester.
It searches for .chester/config.yaml first, then falls back to chester.yaml
(with a deprecation warning).

The old config.py module is preserved for backward compatibility.

Usage:
    from chester.config_v2 import load_config, get_backend

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
    3. chester.yaml in search_from or parent directories (deprecated)

    Returns:
        Path to config file if found, None otherwise.
    """
    # Check environment variable first
    env_path = os.environ.get("CHESTER_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        warnings.warn(
            f"CHESTER_CONFIG_PATH set to {env_path} but file not found",
            stacklevel=2,
        )

    # Determine starting directory
    start = Path(search_from).resolve() if search_from else Path.cwd().resolve()

    current = start
    while True:
        # Check .chester/config.yaml first (new location)
        new_config = current / ".chester" / "config.yaml"
        if new_config.exists():
            return new_config

        # Check chester.yaml (deprecated location)
        old_config = current / "chester.yaml"
        if old_config.exists():
            return old_config

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
    Resolve project_path from config or auto-detect from config file location.

    If config_file is .chester/config.yaml, project_path is the parent of .chester/.
    If config_file is chester.yaml at project root, project_path is its directory.
    """
    if "project_path" in config and config["project_path"]:
        # Explicit project_path in config
        return os.path.abspath(config["project_path"])

    # Auto-detect from config file location
    config_dir = config_file.parent
    if config_dir.name == ".chester":
        # .chester/config.yaml -> project is parent of .chester
        return str(config_dir.parent)
    else:
        # chester.yaml at project root
        return str(config_dir)


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
        ValueError: If config contains invalid values (bad backend type, etc.)
        FileNotFoundError: If CHESTER_CONFIG_PATH points to missing file.
    """
    config_file = _find_config_file(search_from)

    if config_file is None:
        # No config file found -- return minimal defaults
        cwd = str(Path(search_from).resolve()) if search_from else os.getcwd()
        config = {
            "project_path": cwd,
            "log_dir": os.path.join(cwd, "data"),
            "package_manager": "python",
            "rsync_include": [],
            "rsync_exclude": [],
            "hydra_config_path": "configs",
            "backends": {"local": BackendConfig(name="local", type="local")},
        }
        return config

    # Issue deprecation warning for old-style chester.yaml at project root
    if config_file.name == "chester.yaml" and config_file.parent.name != ".chester":
        warnings.warn(
            "Loading config from chester.yaml is deprecated. "
            "Move your config to .chester/config.yaml instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # Load YAML
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

    # Parse backends section
    raw_backends = config.get("backends")
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
