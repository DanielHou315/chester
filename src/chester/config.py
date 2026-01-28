"""
Chester Configuration Module

This module provides configuration for chester via YAML files.
Configuration is loaded from chester.yaml in the parent project directory.

Search order for chester.yaml:
1. CHESTER_CONFIG_PATH environment variable
2. Parent directories from current working directory (up to git root)
3. Fall back to defaults for local-only usage

Usage:
    from chester import config
    print(config.PROJECT_PATH)
    print(config.REMOTE_DIR['gl'])
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Lazy-loaded configuration
_config: Optional[Dict[str, Any]] = None
_config_path: Optional[Path] = None


def _find_config_file() -> Optional[Path]:
    """
    Search for chester.yaml configuration file.

    Search order:
    1. CHESTER_CONFIG_PATH environment variable
    2. Parent directories from cwd (stops at git root or filesystem root)

    Returns:
        Path to chester.yaml if found, None otherwise
    """
    # Check environment variable first
    env_path = os.environ.get('CHESTER_CONFIG_PATH')
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        warnings.warn(f"CHESTER_CONFIG_PATH set to {env_path} but file not found")

    # Search parent directories from cwd
    current = Path.cwd().resolve()
    while current != current.parent:
        config_path = current / 'chester.yaml'
        if config_path.exists():
            return config_path
        # Stop at git root to avoid searching too far
        if (current / '.git').exists():
            break
        current = current.parent

    return None


def _get_defaults() -> Dict[str, Any]:
    """
    Get default configuration for local-only usage.

    These defaults allow chester to work without a config file
    for basic local execution.
    """
    cwd = os.getcwd()
    return {
        'project_path': cwd,
        'log_dir': os.path.join(cwd, 'data'),
        'host_address': {'local': ''},
        'ssh_hosts': [],
        'remote_dir': {'local': cwd},
        'remote_log_dir': {'local': os.path.join(cwd, 'data')},
        'remote_header': {},
        'simg_path': {},
        'cuda_module': {},
        'modules': {},
        'remote_mount_option': {},
        'autobot_nodelist': [],
        'gpu_state_dir': '',
        'chester_queue_dir': '',
        'chester_scheduler_log_dir': '',
        # Scheduler username (for autobot scheduler, defaults to current user)
        'scheduler_username': os.environ.get('USER', os.environ.get('USERNAME', '')),
        # Package manager config
        'package_manager': 'uv',  # 'uv' or 'conda'
        'conda_env': None,  # Required if package_manager is 'conda'
        'conda_command': 'conda',  # 'conda' or 'mamba'
        'sync_on_launch': True,  # Whether to run uv sync / conda env update
        # Rsync patterns (optional, falls back to file-based approach)
        'rsync_include': [],
        'rsync_exclude': [],
        # Custom commands to run after package manager setup (replaces prepare.sh)
        'prepare_commands': [],
        # Hydra config path (relative to project_path, or absolute)
        'hydra_config_path': 'configs',
    }


def _resolve_paths(config: Dict[str, Any], config_dir: Path) -> Dict[str, Any]:
    """
    Resolve relative paths in configuration.

    - project_path: defaults to config file directory if not specified
    - log_dir: resolved relative to project_path if not absolute
    """
    # Resolve project_path
    if 'project_path' not in config or not config['project_path']:
        config['project_path'] = str(config_dir)
    else:
        config['project_path'] = os.path.abspath(config['project_path'])

    project_path = config['project_path']

    # Resolve log_dir relative to project_path
    log_dir = config.get('log_dir', 'data')
    if not os.path.isabs(log_dir):
        config['log_dir'] = os.path.join(project_path, log_dir)
    else:
        config['log_dir'] = log_dir

    # Resolve hydra_config_path relative to project_path
    hydra_config_path = config.get('hydra_config_path', 'configs')
    if not os.path.isabs(hydra_config_path):
        config['hydra_config_path'] = os.path.join(project_path, hydra_config_path)
    else:
        config['hydra_config_path'] = hydra_config_path

    return config


def _merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge loaded config with defaults for missing keys.
    """
    defaults = _get_defaults()
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value
        elif isinstance(default_value, dict) and isinstance(config.get(key), dict):
            # Merge dict values (e.g., host_address)
            merged = default_value.copy()
            merged.update(config[key])
            config[key] = merged
    return config


def _load_config() -> Dict[str, Any]:
    """
    Load configuration from chester.yaml with lazy initialization.

    Returns:
        Configuration dictionary
    """
    global _config, _config_path

    if _config is not None:
        return _config

    _config_path = _find_config_file()

    if _config_path is not None:
        try:
            import yaml
            with open(_config_path) as f:
                _config = yaml.safe_load(f) or {}
            _config = _resolve_paths(_config, _config_path.parent)
            _config = _merge_with_defaults(_config)
        except ImportError:
            warnings.warn(
                "PyYAML not installed. Install with: pip install pyyaml\n"
                "Falling back to default configuration."
            )
            _config = _get_defaults()
        except Exception as e:
            warnings.warn(f"Error loading {_config_path}: {e}\nFalling back to defaults.")
            _config = _get_defaults()
    else:
        # No config file found - use defaults
        _config = _get_defaults()

    return _config


def get_config_path() -> Optional[Path]:
    """Get the path to the loaded configuration file, if any."""
    _load_config()  # Ensure config is loaded
    return _config_path


def reload_config() -> Dict[str, Any]:
    """
    Force reload of configuration.

    Useful for testing or when config file has changed.
    """
    global _config, _config_path
    _config = None
    _config_path = None
    return _load_config()


# Mapping from UPPER_CASE attribute names to yaml keys
_ATTR_MAPPING = {
    'PROJECT_PATH': 'project_path',
    'LOG_DIR': 'log_dir',
    'HOST_ADDRESS': 'host_address',
    'SSH_HOSTS': 'ssh_hosts',
    'REMOTE_DIR': 'remote_dir',
    'REMOTE_LOG_DIR': 'remote_log_dir',
    'REMOTE_HEADER': 'remote_header',
    'SIMG_PATH': 'simg_path',
    'CUDA_MODULE': 'cuda_module',
    'MODULES': 'modules',
    'REMOTE_MOUNT_OPTION': 'remote_mount_option',
    'AUTOBOT_NODELIST': 'autobot_nodelist',
    'GPU_STATE_DIR': 'gpu_state_dir',
    'CHESTER_QUEUE_DIR': 'chester_queue_dir',
    'CHESTER_CHEDULER_LOG_DIR': 'chester_scheduler_log_dir',  # Note: typo preserved for compatibility
    'CHESTER_SCHEDULER_LOG_DIR': 'chester_scheduler_log_dir',  # Correct spelling
    'SCHEDULER_USERNAME': 'scheduler_username',
    # Package manager config
    'PACKAGE_MANAGER': 'package_manager',
    'CONDA_ENV': 'conda_env',
    'CONDA_COMMAND': 'conda_command',
    'SYNC_ON_LAUNCH': 'sync_on_launch',
    # Rsync patterns
    'RSYNC_INCLUDE': 'rsync_include',
    'RSYNC_EXCLUDE': 'rsync_exclude',
    # Custom setup commands
    'PREPARE_COMMANDS': 'prepare_commands',
    # Hydra config path
    'HYDRA_CONFIG_PATH': 'hydra_config_path',
}


def __getattr__(name: str) -> Any:
    """
    Enable config.VARIABLE_NAME access pattern.

    This allows backward-compatible access like:
        config.PROJECT_PATH
        config.REMOTE_DIR['gl']
    """
    if name in _ATTR_MAPPING:
        cfg = _load_config()
        yaml_key = _ATTR_MAPPING[name]
        value = cfg.get(yaml_key)

        # Handle empty string defaults for path configs
        if value is None:
            if yaml_key in ('project_path', 'log_dir', 'gpu_state_dir',
                           'chester_queue_dir', 'chester_scheduler_log_dir'):
                return ''
            return {} if yaml_key.endswith('_dir') or yaml_key in (
                'host_address', 'remote_dir', 'remote_log_dir', 'remote_header',
                'simg_path', 'cuda_module', 'modules', 'remote_mount_option'
            ) else []

        return value

    # Check if it's a module-level function
    if name in ('get_config_path', 'reload_config', '_load_config'):
        return globals()[name]

    raise AttributeError(f"module 'chester.config' has no attribute '{name}'")


def __dir__() -> List[str]:
    """List available attributes for tab completion."""
    return list(_ATTR_MAPPING.keys()) + ['get_config_path', 'reload_config']


# For debugging: print config when run directly
if __name__ == "__main__":
    cfg = _load_config()
    print(f"Config file: {_config_path}")
    print(f"PROJECT_PATH: {cfg.get('project_path')}")
    print(f"LOG_DIR: {cfg.get('log_dir')}")
    print(f"HOST_ADDRESS: {cfg.get('host_address')}")
    print(f"REMOTE_DIR: {cfg.get('remote_dir')}")
