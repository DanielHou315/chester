"""
Utilities for integrating Chester with Hydra.

Note: hydra-core is an optional dependency. Functions that require hydra
will raise ImportError if hydra is not installed.
"""

import os
import os.path as osp
from typing import Dict, Any, List, Union, Callable
from pathlib import Path
import base64
import pickle

# Lazy imports for optional hydra dependency
_hydra = None
_HydraConfig = None
_read_write = None
_open_dict = None


def _require_hydra():
    """Import hydra and related modules. Raises ImportError if not available."""
    global _hydra, _HydraConfig, _read_write, _open_dict
    if _hydra is None:
        try:
            import hydra
            from hydra.core.hydra_config import HydraConfig
            from omegaconf import read_write, open_dict
            _hydra = hydra
            _HydraConfig = HydraConfig
            _read_write = read_write
            _open_dict = open_dict
        except ImportError:
            raise ImportError(
                "hydra-core is required for this function. "
                "Install it with: pip install hydra-core"
            )
    return _hydra, _HydraConfig, _read_write, _open_dict

def _format_hydra_value(value: Any) -> str:
    """Format a value for Hydra command line override."""
    if isinstance(value, bool):
        return str(value).lower()
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # If it contains spaces, quote it
        if ' ' in value:
            return f'"{value}"'
        return value
    elif isinstance(value, list):
        # Format list values with Hydra's format [val1,val2,...]
        formatted_items = []
        for item in value:
            if isinstance(item, str) and ',' in item:
                # Escape commas in string values
                formatted_items.append(f'"{item}"')
            else:
                formatted_items.append(_format_hydra_value(item))
        return f"[{','.join(formatted_items)}]"
    return str(value)


def variant_to_hydra_overrides(variant: Dict[str, Any]) -> List[str]:
    """
    Convert a chester variant dictionary to Hydra command line overrides.
    
    Args:
        variant: Dictionary of parameters from chester's VariantGenerator
        
    Returns:
        List of strings, each being a Hydra command line override (e.g., ['key1=val1', 'key2=val2', ...])
    """
    overrides = []
    
    for key, value in variant.items():
        # Skip special chester keys
        if key in ['chester_first_variant', 'chester_last_variant', 'is_debug']:
            continue
        
        # Handle nested dictionaries
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if str(nested_key) == "_name":
                    # Special case for _name which becomes just the parent key in Hydra
                    overrides.append(f"{key}={_format_hydra_value(nested_value)}")
                else:
                    overrides.append(f"{key}.{nested_key}={_format_hydra_value(nested_value)}")
        else:
            overrides.append(f"{key}={_format_hydra_value(value)}")
    
    return overrides


def to_hydra_command(
    params: Dict[str, Any],
    python_command: str = "python",
    script: str = "main_df.py",
    hydra_flags: Dict[str, Any] = None,
    env: Dict[str, Any] = {},
) -> str:
    """
    Convert parameters to a Hydra command.
    
    Args:
        params: Dictionary of parameters
        python_command: Python command to use
        script: Script to run
        hydra_flags: Additional Hydra flags (e.g., {'multirun': True})
        
    Returns:
        Command string to execute
    """
    variant_data = pickle.loads(base64.b64decode(params.pop("variant_data")))
    variant_data["hydra.run.dir"] = params["log_dir"]
    # Extract variant_data to convert to overrides
    command = python_command + " " + script
    for k, v in env.items():
        command = ("%s=%s " % (k, v)) + command
    # Add Hydra flags
    if hydra_flags:
        for flag, value in hydra_flags.items():
            if value is True:
                command += f" --{flag}"
            elif value not in [False, None]:
                command += f" --{flag}={value}"
    
    # Add Hydra overrides
    overrides = variant_to_hydra_overrides(variant_data)
    if overrides:
        command += " " + " ".join(overrides)
    
    return command 

def run_hydra_command(command: str, log_dir: str, stub_method_call: Callable):
    from chester import config
    hydra, HydraConfig, read_write, open_dict = _require_hydra()

    cmd_parts = command.split()
    # Find the index where the python module starts (after python command and potential -m flag)
    module_start_idx = 0
    for i, part in enumerate(cmd_parts):
        if part.endswith('.py') or (part == '-m' and i+1 < len(cmd_parts)):
            module_start_idx = i + 2 if part == '-m' else i + 1
            break

    # Everything after the python module are hydra overrides
    overrides = cmd_parts[module_start_idx:]
    # Use hydra_config_path from chester.yaml (already resolved to absolute path)
    config_dir = config.HYDRA_CONFIG_PATH
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="config",
                            overrides=overrides,
                            return_hydra_config=True)
        # output_dir = cfg.hydra.run.dir
        output_dir = log_dir
        with read_write(cfg.hydra.runtime):
            with open_dict(cfg.hydra.runtime):
                cfg.hydra.runtime.output_dir = os.path.abspath(output_dir)
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)
        HydraConfig.instance().set_config(cfg)
        stub_method_call(cfg)