"""Shared utility functions for chester."""
import shlex
from typing import Any, Dict


def shellquote(s: str) -> str:
    """Shell-escape a string. Uses shlex.quote from stdlib."""
    if not s:
        return "''"
    return shlex.quote(s)


def to_param_val(v: Any) -> str:
    """Convert a parameter value to its shell-safe string representation."""
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(shellquote(str(item)) for item in v)
    else:
        return shellquote(str(v))


def build_cli_args(params: Dict[str, Any]) -> str:
    """Convert a dict of params to CLI args string.

    Handles nested dicts with ``_name`` keys (e.g. algorithm configs).
    Skips ``pre_commands`` and ``post_commands`` keys.

    Returns:
        A string like ``--lr 0.01  --batch_size 32``.
    """
    parts = []
    for k, v in params.items():
        if k in ("pre_commands", "post_commands"):
            continue
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    parts.append(f"--{k} {to_param_val(nv)}")
                else:
                    parts.append(f"--{k}_{nk} {to_param_val(nv)}")
        else:
            parts.append(f"--{k} {to_param_val(v)}")
    return "  ".join(parts)
