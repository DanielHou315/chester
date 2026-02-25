"""Shared utility functions for chester."""
import shlex
from typing import Any


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
