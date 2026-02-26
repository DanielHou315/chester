"""SLURM time estimation utilities.

Provides helpers for computing SLURM wall-clock time limits from empirical
measurements using a simple affine (y = mx + b) model.
"""
from __future__ import annotations

import math


def parse_slurm_time(time_str: str) -> float:
    """Parse a SLURM time string into total seconds.

    Accepted formats (matching SLURM conventions with hours priority
    when fewer than two colons are present):

    * ``D-HH:MM:SS``  — days, hours, minutes, seconds
    * ``D-HH:MM``     — days, hours, minutes
    * ``D-HH``        — days, hours
    * ``HH:MM:SS``    — hours, minutes, seconds
    * ``HH:MM``       — hours, minutes
    * ``HH``          — hours (single bare number = hours)
    * ``0``           — zero

    Returns:
        Total seconds as a float.
    """
    s = str(time_str).strip()
    if not s:
        return 0.0

    days = 0
    if "-" in s:
        day_part, s = s.split("-", 1)
        days = int(day_part)

    parts = s.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
    elif len(parts) == 2:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), 0.0
    elif len(parts) == 1:
        hours, minutes, seconds = int(parts[0]), 0, 0.0
    else:
        raise ValueError(f"Cannot parse SLURM time string: {time_str!r}")

    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def format_slurm_time(total_seconds: float) -> str:
    """Format total seconds into a SLURM time string.

    Output format: ``D-HH:MM:SS`` if days > 0, else ``HH:MM:SS``.
    Seconds are always rounded up to the nearest whole second.

    Returns:
        Formatted SLURM time string.
    """
    total_seconds = math.ceil(max(0.0, total_seconds))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if days > 0:
        return f"{days}-{hms}"
    return hms


def estimate_slurm_time(
    x: float,
    y: str,
    x_target: float,
    b: str = "0",
) -> str:
    """Estimate SLURM wall-clock time using an affine model.

    Given an empirical observation that *x* units of work took time *y*,
    and a fixed offset *b*, computes the slope ``m = (y - b) / x`` and
    returns the predicted time for *x_target* units:
    ``time = m * x_target + b``.

    Args:
        x: Observed units of work (e.g. epochs completed).
        y: Observed wall-clock time as a SLURM time string.
        x_target: Target units of work to estimate time for.
        b: Offset (intercept) as a SLURM time string.  Defaults to ``"0"``.

    Returns:
        Estimated SLURM time string.

    Example::

        >>> estimate_slurm_time(x=100, y="0-2:00:00", x_target=200, b="0")
        '0-04:00:00'
        >>> estimate_slurm_time(x=100, y="2:00:00", x_target=50)
        '01:00:00'
    """
    if x <= 0:
        raise ValueError(f"x must be positive, got {x}")

    y_sec = parse_slurm_time(y)
    b_sec = parse_slurm_time(b)
    m = (y_sec - b_sec) / x
    target_sec = m * x_target + b_sec
    return format_slurm_time(target_sec)
