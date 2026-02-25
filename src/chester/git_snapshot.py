"""Git snapshot utility for chester experiments.

Saves git state (commit hash, branch, dirty flag, diff) to the experiment
log directory so that experiments are reproducible.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, Optional


def save_git_snapshot(log_dir: str, repo_path: Optional[str] = None) -> dict:
    """Save git commit hash and dirty state to log_dir.

    Creates:
    - {log_dir}/git_info.json with commit hash, branch, dirty flag, timestamp
    - {log_dir}/git_diff.patch if there are uncommitted changes

    Args:
        log_dir: Directory to save git info files into (must exist).
        repo_path: Path to the git repository root. If None, uses the current
            working directory.

    Returns:
        Dict with git info (commit, branch, dirty, timestamp), or empty dict
        if not in a git repo.
    """
    cwd = repo_path or os.getcwd()

    # Check if we are inside a git repo
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}

    # Get commit hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip()
    except subprocess.CalledProcessError:
        # No commits yet (empty repo)
        return {}

    # Get branch name
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        branch = result.stdout.strip()
    except subprocess.CalledProcessError:
        branch = "unknown"

    # Check dirty state (working tree + index)
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    dirty = len(result.stdout.strip()) > 0

    info: Dict[str, object] = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
    }

    # Write git_info.json
    os.makedirs(log_dir, exist_ok=True)
    info_path = os.path.join(log_dir, "git_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Write diff patch if dirty
    if dirty:
        # Capture both staged and unstaged changes
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        diff_content = result.stdout

        # Also capture untracked files list
        result_untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        untracked = result_untracked.stdout.strip()

        patch_path = os.path.join(log_dir, "git_diff.patch")
        with open(patch_path, "w") as f:
            if diff_content:
                f.write(diff_content)
            if untracked:
                f.write(f"\n# Untracked files:\n")
                for line in untracked.splitlines():
                    f.write(f"# {line}\n")

    return info
