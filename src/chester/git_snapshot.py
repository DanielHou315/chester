"""Git snapshot utility for chester experiments.

Saves git state (commit hash, branch, dirty flag, diff, submodules, untracked
symlinks) to the experiment log directory so that experiments are reproducible.
"""
import json
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional


def _get_submodule_info(cwd: str) -> List[dict]:
    """Return a list of dicts describing each submodule.

    Each dict has:
      path        - relative path from repo root
      hash        - current checked-out commit hash
      status      - "up_to_date" | "modified" | "uninitialized" | "merge_conflict"
      description - tag / branch description from git submodule status (may be absent)
    """
    result = subprocess.run(
        ["git", "submodule", "status"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    status_map = {
        " ": "up_to_date",
        "+": "modified",
        "-": "uninitialized",
        "U": "merge_conflict",
    }
    submodules = []
    for line in result.stdout.splitlines():
        if not line:
            continue
        status_char = line[0]
        rest = line[1:].strip()
        status = status_map.get(status_char, "unknown")

        # Format after status char: "<hash> <path>" or "<hash> <path> (<description>)"
        parts = rest.split(" ", 1)
        hash_val = parts[0]
        path_and_desc = parts[1] if len(parts) > 1 else ""

        if "(" in path_and_desc:
            path_part, desc_part = path_and_desc.rsplit("(", 1)
            path_val = path_part.strip()
            description = desc_part.rstrip(")").strip()
        else:
            path_val = path_and_desc.strip()
            description = None

        entry: Dict[str, object] = {
            "path": path_val,
            "hash": hash_val,
            "status": status,
        }
        if description:
            entry["description"] = description

        # Detect dirty state inside the submodule working tree (initialized only)
        sub_abs = os.path.join(cwd, path_val)
        if status != "uninitialized" and os.path.isdir(sub_abs):
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=sub_abs,
                capture_output=True,
                text=True,
            )
            entry["dirty"] = bool(status_result.stdout.strip())

            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=sub_abs,
                capture_output=True,
                text=True,
            )
            entry["untracked_files"] = [
                f for f in untracked_result.stdout.splitlines() if f.strip()
            ]
        else:
            entry["dirty"] = False
            entry["untracked_files"] = []

        submodules.append(entry)

    return submodules


def _get_untracked_symlinks(cwd: str) -> List[dict]:
    """Find symlinks in the working tree that are not tracked by git.

    Excludes common build / runtime directories (.git, .venv, .worktrees,
    __pycache__, node_modules, .mypy_cache) from the search.

    Returns a list of dicts:
      path          - symlink path relative to repo root
      target        - value returned by os.readlink (may be relative or absolute)
      target_exists - whether the target path currently exists
    """
    EXCLUDE_DIRS = [
        ".git",
        ".venv",
        ".worktrees",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        # Experiment output directories — symlinks here are internal
        # wandb / chester artifacts, not external dependencies.
        "data",
        "wandb",
        ".wandb",
    ]
    cmd = ["find", ".", "-type", "l"]
    for d in EXCLUDE_DIRS:
        # Match both top-level (./{d}/*) and nested (*/{d}/*) occurrences.
        cmd.extend(["-not", "-path", f"*/{d}/*"])

    find_result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    found_paths = [
        os.path.normpath(p) for p in find_result.stdout.splitlines() if p.strip()
    ]

    if not found_paths:
        return []

    # Determine which symlinks git already tracks (mode 120000 = symlink).
    ls_result = subprocess.run(
        ["git", "ls-files", "-s"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    tracked: set = set()
    for line in ls_result.stdout.splitlines():
        if line.startswith("120000 "):
            # "120000 <hash> <stage>\t<path>"
            tab_idx = line.find("\t")
            if tab_idx != -1:
                tracked.add(os.path.normpath(line[tab_idx + 1 :].strip()))

    untracked = []
    for rel_path in found_paths:
        if rel_path in tracked:
            continue
        abs_path = os.path.join(cwd, rel_path)
        target = os.readlink(abs_path)
        target_exists = os.path.exists(abs_path)
        untracked.append(
            {
                "path": rel_path,
                "target": target,
                "target_exists": target_exists,
            }
        )

    return untracked


def save_git_snapshot(log_dir: str, repo_path: Optional[str] = None) -> dict:
    """Save git state to log_dir for experiment reproducibility.

    Creates:
    - {log_dir}/git_info.json  — commit hash, branch, dirty flag, submodule
                                  status, untracked symlinks, timestamp
    - {log_dir}/git_diff.patch — unified diff of all uncommitted changes
                                  (staged + unstaged), plus untracked file list

    Args:
        log_dir:   Directory to save git info files into (must exist).
        repo_path: Path to the git repository root.  Defaults to cwd.

    Returns:
        Dict with git info, or empty dict if not inside a git repo.
    """
    cwd = repo_path or os.getcwd()

    # System boundary: check we are inside a git repo before doing anything.
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

    # Get commit hash — abort early if there are no commits yet.
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
        return {}

    # Branch name
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip() if result.returncode == 0 else "unknown"

    # Dirty flag (working tree + index, ignoring submodule state)
    result = subprocess.run(
        ["git", "status", "--porcelain", "--ignore-submodules=all"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    dirty = bool(result.stdout.strip())

    # Submodule status
    submodules = _get_submodule_info(cwd)

    # Untracked symlinks (external dataset paths, containers, etc.)
    untracked_symlinks = _get_untracked_symlinks(cwd)

    info: Dict[str, object] = {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        "timestamp": datetime.now().isoformat(),
        "submodules": submodules,
        "untracked_symlinks": untracked_symlinks,
    }

    os.makedirs(log_dir, exist_ok=True)
    info_path = os.path.join(log_dir, "git_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Collect submodule diffs
    submodule_patches = []
    for sub in submodules:
        if not sub.get("dirty"):
            continue
        sub_abs = os.path.join(cwd, sub["path"])
        if not os.path.isdir(sub_abs):
            continue

        # Full diff of tracked changes inside the submodule
        sub_diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=sub_abs,
            capture_output=True,
            text=True,
        )
        sub_diff = sub_diff_result.stdout

        section_lines = [f"# === Submodule: {sub['path']} ===\n"]
        if sub_diff:
            section_lines.append(sub_diff)
        if sub.get("untracked_files"):
            section_lines.append("\n# Untracked files:\n")
            for uf in sub["untracked_files"]:
                section_lines.append(f"# {uf}\n")
        submodule_patches.append("".join(section_lines))

    # Write diff patch if there are uncommitted changes (parent or any submodule)
    any_dirty = dirty or bool(submodule_patches)
    if any_dirty:
        # Staged + unstaged changes for tracked files in the parent
        result = subprocess.run(
            ["git", "diff", "HEAD", "--ignore-submodules=all"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        diff_content = result.stdout

        # Untracked file names in the parent (not their content — could be large)
        result_untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        untracked_files = result_untracked.stdout.strip()

        patch_path = os.path.join(log_dir, "git_diff.patch")
        with open(patch_path, "w") as f:
            if diff_content:
                f.write(diff_content)
            if untracked_files:
                f.write("\n# Untracked files:\n")
                for line in untracked_files.splitlines():
                    f.write(f"# {line}\n")
            for section in submodule_patches:
                f.write("\n")
                f.write(section)

    return info
