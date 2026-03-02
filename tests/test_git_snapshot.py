# tests/test_git_snapshot.py
import os
import json
import subprocess
import pytest
from chester.git_snapshot import save_git_snapshot


def test_save_git_snapshot_in_repo(tmp_path):
    """Saves git info when in a git repo."""
    # Create a git repo
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))

    assert info["commit"]
    assert info["dirty"] is False
    assert os.path.exists(log_dir / "git_info.json")

    with open(log_dir / "git_info.json") as f:
        saved = json.load(f)
    assert saved["commit"] == info["commit"]


def test_save_git_snapshot_dirty_repo(tmp_path):
    """Saves diff when repo has uncommitted changes."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    # Make dirty
    (tmp_path / "file.txt").write_text("modified")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))

    assert info["dirty"] is True
    assert os.path.exists(log_dir / "git_diff.patch")

    diff = (log_dir / "git_diff.patch").read_text()
    assert "modified" in diff


def test_save_git_snapshot_not_a_repo(tmp_path):
    """Returns empty dict when not in a git repo."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))
    assert info == {}
    assert not os.path.exists(log_dir / "git_info.json")


def test_save_git_snapshot_branch_name(tmp_path):
    """Records the branch name correctly."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))
    # Default branch is typically master or main
    assert info["branch"] in ("master", "main")


def test_save_git_snapshot_creates_log_dir(tmp_path):
    """Creates log_dir if it does not exist."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    log_dir = tmp_path / "nonexistent" / "logs"
    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))

    assert info["commit"]
    assert os.path.exists(log_dir / "git_info.json")


def test_save_git_snapshot_has_timestamp(tmp_path):
    """Info dict contains a timestamp."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "file.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)

    log_dir = tmp_path / "logs"
    log_dir.mkdir()

    info = save_git_snapshot(str(log_dir), repo_path=str(tmp_path))
    assert "timestamp" in info
    # Should be an ISO format string
    assert "T" in info["timestamp"]


def test_submodule_info_dirty_flag(tmp_path):
    """_get_submodule_info sets dirty=True when submodule has uncommitted changes."""
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=parent, check=True, capture_output=True)

    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub, check=True, capture_output=True)
    (sub / "sub_file.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "sub init"], cwd=sub, check=True, capture_output=True)

    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "sub"],
        cwd=parent, check=True, capture_output=True,
    )
    (parent / "root.txt").write_text("root")
    subprocess.run(["git", "add", "."], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "parent init"], cwd=parent, check=True, capture_output=True)

    (parent / "sub" / "sub_file.txt").write_text("modified")

    from chester.git_snapshot import _get_submodule_info
    infos = _get_submodule_info(str(parent))
    assert len(infos) == 1
    assert infos[0]["dirty"] is True
    assert infos[0]["untracked_files"] == []


def test_submodule_info_clean_flag(tmp_path):
    """_get_submodule_info sets dirty=False when submodule is clean."""
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=parent, check=True, capture_output=True)

    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub, check=True, capture_output=True)
    (sub / "sub_file.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "sub init"], cwd=sub, check=True, capture_output=True)

    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "sub"],
        cwd=parent, check=True, capture_output=True,
    )
    (parent / "root.txt").write_text("root")
    subprocess.run(["git", "add", "."], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "parent init"], cwd=parent, check=True, capture_output=True)

    from chester.git_snapshot import _get_submodule_info
    infos = _get_submodule_info(str(parent))
    assert len(infos) == 1
    assert infos[0]["dirty"] is False
    assert infos[0]["untracked_files"] == []


def test_submodule_info_untracked_files(tmp_path):
    """_get_submodule_info records untracked filenames inside dirty submodule."""
    parent = tmp_path / "parent"
    parent.mkdir()
    subprocess.run(["git", "init"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=parent, check=True, capture_output=True)

    sub = tmp_path / "sub"
    sub.mkdir()
    subprocess.run(["git", "init"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub, check=True, capture_output=True)
    (sub / "sub_file.txt").write_text("original")
    subprocess.run(["git", "add", "."], cwd=sub, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "sub init"], cwd=sub, check=True, capture_output=True)

    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", str(sub), "sub"],
        cwd=parent, check=True, capture_output=True,
    )
    (parent / "root.txt").write_text("root")
    subprocess.run(["git", "add", "."], cwd=parent, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "parent init"], cwd=parent, check=True, capture_output=True)

    (parent / "sub" / "new_script.py").write_text("print('hello')")

    from chester.git_snapshot import _get_submodule_info
    infos = _get_submodule_info(str(parent))
    assert len(infos) == 1
    assert infos[0]["dirty"] is True
    assert "new_script.py" in infos[0]["untracked_files"]
