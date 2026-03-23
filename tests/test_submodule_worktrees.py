import pytest
from chester.backends.base import _rewrite_mounts_for_worktrees


REMOTE_DIR = "/home/user/project"


def test_rewrites_subdirectory_mount():
    # source subpath of the submodule — should get wt_path/source
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/home/user/project/IsaacLabTactile/.worktrees/wt0/source:/workspace/IsaacLabTactile/source"]


def test_rewrites_bare_submodule_mount():
    mounts = ["third_party/rl_games:/workspace/third_party/rl_games"]
    worktrees = {"third_party/rl_games": "/home/user/project/third_party/rl_games/.worktrees/wt1"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/home/user/project/third_party/rl_games/.worktrees/wt1:/workspace/third_party/rl_games"]


def test_does_not_rewrite_non_submodule_mount():
    mounts = ["configs:/workspace/configs"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["configs:/workspace/configs"]


def test_no_prefix_collision():
    # IsaacLabTactile_v2 must NOT be rewritten when submodule is IsaacLabTactile
    mounts = ["IsaacLabTactile_v2/source:/workspace/foo"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["IsaacLabTactile_v2/source:/workspace/foo"]


def test_dollar_prefixed_mount_not_rewritten():
    mounts = ["$ISAAC_KIT_DATA:/opt/isaac-sim/kit/data"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["$ISAAC_KIT_DATA:/opt/isaac-sim/kit/data"]


def test_tilde_prefixed_mount_not_rewritten():
    mounts = ["~/.isaac-cache:/opt/isaac-sim/kit/cache"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["~/.isaac-cache:/opt/isaac-sim/kit/cache"]


def test_bare_mount_no_dst():
    mounts = ["/usr/share/glvnd"]
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result == ["/usr/share/glvnd"]


def test_multiple_submodules_rewritten():
    mounts = [
        "IsaacLabTactile/source:/workspace/IsaacLabTactile/source",
        "third_party/rl_games:/workspace/third_party/rl_games",
        "configs:/workspace/configs",
    ]
    worktrees = {
        "IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0",
        "third_party/rl_games": "/home/user/project/third_party/rl_games/.worktrees/wt1",
    }
    result = _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert result[0] == "/home/user/project/IsaacLabTactile/.worktrees/wt0/source:/workspace/IsaacLabTactile/source"
    assert result[1] == "/home/user/project/third_party/rl_games/.worktrees/wt1:/workspace/third_party/rl_games"
    assert result[2] == "configs:/workspace/configs"


def test_does_not_mutate_input():
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    original = list(mounts)
    worktrees = {"IsaacLabTactile": "/home/user/project/IsaacLabTactile/.worktrees/wt0"}
    _rewrite_mounts_for_worktrees(mounts, worktrees, REMOTE_DIR)
    assert mounts == original


def test_empty_worktrees_returns_mounts_unchanged():
    mounts = ["IsaacLabTactile/source:/workspace/IsaacLabTactile/source"]
    result = _rewrite_mounts_for_worktrees(mounts, {}, REMOTE_DIR)
    assert result == mounts
