from chester.backends.base import (
    _rewrite_mounts_for_worktrees,
    _build_worktree_setup_commands,
    _build_worktree_cleanup_commands,
)


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


def test_setup_commands_variable_assignments():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "abc" * 13 + "abcd"}  # 40 chars
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "CHESTER_WT_0=/remote/IsaacLabTactile/.worktrees/wt0" in combined


def test_setup_commands_trap():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": "a" * 40}
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "trap '_chester_wt_cleanup' EXIT" in combined
    assert "trap 'trap - EXIT; _chester_wt_cleanup; exit 130' INT" in combined
    assert "trap 'trap - EXIT; _chester_wt_cleanup; exit 143' TERM" in combined


def test_setup_commands_git_worktree_add():
    sha = "abc1234def5678abc1234def5678abc1234def56"
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    commits = {"IsaacLabTactile": sha}
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "git -C /remote/IsaacLabTactile worktree add" in combined
    assert sha in combined
    # The worktree add must use the shell variable reference, not the literal path
    assert '"$CHESTER_WT_0"' in combined


def test_setup_commands_multiple_submodules():
    worktrees = {
        "IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0",
        "third_party/rl_games": "/remote/third_party/rl_games/.worktrees/wt1",
    }
    commits = {
        "IsaacLabTactile": "a" * 40,
        "third_party/rl_games": "b" * 40,
    }
    cmds = _build_worktree_setup_commands(worktrees, commits, "/remote")
    combined = "\n".join(cmds)
    assert "CHESTER_WT_0=" in combined
    assert "CHESTER_WT_1=" in combined
    assert "git -C /remote/IsaacLabTactile" in combined
    assert "git -C /remote/third_party/rl_games" in combined


def test_cleanup_commands_or_true():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    cmds = _build_worktree_cleanup_commands(worktrees, "/remote")
    combined = "\n".join(cmds)
    # Must use || true so cleanup doesn't fail on non-existent worktrees
    assert "|| true" in combined


def test_cleanup_commands_git_worktree_remove():
    worktrees = {"IsaacLabTactile": "/remote/IsaacLabTactile/.worktrees/wt0"}
    cmds = _build_worktree_cleanup_commands(worktrees, "/remote")
    combined = "\n".join(cmds)
    assert "git -C /remote/IsaacLabTactile worktree remove --force" in combined
    assert '"$CHESTER_WT_0"' in combined
