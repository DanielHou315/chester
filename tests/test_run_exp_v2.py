# tests/test_run_exp_v2.py
import os
import pytest


def test_run_experiment_lite_rejects_ec2_mode():
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="ec2",
            exp_prefix="test",
        )


def test_run_experiment_lite_rejects_autobot_mode():
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="autobot",
            exp_prefix="test",
        )


def test_run_experiment_lite_rejects_singularity_mode():
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="singularity",
            exp_prefix="test",
        )


def test_run_experiment_lite_rejects_local_singularity_mode():
    from chester.run_exp import run_experiment_lite
    with pytest.raises(ValueError, match="deprecated"):
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="local_singularity",
            exp_prefix="test",
        )


def test_variant_generator_unchanged():
    from chester.run_exp import VariantGenerator
    vg = VariantGenerator()
    vg.add('lr', [0.001, 0.01])
    vg.add('bs', [32, 64])
    variants = vg.variants()
    assert len(variants) == 4
    assert variants[0].get('chester_first_variant')
    assert variants[-1].get('chester_last_variant')


def test_run_experiment_lite_local_dry_run(tmp_path):
    """Local backend dry run should not crash."""
    from chester.run_exp import run_experiment_lite

    # Set up minimal config
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: data
package_manager: python
backends:
  local:
    type: local
""")

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
        run_experiment_lite(
            stub_method_call=lambda v, l, e: None,
            variant={"chester_first_variant": True, "chester_last_variant": True},
            mode="local",
            exp_prefix="test",
            dry=True,
            git_snapshot=False,
        )
    finally:
        os.chdir(old_cwd)
        os.environ.pop("CHESTER_CONFIG_PATH", None)


def test_run_experiment_lite_mutable_defaults():
    """Verify that env={} and variations=[] mutable defaults are fixed."""
    import inspect
    from chester.run_exp import run_experiment_lite

    sig = inspect.signature(run_experiment_lite)

    # env default should be None, not {}
    assert sig.parameters["env"].default is None
    # variations default should be None, not []
    assert sig.parameters["variations"].default is None


def test_run_experiment_lite_has_slurm_overrides_param():
    """Verify the new slurm_overrides parameter exists."""
    import inspect
    from chester.run_exp import run_experiment_lite

    sig = inspect.signature(run_experiment_lite)
    assert "slurm_overrides" in sig.parameters


def test_run_experiment_lite_has_git_snapshot_param():
    """Verify the new git_snapshot parameter exists."""
    import inspect
    from chester.run_exp import run_experiment_lite

    sig = inspect.signature(run_experiment_lite)
    assert "git_snapshot" in sig.parameters
    # Default should be True
    assert sig.parameters["git_snapshot"].default is True


def test_map_local_to_remote_log_dir_v2():
    """Test the new v2 log dir mapping function."""
    from chester.run_exp import _map_local_to_remote_log_dir_v2

    result = _map_local_to_remote_log_dir_v2(
        local_log_dir="/home/user/project/data/train/exp1",
        project_path="/home/user/project",
        remote_dir="/home/remote/project",
    )
    assert result == "/home/remote/project/data/train/exp1"


def test_map_local_to_remote_log_dir_v2_rejects_outside():
    """Test that v2 mapping rejects paths outside project."""
    from chester.run_exp import _map_local_to_remote_log_dir_v2

    with pytest.raises(ValueError, match="project_path"):
        _map_local_to_remote_log_dir_v2(
            local_log_dir="/other/path/logs",
            project_path="/home/user/project",
            remote_dir="/home/remote/project",
        )


def test_resolve_extra_pull_dirs_v2():
    """Test v2 extra pull dirs resolution."""
    from chester.run_exp import _resolve_extra_pull_dirs_v2

    result = _resolve_extra_pull_dirs_v2(
        extra_pull_dirs=["models", "/absolute/path"],
        project_path="/home/user/project",
        remote_dir="/home/remote/project",
    )
    assert len(result) == 2
    assert result[0] == {"local": "/home/user/project/models", "remote": "/home/remote/project/models"}
    assert result[1] == {"local": "/absolute/path", "remote": "/absolute/path"}


def test_resolve_extra_pull_dirs_v2_empty():
    """Test v2 extra pull dirs with None input."""
    from chester.run_exp import _resolve_extra_pull_dirs_v2

    assert _resolve_extra_pull_dirs_v2(None, "/p", "/r") == []
    assert _resolve_extra_pull_dirs_v2([], "/p", "/r") == []
