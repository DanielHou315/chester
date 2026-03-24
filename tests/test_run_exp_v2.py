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


class TestRemoteJobWritesJobFile:
    """Verify that a remote job submission with auto_pull=True writes one job file."""

    def _make_chester_config(self, tmp_path):
        chester_dir = tmp_path / ".chester"
        chester_dir.mkdir()
        (chester_dir / "config.yaml").write_text(f"""
log_dir: {tmp_path}/data
project_path: {tmp_path}
package_manager: python
backends:
  myremote:
    type: ssh
    host: myhost.example.com
    remote_dir: /remote/project
""")
        return chester_dir

    def test_auto_pull_writes_exactly_one_job_file(self, tmp_path, monkeypatch):
        """Remote submission with auto_pull=True writes one pending job file."""
        import json
        import os
        from pathlib import Path

        chester_dir = self._make_chester_config(tmp_path)
        job_store_dir = tmp_path / "job_store"

        # Redirect job store dir to tmp_path so the real _register_job_for_pull
        # writes files into a known location without touching the real project dir.
        monkeypatch.setenv("CHESTER_CONFIG_PATH", str(chester_dir / "config.yaml"))
        import chester.job_store as job_store_mod
        monkeypatch.setattr(job_store_mod, "get_default_job_store_dir", lambda: job_store_dir)

        import chester.run_exp as run_exp_mod

        # Mock the backend submission so we don't actually SSH anywhere
        import chester.backends as backends_mod

        class FakeSSHBackend:
            def __init__(self):
                self.config = type('Cfg', (), {
                    'type': 'ssh',
                    'host': 'myhost.example.com',
                    'remote_dir': '/remote/project',
                    'singularity': None,
                })()

            def generate_script(self, task, **kwargs):
                return "#!/bin/bash\necho hello"

            def submit(self, task, script_content, dry=False):
                return None  # SSH backends return None (no job id)

        fake_backend = FakeSSHBackend()

        original_create = backends_mod.create_backend
        monkeypatch.setattr(backends_mod, "create_backend",
                            lambda bc, cfg: fake_backend)
        monkeypatch.setattr(run_exp_mod, "create_backend",
                            lambda bc, cfg: fake_backend)

        # Also stub rsync so it doesn't run
        monkeypatch.setattr(run_exp_mod, "rsync_code_v2", lambda **kw: None)

        # Bypass remote confirmation prompt
        monkeypatch.setattr(run_exp_mod, "remote_confirmed", True)

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            run_exp_mod.run_experiment_lite(
                stub_method_call=lambda v, l, e: None,
                variant={
                    "chester_first_variant": True,
                    "chester_last_variant": True,
                },
                mode="myremote",
                exp_prefix="test_autopull",
                auto_pull=True,
                dry=False,
                git_snapshot=False,
                confirm=True,
            )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

        # Verify exactly one .json file was written
        json_files = list(job_store_dir.glob("*.json"))
        assert len(json_files) == 1, f"Expected 1 job file, got {len(json_files)}"

        job_data = json.loads(json_files[0].read_text())
        assert job_data.get("status") == "pending"
        assert job_data.get("host") is not None
        assert job_data["host"] == "myhost.example.com"
        assert job_data.get("exp_prefix") == "test_autopull"
        assert job_data.get("exp_name") is not None and len(job_data["exp_name"]) > 0
        assert job_data.get("remote_log_dir", "").startswith("/remote/project")
        assert "local_log_dir" in job_data


class TestVariantGeneratorDerive:
    def test_derive_computes_value_from_variant(self):
        from chester.run_exp import VariantGenerator
        vg = VariantGenerator()
        vg.add("num_train_sim", [127])
        vg.derive("num_train_real", lambda v: 128 - v["num_train_sim"])
        variants = vg.variants()
        assert variants[0]["num_train_real"] == 1

    def test_derive_applied_to_all_variants(self):
        from chester.run_exp import VariantGenerator
        vg = VariantGenerator()
        vg.add("num_train_sim", [127, 63, 1])
        vg.derive("num_train_real", lambda v: 128 - v["num_train_sim"])
        for vv in vg.variants():
            assert vv["num_train_sim"] + vv["num_train_real"] == 128

    def test_derive_chain_later_can_reference_earlier(self):
        from chester.run_exp import VariantGenerator
        vg = VariantGenerator()
        vg.add("num_train_sim", [127, 1])
        vg.derive("num_train_real", lambda v: 128 - v["num_train_sim"])
        vg.derive("sim_scale", lambda v: 2.0 if v["num_train_sim"] > v["num_train_real"] else 1.0)
        variants = vg.variants()
        assert variants[0]["sim_scale"] == 2.0  # 127 > 1
        assert variants[1]["sim_scale"] == 1.0  # 1 < 127

    def test_derive_dotted_keys(self):
        from chester.run_exp import VariantGenerator
        vg = VariantGenerator()
        vg.add("experiment.training.env.num_train_sim", [127])
        vg.derive(
            "experiment.training.env.num_train_real",
            lambda v: 128 - v["experiment.training.env.num_train_sim"],
        )
        variants = vg.variants()
        assert variants[0]["experiment.training.env.num_train_real"] == 1

    def test_derive_does_not_affect_base_sweep(self):
        from chester.run_exp import VariantGenerator
        vg = VariantGenerator()
        vg.add("seed", [1, 2])
        vg.derive("doubled", lambda v: v["seed"] * 2)
        variants = vg.variants()
        assert len(variants) == 2
        assert variants[0]["doubled"] == 2
        assert variants[1]["doubled"] == 4


import subprocess


def test_validate_submodule_commits_resolves_sha(tmp_path):
    from chester.run_exp import _validate_submodule_commits

    # Create a fake git repo in tmp_path/MySub
    sub_path = tmp_path / "MySub"
    sub_path.mkdir()
    subprocess.run(["git", "init"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub_path, check=True, capture_output=True)
    (sub_path / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sub_path, check=True, capture_output=True)
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sub_path, text=True
    ).strip()

    result = _validate_submodule_commits({"MySub": sha[:8]}, str(tmp_path))
    assert result["MySub"] == sha  # resolved to full 40-char SHA


def test_validate_submodule_commits_raises_on_bad_ref(tmp_path):
    from chester.run_exp import _validate_submodule_commits

    sub_path = tmp_path / "MySub"
    sub_path.mkdir()
    subprocess.run(["git", "init"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=sub_path, check=True, capture_output=True)
    (sub_path / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=sub_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=sub_path, check=True, capture_output=True)

    with pytest.raises(ValueError, match="MySub"):
        _validate_submodule_commits({"MySub": "deadbeef0000"}, str(tmp_path))


def test_validate_submodule_commits_raises_on_missing_path(tmp_path):
    from chester.run_exp import _validate_submodule_commits
    with pytest.raises(ValueError, match="not found"):
        _validate_submodule_commits({"nonexistent": "abc"}, str(tmp_path))


def test_build_worktree_paths_format():
    from chester.run_exp import _build_worktree_paths
    commits = {"IsaacLabTactile": "abc1234def5678" + "a" * 26}
    paths = _build_worktree_paths(commits, "/remote/project", "03_23_10_00")
    wt = paths["IsaacLabTactile"]
    assert wt.startswith("/remote/project/IsaacLabTactile/.worktrees/")
    assert "03_23_10_00" in wt
    assert "abc1234def5" in wt  # short SHA present (first 12 chars)


def test_build_worktree_paths_unique():
    from chester.run_exp import _build_worktree_paths
    commits = {"IsaacLabTactile": "a" * 40}
    p1 = _build_worktree_paths(commits, "/remote", "03_23_10_00")
    p2 = _build_worktree_paths(commits, "/remote", "03_23_10_00")
    # Random suffix makes them unique even with same timestamp
    assert p1["IsaacLabTactile"] != p2["IsaacLabTactile"]
