import os
import pytest
from chester.run_exp import run_experiment_lite, VariantGenerator


class TestOrderMetadata:
    def test_get_dependent_keys_none(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        vg.add("seed", [1, 2])
        assert vg.get_dependent_keys() == []

    def test_get_dependent_keys_one(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("seed", [1, 2])
        assert vg.get_dependent_keys() == ["task"]

    def test_get_dependent_keys_multiple(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("seed", [1, 2])
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        assert set(vg.get_dependent_keys()) == {"task", "phase"}

    def test_get_serial_keys(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        assert vg.get_serial_keys() == ["task"]

    def test_order_single_value_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="at least 2 values"):
            vg.add("task", ["training"], order="dependent")

    def test_serial_single_value_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="at least 2 values"):
            vg.add("task", ["training"], order="serial")

    def test_order_with_callable_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="cannot be used with callable"):
            vg.add("task", lambda seed: ["train", "eval"], order="dependent")

    def test_invalid_order_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="not valid"):
            vg.add("task", ["a", "b"], order="parallel")

    def test_multiple_serial_keys_raises(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        with pytest.raises(ValueError, match="Only one serial key"):
            vg.add("phase", ["warmup", "finetune"], order="serial")


class TestDependencyMap:
    def test_no_dependent_keys(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)
        assert dep_map == {}

    def test_single_dependent_field(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(task, seed):
            for i, v in enumerate(variants):
                if v["task"] == task and v["seed"] == seed:
                    return i
            raise ValueError(f"Not found: task={task}, seed={seed}")

        assert find("training", 1) not in dep_map
        assert find("training", 2) not in dep_map
        assert dep_map[find("evaluate", 1)] == [find("training", 1)]
        assert dep_map[find("evaluate", 2)] == [find("training", 2)]

    def test_two_dependent_fields(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(**kw):
            for i, v in enumerate(variants):
                if all(v[k] == val for k, val in kw.items()):
                    return i
            raise ValueError(f"Not found: {kw}")

        assert find(task="training", phase="warmup") not in dep_map

        assert dep_map[find(task="evaluate", phase="warmup")] == [
            find(task="training", phase="warmup")
        ]

        assert dep_map[find(task="training", phase="finetune")] == [
            find(task="training", phase="warmup")
        ]

        deps = set(dep_map[find(task="evaluate", phase="finetune")])
        assert deps == {
            find(task="training", phase="finetune"),
            find(task="evaluate", phase="warmup"),
        }

    def test_three_value_chain(self):
        vg = VariantGenerator()
        vg.add("stage", ["prep", "train", "eval"], order="dependent")
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(stage):
            for i, v in enumerate(variants):
                if v["stage"] == stage:
                    return i

        assert find("prep") not in dep_map
        assert dep_map[find("train")] == [find("prep")]
        assert dep_map[find("eval")] == [find("train")]


class TestVariantDependentMetadata:
    def test_variants_carry_seq_identity(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" in v
            assert isinstance(v["_chester_seq_identity"], tuple)

    def test_variants_carry_pred_identities(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_pred_identities" in v

        for v in variants:
            if v["task"] == "training":
                assert v["_chester_pred_identities"] == []
            else:
                assert len(v["_chester_pred_identities"]) == 1

    def test_no_dep_metadata_when_no_dependent(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" not in v
            assert "_chester_pred_identities" not in v

    def test_randomized_with_dependent_raises(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="dependent")
        with pytest.raises(ValueError, match="randomized"):
            vg.variants(randomized=True)

    def test_randomized_with_serial_raises(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        with pytest.raises(ValueError, match="randomized"):
            vg.variants(randomized=True)


class TestSerialVariants:
    def test_serial_collapses_variants(self):
        """order='serial' should collapse cross-product into single variants."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        # 2 seeds, serial tasks collapsed: should be 2 variants
        assert len(variants) == 2

    def test_serial_carries_steps(self):
        """Collapsed variants should carry _chester_serial_steps metadata."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        assert len(variants) == 1
        v = variants[0]
        assert "_chester_serial_steps" in v
        steps = v["_chester_serial_steps"]
        assert len(steps) == 1
        key, vals = steps[0]
        assert key == "task"
        assert vals == ["training", "evaluate"]

    def test_serial_excluded_from_variations(self):
        """Serial keys should not appear in variations() (don't affect exp_name)."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        variations = vg.variations()
        assert "task" not in variations
        assert "seed" in variations

    def test_serial_no_dep_metadata(self):
        """Serial variants should not carry dependency metadata."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" not in v
            assert "_chester_pred_identities" not in v


class TestRunExpDependencyWiring:
    def _make_local_config(self, tmp_path):
        chester_dir = tmp_path / ".chester"
        chester_dir.mkdir()
        (chester_dir / "config.yaml").write_text(f"""
log_dir: {tmp_path}/data
project_path: {tmp_path}
package_manager: python
backends:
  local:
    type: local
""")
        return chester_dir

    def test_non_slurm_with_dependent_raises(self, tmp_path):
        """Non-SLURM mode with order='dependent' should raise ValueError."""
        chester_dir = self._make_local_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="dependent")
            variants = vg.variants()

            with pytest.raises(ValueError, match="order='dependent'"):
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=variants[0],
                    mode="local",
                    exp_prefix="test",
                    skip_dependency_check=False,
                    dry=True,
                    git_snapshot=False,
                )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

    def test_non_slurm_with_skip_does_not_raise(self, tmp_path):
        """skip_dependency_check=True should suppress the error."""
        chester_dir = self._make_local_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="dependent")
            variants = vg.variants()

            # Should not raise
            run_experiment_lite(
                stub_method_call=lambda v, log_dir, exp_name: None,
                variant=variants[0],
                mode="local",
                exp_prefix="test",
                skip_dependency_check=True,
                dry=True,
                git_snapshot=False,
            )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)

    def test_no_order_no_warning(self, tmp_path):
        """No ordered fields -> no dependency check, no raise."""
        chester_dir = self._make_local_config(tmp_path)
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
            vg = VariantGenerator()
            vg.add("lr", [0.01, 0.1])
            variants = vg.variants()

            run_experiment_lite(
                stub_method_call=lambda v, log_dir, exp_name: None,
                variant=variants[0],
                mode="local",
                exp_prefix="test",
                skip_dependency_check=False,
                dry=True,
                git_snapshot=False,
            )
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)


class TestDependentIntegration:
    """End-to-end test simulating a launcher loop with order='dependent' on SLURM."""

    def test_slurm_dependency_chain(self, monkeypatch, tmp_path):
        """Simulate launching dependent variants and verify dependency flags."""
        import chester.run_exp as run_exp

        # Set up config with a SLURM backend
        chester_dir = tmp_path / ".chester"
        chester_dir.mkdir()
        (chester_dir / "config.yaml").write_text(f"""
log_dir: {tmp_path}/data
project_path: {tmp_path}
package_manager: python
backends:
  gl:
    type: slurm
    host: gl-login.example.com
    remote_dir: /home/user/project
    slurm:
      partition: spgpu
      time: "1:00:00"
      gpus: 1
""")
        monkeypatch.setenv("CHESTER_CONFIG_PATH", str(chester_dir / "config.yaml"))

        old_cwd = os.getcwd()
        os.chdir(tmp_path)

        try:
            # Reset module-level state
            run_exp.exp_count = -2
            run_exp.remote_confirmed = False
            run_exp._slurm_job_registry.clear()

            vg = VariantGenerator()
            vg.add("task", ["training", "evaluate"], order="dependent")
            vg.add("seed", [1])
            variants = vg.variants()

            submitted_jobs = []

            def mock_submit(self, task, script_content, dry=False, dependency_job_ids=None):
                job_id = len(submitted_jobs) + 100
                submitted_jobs.append({
                    "exp_name": task.get("exp_name", task.get("params", {}).get("exp_name", "")),
                    "dependency_job_ids": dependency_job_ids,
                    "job_id": job_id,
                })
                return job_id

            from chester.backends.slurm import SlurmBackend
            monkeypatch.setattr(SlurmBackend, "submit", mock_submit)
            monkeypatch.setattr(run_exp, "rsync_code_v2", lambda **kw: None)

            # Launch all variants
            for v in variants:
                run_experiment_lite(
                    stub_method_call=lambda v, log_dir, exp_name: None,
                    variant=v,
                    mode="gl",
                    exp_prefix="seq_test",
                    confirm=True,
                    dry=False,
                    git_snapshot=False,
                    auto_pull=False,
                )

            # First job (training) should have no dependencies
            assert submitted_jobs[0]["dependency_job_ids"] is None

            # Second job (evaluate) should depend on first job
            assert submitted_jobs[1]["dependency_job_ids"] == [100]
        finally:
            os.chdir(old_cwd)
            os.environ.pop("CHESTER_CONFIG_PATH", None)
