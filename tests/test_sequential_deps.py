import pytest
from chester.run_exp import VariantGenerator


class TestSequentialMetadata:
    def test_get_sequential_keys_none(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        vg.add("seed", [1, 2])
        assert vg.get_sequential_keys() == []

    def test_get_sequential_keys_one(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        assert vg.get_sequential_keys() == ["task"]

    def test_get_sequential_keys_multiple(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        vg.add("phase", ["warmup", "finetune"], sequential=True)
        assert set(vg.get_sequential_keys()) == {"task", "phase"}

    def test_sequential_single_value_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="at least 2 values"):
            vg.add("task", ["training"], sequential=True)

    def test_sequential_with_callable_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="cannot be used with callable"):
            vg.add("task", lambda seed: ["train", "eval"], sequential=True)


class TestDependencyMap:
    def test_no_sequential_keys(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)
        assert dep_map == {}

    def test_single_sequential_field(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
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

    def test_two_sequential_fields(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("phase", ["warmup", "finetune"], sequential=True)
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
        vg.add("stage", ["prep", "train", "eval"], sequential=True)
        variants = vg.variants()
        dep_map = vg.get_dependency_map(variants)

        def find(stage):
            for i, v in enumerate(variants):
                if v["stage"] == stage:
                    return i

        assert find("prep") not in dep_map
        assert dep_map[find("train")] == [find("prep")]
        assert dep_map[find("eval")] == [find("train")]


class TestVariantSequentialMetadata:
    def test_variants_carry_seq_identity(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" in v
            assert isinstance(v["_chester_seq_identity"], tuple)

    def test_variants_carry_pred_identities(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_pred_identities" in v

        for v in variants:
            if v["task"] == "training":
                assert v["_chester_pred_identities"] == []
            else:
                assert len(v["_chester_pred_identities"]) == 1

    def test_no_seq_metadata_when_no_sequential(self):
        vg = VariantGenerator()
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" not in v
            assert "_chester_pred_identities" not in v

    def test_randomized_with_sequential_raises(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], sequential=True)
        with pytest.raises(ValueError, match="randomized"):
            vg.variants(randomized=True)
