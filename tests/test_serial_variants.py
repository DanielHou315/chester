"""Tests for order='serial' variant collapsing logic in VariantGenerator.

Covers:
- Basic collapsing behavior
- _chester_serial_steps metadata structure and value ordering
- Base variant carries the first serial value
- hide=True on non-serial params
- Lambda dependencies on non-serial params combined with serial
- serial + dependent together
- _iter_serial_overrides edge cases
- Tuple values as serial values
- 3+ serial values
- Exact variant count verification
- Exclusion from variations()
- randomized=True raises ValueError when serial fields exist
- Multiple serial keys raises ValueError
"""

import pytest
from chester.run_exp import VariantGenerator, _iter_serial_overrides


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _find_variant(variants, **kwargs):
    """Return the first variant where all key=value pairs match."""
    for v in variants:
        if all(v.get(k) == val for k, val in kwargs.items()):
            return v
    raise KeyError(f"No variant found matching {kwargs}")


# ---------------------------------------------------------------------------
# a. Collapsing with multiple non-serial params
# ---------------------------------------------------------------------------

class TestCollapsingWithMultipleNonSerialParams:
    def test_two_non_serial_params_product(self):
        """serial on task, seed=[1,2], lr=[0.01,0.1] -> 4 collapsed variants."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        assert len(variants) == 4

    def test_all_collapsed_variants_carry_serial_steps(self):
        """Every collapsed variant must have _chester_serial_steps."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_serial_steps" in v

    def test_each_collapsed_variant_has_both_task_values(self):
        """Each collapsed variant's serial_steps should list both task values."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        for v in variants:
            key, vals = v["_chester_serial_steps"][0]
            assert key == "task"
            assert vals == ["training", "evaluate"]

    def test_distinct_non_serial_combos(self):
        """Each of the 4 collapsed variants covers a unique (seed, lr) combo."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        vg.add("lr", [0.01, 0.1])
        variants = vg.variants()
        combos = {(v["seed"], v["lr"]) for v in variants}
        assert combos == {(1, 0.01), (1, 0.1), (2, 0.01), (2, 0.1)}


# ---------------------------------------------------------------------------
# b. Serial steps preserve value order
# ---------------------------------------------------------------------------

class TestSerialStepsValueOrder:
    def test_two_values_order_preserved(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert vals == ["training", "evaluate"]

    def test_three_values_order_preserved(self):
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert vals == ["prep", "train", "eval"]

    def test_reverse_order_preserved(self):
        """If user gives values in reverse, that order must be retained."""
        vg = VariantGenerator()
        vg.add("task", ["evaluate", "training"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert vals == ["evaluate", "training"]

    def test_value_order_independent_of_seed(self):
        """Value order in serial_steps should be the same for every collapsed variant."""
        vg = VariantGenerator()
        vg.add("task", ["alpha", "beta", "gamma"], order="serial")
        vg.add("seed", [10, 20, 30])
        variants = vg.variants()
        orders = []
        for v in variants:
            key, vals = v["_chester_serial_steps"][0]
            orders.append(vals)
        # All groups should carry the same ordered list
        assert all(o == ["alpha", "beta", "gamma"] for o in orders)


# ---------------------------------------------------------------------------
# c. Serial key value appears in base variant (first value)
# ---------------------------------------------------------------------------

class TestSerialKeyBaseValue:
    def test_base_variant_has_first_serial_value(self):
        """The collapsed variant's value for the serial key must be the first one."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        assert variants[0]["task"] == "training"

    def test_base_variant_has_first_value_for_three_values(self):
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [42])
        variants = vg.variants()
        assert variants[0]["task"] == "prep"

    def test_base_value_consistent_across_all_collapsed_variants(self):
        """Across multiple non-serial combos, each collapsed variant has the first serial value."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2, 3])
        variants = vg.variants()
        for v in variants:
            assert v["task"] == "training"


# ---------------------------------------------------------------------------
# d. Serial with non-serial single-value params
# ---------------------------------------------------------------------------

class TestSerialWithSingleValueParams:
    def test_single_value_param_present_in_collapsed_variant(self):
        """Non-serial single-value params should still be in collapsed variants."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [42])
        variants = vg.variants()
        assert len(variants) == 1
        assert variants[0]["seed"] == 42

    def test_serial_steps_with_single_value_param(self):
        """Serial steps metadata present when non-serial params have one value."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [42])
        variants = vg.variants()
        assert "_chester_serial_steps" in variants[0]
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert key == "task"
        assert vals == ["training", "evaluate"]

    def test_serial_excluded_single_value_excluded_from_variations(self):
        """Both serial key and single-value param should be excluded from variations()."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [42])
        variations = vg.variations()
        assert "task" not in variations
        assert "seed" not in variations  # single value → not a variation


# ---------------------------------------------------------------------------
# e. Serial with lambda dependencies on non-serial params
# ---------------------------------------------------------------------------

class TestSerialWithLambdaDependencies:
    def test_lambda_dependent_lr_with_serial_task(self):
        """Lambda-dependent non-serial param works with serial collapsing."""
        vg = VariantGenerator()
        vg.add("seed", [1, 2])
        vg.add("lr", lambda seed: [seed * 0.01])
        vg.add("task", ["training", "evaluate"], order="serial")
        variants = vg.variants()
        # 2 seeds, serial tasks collapsed -> 2 variants
        assert len(variants) == 2

    def test_lambda_lr_value_correct_after_collapse(self):
        """Each collapsed variant should have the lambda-computed lr for its seed."""
        vg = VariantGenerator()
        vg.add("seed", [1, 2])
        vg.add("lr", lambda seed: [seed * 0.01])
        vg.add("task", ["training", "evaluate"], order="serial")
        variants = vg.variants()

        seed1_variant = _find_variant(variants, seed=1)
        seed2_variant = _find_variant(variants, seed=2)
        assert pytest.approx(seed1_variant["lr"]) == 0.01
        assert pytest.approx(seed2_variant["lr"]) == 0.02

    def test_lambda_serial_steps_intact(self):
        """Serial steps are intact even when lr is lambda-dependent."""
        vg = VariantGenerator()
        vg.add("seed", [1])
        vg.add("lr", lambda seed: [seed * 0.01])
        vg.add("task", ["training", "evaluate"], order="serial")
        variants = vg.variants()
        assert len(variants) == 1
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert key == "task"
        assert vals == ["training", "evaluate"]

    def test_lambda_multi_value_with_serial(self):
        """Lambda returning multiple values per seed produces correct collapsed count."""
        vg = VariantGenerator()
        vg.add("seed", [10, 20])
        vg.add("lr", lambda seed: [seed * 0.001, seed * 0.01])
        vg.add("task", ["training", "evaluate"], order="serial")
        variants = vg.variants()
        # 2 seeds × 2 lr values = 4 collapsed variants
        assert len(variants) == 4


# ---------------------------------------------------------------------------
# f. Serial + dependent together
# ---------------------------------------------------------------------------

class TestSerialAndDependentTogether:
    def test_serial_and_dependent_produce_correct_count(self):
        """With serial on task and dependent on phase, count = len(seed) * len(phase)."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        # serial collapses task (2 values) into each group
        # dependent on phase: 2 phase values × 2 seeds = 4 variants
        assert len(variants) == 4

    def test_serial_steps_present_with_dependent(self):
        """_chester_serial_steps must be in every variant when serial is used."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        vg.add("seed", [1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_serial_steps" in v

    def test_dependent_metadata_present_with_serial(self):
        """_chester_seq_identity and _chester_pred_identities must exist when dependent is used."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        vg.add("seed", [1])
        variants = vg.variants()
        for v in variants:
            assert "_chester_seq_identity" in v
            assert "_chester_pred_identities" in v

    def test_dependent_predecessor_chain_with_serial(self):
        """Predecessor relationships should be correct even alongside serial collapsing."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        vg.add("seed", [1])
        variants = vg.variants()

        warmup_variants = [v for v in variants if v["phase"] == "warmup"]
        finetune_variants = [v for v in variants if v["phase"] == "finetune"]

        # warmup variants have no predecessors for the phase key
        for v in warmup_variants:
            assert v["_chester_pred_identities"] == []

        # finetune variants have warmup predecessors
        for v in finetune_variants:
            assert len(v["_chester_pred_identities"]) == 1

    def test_serial_key_excluded_from_variations_with_dependent(self):
        """serial key still excluded from variations() when combined with dependent."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("phase", ["warmup", "finetune"], order="dependent")
        assert "task" not in vg.variations()


# ---------------------------------------------------------------------------
# g. _iter_serial_overrides — single key standard case
# ---------------------------------------------------------------------------

class TestIterSerialOverridesSingleKey:
    def test_yields_one_dict_per_value(self):
        overrides = list(_iter_serial_overrides([("task", ["training", "evaluate"])]))
        assert len(overrides) == 2

    def test_first_override_correct(self):
        overrides = list(_iter_serial_overrides([("task", ["training", "evaluate"])]))
        assert overrides[0] == {"task": "training"}

    def test_second_override_correct(self):
        overrides = list(_iter_serial_overrides([("task", ["training", "evaluate"])]))
        assert overrides[1] == {"task": "evaluate"}

    def test_three_values(self):
        overrides = list(_iter_serial_overrides([("stage", ["a", "b", "c"])]))
        assert overrides == [{"stage": "a"}, {"stage": "b"}, {"stage": "c"}]

    def test_key_name_preserved(self):
        overrides = list(_iter_serial_overrides([("my_key", ["x", "y"])]))
        for ov in overrides:
            assert "my_key" in ov

    def test_values_are_individual(self):
        """Each yielded dict should contain one value, not the full list."""
        overrides = list(_iter_serial_overrides([("task", ["training", "evaluate"])]))
        for ov in overrides:
            assert not isinstance(ov["task"], list)


# ---------------------------------------------------------------------------
# h. _iter_serial_overrides — empty list
# ---------------------------------------------------------------------------

class TestIterSerialOverridesEmptyList:
    def test_empty_list_yields_nothing(self):
        overrides = list(_iter_serial_overrides([]))
        assert overrides == []

    def test_empty_list_is_iterable(self):
        # Should not raise, just produce an empty iterator
        result = _iter_serial_overrides([])
        assert list(result) == []


# ---------------------------------------------------------------------------
# i. _iter_serial_overrides — None
# ---------------------------------------------------------------------------

class TestIterSerialOverridesNone:
    def test_none_returns_immediately(self):
        overrides = list(_iter_serial_overrides(None))
        assert overrides == []

    def test_none_does_not_raise(self):
        # Should not raise any exception
        for _ in _iter_serial_overrides(None):
            pass  # pragma: no cover


# ---------------------------------------------------------------------------
# j. Serial with tuple values
# ---------------------------------------------------------------------------

class TestSerialWithTupleValues:
    def test_tuple_values_in_serial_steps(self):
        """Tuple values as serial step values should be preserved as-is."""
        vg = VariantGenerator()
        vg.add("task", [("training",), ("evaluate",)], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        assert len(variants) == 1
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert key == "task"
        assert vals == [("training",), ("evaluate",)]

    def test_tuple_base_value(self):
        """Base variant should carry the first tuple as the serial key value."""
        vg = VariantGenerator()
        vg.add("task", [("training",), ("evaluate",)], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        assert variants[0]["task"] == ("training",)

    def test_tuple_values_with_multiple_seeds(self):
        """Tuple serial values should collapse correctly with multiple seeds."""
        vg = VariantGenerator()
        vg.add("task", [("training",), ("evaluate",)], order="serial")
        vg.add("seed", [1, 2])
        variants = vg.variants()
        assert len(variants) == 2
        for v in variants:
            key, vals = v["_chester_serial_steps"][0]
            assert vals == [("training",), ("evaluate",)]

    def test_iter_serial_overrides_with_tuples(self):
        """_iter_serial_overrides should yield dicts with tuple values intact."""
        overrides = list(_iter_serial_overrides([("task", [("training",), ("evaluate",)])]))
        assert overrides == [{"task": ("training",)}, {"task": ("evaluate",)}]


# ---------------------------------------------------------------------------
# k. Serial with 3+ values
# ---------------------------------------------------------------------------

class TestSerialWithThreePlusValues:
    def test_three_serial_values_in_steps(self):
        """All three values should appear in _chester_serial_steps."""
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert vals == ["prep", "train", "eval"]

    def test_five_serial_values(self):
        vg = VariantGenerator()
        vg.add("stage", ["a", "b", "c", "d", "e"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        key, vals = variants[0]["_chester_serial_steps"][0]
        assert vals == ["a", "b", "c", "d", "e"]

    def test_three_serial_values_base_is_first(self):
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        assert variants[0]["task"] == "prep"

    def test_three_serial_values_count_unaffected(self):
        """Variant count is based only on non-serial params, not number of serial values."""
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [1, 2, 3])
        variants = vg.variants()
        # 3 seeds, serial collapsed: should be 3 variants regardless of 3 serial values
        assert len(variants) == 3


# ---------------------------------------------------------------------------
# l. Variant count verification
# ---------------------------------------------------------------------------

class TestVariantCountVerification:
    def test_serial_two_values_three_seeds(self):
        """serial on task (2 values) + seed (3 values) = 3 variants, not 6."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2, 3])
        variants = vg.variants()
        assert len(variants) == 3

    def test_without_serial_two_values_three_seeds(self):
        """Without serial, same setup produces 6 variants (full cross-product)."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"])
        vg.add("seed", [1, 2, 3])
        variants = vg.variants()
        assert len(variants) == 6

    def test_serial_single_seed_single_variant(self):
        """serial + single seed -> exactly 1 collapsed variant."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [42])
        variants = vg.variants()
        assert len(variants) == 1

    def test_serial_no_other_params(self):
        """serial with no other params produces exactly 1 collapsed variant."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        variants = vg.variants()
        assert len(variants) == 1

    def test_serial_four_values_five_seeds(self):
        """serial on task (4 values) + seed (5 values) = 5 variants."""
        vg = VariantGenerator()
        vg.add("task", ["a", "b", "c", "d"], order="serial")
        vg.add("seed", [1, 2, 3, 4, 5])
        variants = vg.variants()
        assert len(variants) == 5

    def test_serial_two_non_serial_params(self):
        """serial on task (3 values) + seed (2) + lr (3) = 6 variants."""
        vg = VariantGenerator()
        vg.add("task", ["prep", "train", "eval"], order="serial")
        vg.add("seed", [1, 2])
        vg.add("lr", [0.001, 0.01, 0.1])
        variants = vg.variants()
        assert len(variants) == 6


# ---------------------------------------------------------------------------
# Additional edge-case and constraint tests
# ---------------------------------------------------------------------------

class TestSerialConstraints:
    def test_multiple_serial_keys_raises(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        with pytest.raises(ValueError, match="Only one serial key"):
            vg.add("phase", ["warmup", "finetune"], order="serial")

    def test_single_value_serial_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="at least 2 values"):
            vg.add("task", ["training"], order="serial")

    def test_callable_serial_raises(self):
        vg = VariantGenerator()
        with pytest.raises(ValueError, match="cannot be used with callable"):
            vg.add("task", lambda seed: ["train", "eval"], order="serial")

    def test_randomized_raises_with_serial(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        with pytest.raises(ValueError, match="randomized"):
            vg.variants(randomized=True)

    def test_serial_key_not_in_variations(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        assert "task" not in vg.variations()

    def test_non_serial_key_in_variations(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        assert "seed" in vg.variations()

    def test_get_serial_keys(self):
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2])
        assert vg.get_serial_keys() == ["task"]

    def test_serial_metadata_structure(self):
        """_chester_serial_steps must be a list of (key, values) tuples."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1])
        variants = vg.variants()
        steps = variants[0]["_chester_serial_steps"]
        assert isinstance(steps, list)
        assert len(steps) == 1
        key, vals = steps[0]
        assert isinstance(key, str)
        assert isinstance(vals, list)

    def test_no_serial_steps_without_serial_order(self):
        """Variants without any serial key should not carry _chester_serial_steps."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"])
        vg.add("seed", [1, 2])
        variants = vg.variants()
        for v in variants:
            assert "_chester_serial_steps" not in v

    def test_chester_first_last_markers_present(self):
        """chester_first_variant and chester_last_variant must be set after collapsing."""
        vg = VariantGenerator()
        vg.add("task", ["training", "evaluate"], order="serial")
        vg.add("seed", [1, 2, 3])
        variants = vg.variants()
        assert variants[0].get("chester_first_variant") is True
        assert variants[-1].get("chester_last_variant") is True
