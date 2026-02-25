# tests/test_hydra_utils.py
"""Tests for Hydra override formatting — especially OmegaConf interpolations."""
import pytest

from chester.hydra_utils import _format_hydra_value, variant_to_hydra_overrides


# ---------------------------------------------------------------------------
# _format_hydra_value
# ---------------------------------------------------------------------------

class TestFormatHydraValue:
    """Unit tests for _format_hydra_value."""

    # --- Interpolations (must pass through unquoted) ---

    def test_simple_interpolation(self):
        assert _format_hydra_value("${db.host}") == "${db.host}"

    def test_eval_resolver(self):
        val = "${eval:'2 ** 10'}"
        assert _format_hydra_value(val) == val

    def test_nested_interpolation(self):
        val = "${oc.select:model.${arch}.layers,4}"
        assert _format_hydra_value(val) == val

    def test_interpolation_with_default(self):
        val = "${oc.env:CUDA_VISIBLE_DEVICES,0}"
        assert _format_hydra_value(val) == val

    def test_interpolation_with_spaces_not_quoted(self):
        """Interpolations containing spaces must NOT be double-quoted."""
        val = "${eval:'len([1, 2, 3])'}"
        assert _format_hydra_value(val) == val
        assert '"' not in _format_hydra_value(val)

    # --- Plain strings ---

    def test_plain_string(self):
        assert _format_hydra_value("adam") == "adam"

    def test_string_with_spaces_quoted(self):
        assert _format_hydra_value("hello world") == '"hello world"'

    def test_string_with_dollar_not_interpolation(self):
        """A leading $ without braces is just a string."""
        assert _format_hydra_value("$HOME/data") == "$HOME/data"

    def test_partial_interpolation_not_special(self):
        """Only full ${...} wrapping counts as interpolation."""
        assert _format_hydra_value("prefix_${x}") == "prefix_${x}"

    # --- Booleans ---

    def test_bool_true(self):
        assert _format_hydra_value(True) == "true"

    def test_bool_false(self):
        assert _format_hydra_value(False) == "false"

    # --- Numerics ---

    def test_int(self):
        assert _format_hydra_value(42) == "42"

    def test_float(self):
        assert _format_hydra_value(0.001) == "0.001"

    # --- Lists ---

    def test_list_of_ints(self):
        assert _format_hydra_value([1, 2, 3]) == "[1,2,3]"

    def test_list_with_interpolation(self):
        result = _format_hydra_value(["${db.host}", 8080])
        assert result == "[${db.host},8080]"

    def test_list_with_comma_string(self):
        result = _format_hydra_value(["a,b", "c"])
        assert result == '["a,b",c]'


# ---------------------------------------------------------------------------
# variant_to_hydra_overrides
# ---------------------------------------------------------------------------

class TestVariantToHydraOverrides:
    """Tests for variant_to_hydra_overrides — full variant → override list."""

    def test_interpolation_in_variant(self):
        variant = {"lr": 0.01, "schedule": "${eval:'[0.1 * i for i in range(10)]'}"}
        overrides = variant_to_hydra_overrides(variant)
        assert "lr=0.01" in overrides
        assert "schedule=${eval:'[0.1 * i for i in range(10)]'}" in overrides

    def test_nested_dict_with_interpolation(self):
        variant = {
            "model": {"_name": "resnet", "layers": "${eval:'2 ** 4'}"},
        }
        overrides = variant_to_hydra_overrides(variant)
        assert "model=resnet" in overrides
        assert "model.layers=${eval:'2 ** 4'}" in overrides

    def test_skips_chester_keys(self):
        variant = {
            "chester_first_variant": True,
            "chester_last_variant": False,
            "is_debug": False,
            "lr": 0.01,
        }
        overrides = variant_to_hydra_overrides(variant)
        assert overrides == ["lr=0.01"]

    def test_env_resolver(self):
        variant = {"data_dir": "${oc.env:DATA_DIR,/tmp/data}"}
        overrides = variant_to_hydra_overrides(variant)
        assert overrides == ["data_dir=${oc.env:DATA_DIR,/tmp/data}"]

    def test_mixed_types(self):
        variant = {
            "seed": 42,
            "use_amp": True,
            "arch": "transformer",
            "hidden": "${eval:'512 * 2'}",
        }
        overrides = variant_to_hydra_overrides(variant)
        assert "seed=42" in overrides
        assert "use_amp=true" in overrides
        assert "arch=transformer" in overrides
        assert "hidden=${eval:'512 * 2'}" in overrides
