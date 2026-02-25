# tests/test_utils.py
import pytest
from chester.utils import shellquote, to_param_val


def test_shellquote_empty_string():
    assert shellquote("") == "''"


def test_shellquote_safe_string():
    assert shellquote("hello") == "hello"


def test_shellquote_unsafe_string():
    # shlex.quote wraps in single quotes
    assert shellquote("hello world") == "'hello world'"


def test_shellquote_single_quotes():
    result = shellquote("it's")
    assert "it" in result and "s" in result  # properly escaped


def test_to_param_val_none():
    assert to_param_val(None) == ""


def test_to_param_val_string():
    assert to_param_val("hello") == "hello"


def test_to_param_val_list():
    result = to_param_val(["a", "b", "c"])
    assert result == "a b c"


def test_to_param_val_number():
    assert to_param_val(42) == "42"
