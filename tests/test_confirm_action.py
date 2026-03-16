from chester.run_exp import confirm_action

def test_confirm_action_skip():
    assert confirm_action("Delete?", skip=True) is True

def test_confirm_action_yes(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _="": 'yes')
    assert confirm_action("Delete?") is True

def test_confirm_action_y(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _="": 'y')
    assert confirm_action("Delete?") is True

def test_confirm_action_no(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _="": 'no')
    assert confirm_action("Delete?") is False

def test_confirm_action_default_yes(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _="": '')
    assert confirm_action("Delete?", default="yes") is True

def test_confirm_action_default_no(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda _="": '')
    assert confirm_action("Delete?", default="no") is False
