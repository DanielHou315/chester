# tests/test_config_v2.py
import os
import pytest
import tempfile
from pathlib import Path


def test_load_config_from_chester_dir(tmp_path):
    """Config loads from .chester/config.yaml."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    config_file = chester_dir / "config.yaml"
    config_file.write_text("""
project_path: /tmp/test_project
log_dir: data
package_manager: uv
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["project_path"] == "/tmp/test_project"
    assert cfg["package_manager"] == "uv"
    assert "local" in cfg["backends"]


def test_load_config_falls_back_to_chester_yaml(tmp_path):
    """Falls back to chester.yaml at project root with deprecation warning."""
    from chester.config_v2 import load_config

    config_file = tmp_path / "chester.yaml"
    config_file.write_text("""
log_dir: data
package_manager: conda
backends:
  local:
    type: local
""")
    with pytest.warns(DeprecationWarning, match="chester.yaml"):
        cfg = load_config(search_from=tmp_path)
    assert cfg["package_manager"] == "conda"


def test_load_config_validates_backends(tmp_path):
    """Invalid backend type raises ValueError."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  bad:
    type: ec2
""")
    with pytest.raises(ValueError, match="Unknown backend type"):
        load_config(search_from=tmp_path)


def test_load_config_auto_detects_project_path(tmp_path):
    """project_path defaults to config directory's parent."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: data
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["project_path"] == str(tmp_path)


def test_load_config_resolves_log_dir(tmp_path):
    """log_dir is resolved relative to project_path."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: my_data
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    assert cfg["log_dir"] == os.path.join(str(tmp_path), "my_data")


def test_load_config_default_local_backend(tmp_path):
    """If no backends defined, a default 'local' backend is created."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
log_dir: data
""")
    cfg = load_config(search_from=tmp_path)
    assert "local" in cfg["backends"]
    assert cfg["backends"]["local"].type == "local"


def test_load_config_package_manager_validation(tmp_path):
    """Only python, conda, uv are valid package managers."""
    from chester.config_v2 import load_config

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
package_manager: poetry
backends:
  local:
    type: local
""")
    with pytest.raises(ValueError, match="package_manager"):
        load_config(search_from=tmp_path)


def test_load_config_env_var_override(tmp_path, monkeypatch):
    """CHESTER_CONFIG_PATH env var overrides search."""
    from chester.config_v2 import load_config

    config_file = tmp_path / "custom_config.yaml"
    config_file.write_text("""
package_manager: python
backends:
  local:
    type: local
""")
    monkeypatch.setenv("CHESTER_CONFIG_PATH", str(config_file))
    cfg = load_config()
    assert cfg["package_manager"] == "python"


def test_get_backend_returns_parsed_config(tmp_path):
    """get_backend returns a BackendConfig for the named backend."""
    from chester.config_v2 import load_config, get_backend

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  myhost:
    type: ssh
    host: myhost.example.com
    remote_dir: /home/user/project
""")
    cfg = load_config(search_from=tmp_path)
    backend = get_backend("myhost", cfg)
    assert backend.type == "ssh"
    assert backend.host == "myhost.example.com"


def test_get_backend_unknown_raises(tmp_path):
    """get_backend raises KeyError for unknown backend."""
    from chester.config_v2 import load_config, get_backend

    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text("""
backends:
  local:
    type: local
""")
    cfg = load_config(search_from=tmp_path)
    with pytest.raises(KeyError, match="nonexistent"):
        get_backend("nonexistent", cfg)
