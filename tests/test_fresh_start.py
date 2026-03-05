import os
import pytest
from unittest.mock import patch


def test_format_size_bytes():
    from chester.run_exp import _format_size
    assert _format_size(0) == "0 B"
    assert _format_size(512) == "512 B"
    assert _format_size(1023) == "1023 B"


def test_format_size_kb():
    from chester.run_exp import _format_size
    assert _format_size(1024) == "1.0 KB"
    assert _format_size(1536) == "1.5 KB"


def test_format_size_mb():
    from chester.run_exp import _format_size
    assert _format_size(1024 * 1024) == "1.0 MB"
    assert _format_size(int(2.5 * 1024 * 1024)) == "2.5 MB"


def test_format_size_gb():
    from chester.run_exp import _format_size
    assert _format_size(1024 ** 3) == "1.0 GB"


def test_scan_local_batch_dir_empty(tmp_path):
    from chester.run_exp import _scan_local_batch_dir
    result = _scan_local_batch_dir(str(tmp_path / "nonexistent"))
    assert result == []


def test_scan_local_batch_dir_with_variants(tmp_path):
    from chester.run_exp import _scan_local_batch_dir

    # Create two fake variant dirs
    var1 = tmp_path / "1_myexp_lr_0.001"
    var1.mkdir()
    (var1 / "checkpoint.pt").write_bytes(b"x" * 1024)
    (var1 / "final.pth").write_bytes(b"x" * 2048)
    (var1 / "log.txt").write_bytes(b"x" * 512)

    var2 = tmp_path / "2_myexp_lr_0.01"
    var2.mkdir()
    (var2 / "checkpoint.pt").write_bytes(b"x" * 4096)

    result = _scan_local_batch_dir(str(tmp_path))

    assert len(result) == 2
    names = [r["name"] for r in result]
    assert "1_myexp_lr_0.001" in names
    assert "2_myexp_lr_0.01" in names

    row1 = next(r for r in result if r["name"] == "1_myexp_lr_0.001")
    assert row1["pt_count"] == 2          # .pt and .pth
    assert row1["size_bytes"] == 1024 + 2048 + 512

    row2 = next(r for r in result if r["name"] == "2_myexp_lr_0.01")
    assert row2["pt_count"] == 1
    assert row2["size_bytes"] == 4096


def test_scan_local_batch_dir_ignores_files(tmp_path):
    from chester.run_exp import _scan_local_batch_dir

    # Files at batch_dir level should not appear as variant rows
    (tmp_path / "somefile.txt").write_text("hello")
    var1 = tmp_path / "1_myexp"
    var1.mkdir()

    result = _scan_local_batch_dir(str(tmp_path))
    assert len(result) == 1
    assert result[0]["name"] == "1_myexp"


def test_scan_local_batch_dir_counts_nested_pt(tmp_path):
    from chester.run_exp import _scan_local_batch_dir

    var1 = tmp_path / "1_myexp"
    var1.mkdir()
    sub = var1 / "checkpoints"
    sub.mkdir()
    (sub / "epoch_1.pt").write_bytes(b"a" * 100)
    (sub / "epoch_2.pt").write_bytes(b"b" * 100)

    result = _scan_local_batch_dir(str(tmp_path))
    assert result[0]["pt_count"] == 2


def test_scan_remote_batch_dir_parses_output():
    from chester.run_exp import _scan_remote_batch_dir
    fake_output = (
        "1_myexp_lr_0.001|2.3G|47\n"
        "2_myexp_lr_0.01|—|0\n"
        "3_myexp_lr_0.1|512M|8\n"
    )
    with patch("chester.run_exp.subprocess.run") as mock_run:
        mock_run.return_value.stdout = fake_output
        mock_run.return_value.returncode = 0
        result = _scan_remote_batch_dir("myhost", "/remote/logs/train/myexp")

    assert len(result) == 3

    r1 = next(r for r in result if r["name"] == "1_myexp_lr_0.001")
    assert r1["exists"] is True
    assert r1["size_str"] == "2.3G"
    assert r1["pt_count"] == 47

    r2 = next(r for r in result if r["name"] == "2_myexp_lr_0.01")
    assert r2["exists"] is False
    assert r2["pt_count"] == 0

    r3 = next(r for r in result if r["name"] == "3_myexp_lr_0.1")
    assert r3["exists"] is True
    assert r3["pt_count"] == 8


def test_scan_remote_batch_dir_ssh_failure():
    from chester.run_exp import _scan_remote_batch_dir
    with patch("chester.run_exp.subprocess.run") as mock_run:
        mock_run.side_effect = Exception("SSH timeout")
        result = _scan_remote_batch_dir("myhost", "/remote/logs/train/myexp")

    assert result is None


def test_scan_remote_batch_dir_empty_output():
    from chester.run_exp import _scan_remote_batch_dir
    with patch("chester.run_exp.subprocess.run") as mock_run:
        mock_run.return_value.stdout = ""
        mock_run.return_value.returncode = 0
        result = _scan_remote_batch_dir("myhost", "/remote/logs/train/myexp")

    assert result == []


def _make_batch_tasks(tmp_path, exp_prefix="myexp", n=3):
    """Build minimal batch_tasks list with _local_log_dir and log_dir set."""
    tasks = []
    for i in range(1, n + 1):
        exp_name = f"{i}_{exp_prefix}_lr_{i}"
        local_dir = str(tmp_path / "data" / "train" / exp_prefix / exp_name)
        tasks.append({
            "exp_name": exp_name,
            "_local_log_dir": local_dir,
            "log_dir": local_dir,  # local backend: same as _local_log_dir
        })
    return tasks


def test_fresh_start_v2_no_dirs_prints_message(tmp_path, capsys):
    from chester.run_exp import _fresh_start_v2

    batch_dir = tmp_path / "data" / "train" / "myexp"
    # batch_dir doesn't exist → no existing dirs

    _fresh_start_v2(
        exp_prefix="myexp",
        sub_dir="train",
        cfg_log_dir=str(tmp_path / "data"),
        backend_config=None,
        project_path=str(tmp_path),
        is_remote=False,
        mode="local",
    )

    captured = capsys.readouterr()
    assert "no existing directories found" in captured.out.lower()


def test_fresh_start_v2_aborts_on_non_yes(tmp_path, capsys):
    from chester.run_exp import _fresh_start_v2

    # Create an existing variant dir
    var_dir = tmp_path / "data" / "train" / "myexp" / "1_myexp_lr_0.001"
    var_dir.mkdir(parents=True)
    (var_dir / "model.pt").write_bytes(b"x" * 100)

    with patch("builtins.input", return_value="no"):
        with pytest.raises(SystemExit) as exc_info:
            _fresh_start_v2(
                exp_prefix="myexp",
                sub_dir="train",
                cfg_log_dir=str(tmp_path / "data"),
                backend_config=None,
                project_path=str(tmp_path),
                is_remote=False,
                mode="local",
            )
    assert exc_info.value.code == 0
    # Directory should NOT have been deleted
    assert var_dir.exists()


def test_fresh_start_v2_deletes_local_on_yes(tmp_path):
    from chester.run_exp import _fresh_start_v2

    var_dir = tmp_path / "data" / "train" / "myexp" / "1_myexp_lr_0.001"
    var_dir.mkdir(parents=True)
    (var_dir / "model.pt").write_bytes(b"x" * 100)

    with patch("builtins.input", return_value="yes"):
        _fresh_start_v2(
            exp_prefix="myexp",
            sub_dir="train",
            cfg_log_dir=str(tmp_path / "data"),
            backend_config=None,
            project_path=str(tmp_path),
            is_remote=False,
            mode="local",
        )

    assert not var_dir.exists()


def test_fresh_start_v2_shows_table_with_pt_count(tmp_path, capsys):
    from chester.run_exp import _fresh_start_v2

    var_dir = tmp_path / "data" / "train" / "myexp" / "1_myexp_lr_0.001"
    var_dir.mkdir(parents=True)
    (var_dir / "model.pt").write_bytes(b"x" * 1024)
    (var_dir / "final.pth").write_bytes(b"x" * 2048)

    with patch("builtins.input", return_value="yes"):
        _fresh_start_v2(
            exp_prefix="myexp",
            sub_dir="train",
            cfg_log_dir=str(tmp_path / "data"),
            backend_config=None,
            project_path=str(tmp_path),
            is_remote=False,
            mode="local",
        )

    captured = capsys.readouterr()
    assert "1_myexp_lr_0.001" in captured.out
    assert "2 pt" in captured.out


def test_fresh_start_v2_remote_scans_and_deletes(tmp_path, capsys):
    from chester.run_exp import _fresh_start_v2
    from chester.backends.base import BackendConfig

    # Create local variant dir
    var_dir = tmp_path / "data" / "train" / "myexp" / "1_myexp"
    var_dir.mkdir(parents=True)
    (var_dir / "model.pt").write_bytes(b"x" * 512)

    backend_config = BackendConfig(
        name="gl", type="slurm",
        host="gl",
        remote_dir="/remote/project",
    )

    fake_remote_rows = [
        {"name": "1_myexp", "exists": True, "size_str": "4.1G", "pt_count": 89},
    ]

    with patch("chester.run_exp._scan_remote_batch_dir", return_value=fake_remote_rows), \
         patch("chester.run_exp.subprocess.run") as mock_subproc, \
         patch("builtins.input", return_value="yes"):
        _fresh_start_v2(
            exp_prefix="myexp",
            sub_dir="train",
            cfg_log_dir=str(tmp_path / "data"),
            backend_config=backend_config,
            project_path=str(tmp_path),
            is_remote=True,
            mode="gl",
        )

    captured = capsys.readouterr()
    assert "4.1G" in captured.out
    assert "89 pt" in captured.out
    # ssh rm -rf should have been called
    rm_calls = [c for c in mock_subproc.call_args_list
                if c.args and 'ssh' in str(c.args[0]) and 'rm' in str(c.args[0])]
    assert len(rm_calls) == 1


# ---------------------------------------------------------------------------
# Integration tests: run_experiment_lite fresh= parameter
# ---------------------------------------------------------------------------

def test_run_experiment_lite_has_fresh_param():
    """run_experiment_lite must expose fresh=False in its signature."""
    import inspect
    from chester.run_exp import run_experiment_lite
    sig = inspect.signature(run_experiment_lite)
    assert 'fresh' in sig.parameters
    assert sig.parameters['fresh'].default is False


def test_run_experiment_lite_fresh_deletes_existing_dir(tmp_path):
    """Integration: fresh=True via run_experiment_lite deletes existing variant dirs."""
    import os
    from chester.run_exp import run_experiment_lite

    # Write a minimal .chester/config.yaml so load_config() finds a local backend
    chester_dir = tmp_path / ".chester"
    chester_dir.mkdir()
    (chester_dir / "config.yaml").write_text(
        f"log_dir: {tmp_path / 'data'}\n"
        "package_manager: python\n"
        "backends:\n"
        "  local:\n"
        "    type: local\n"
    )

    # Pre-create an "existing" variant directory that fresh=True should delete
    existing_var = tmp_path / "data" / "train" / "myexp" / "1_myexp_lr_0.001"
    existing_var.mkdir(parents=True)
    (existing_var / "model.pt").write_bytes(b"x" * 100)

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        os.environ["CHESTER_CONFIG_PATH"] = str(chester_dir / "config.yaml")
        with patch("builtins.input", return_value="yes"):
            run_experiment_lite(
                stub_method_call=lambda variant, log_dir, exp_name: None,
                variant={"chester_first_variant": True, "chester_last_variant": True},
                variations=[],
                mode="local",
                exp_prefix="myexp",
                fresh=True,
                dry=True,
                git_snapshot=False,
            )
    finally:
        os.chdir(old_cwd)
        os.environ.pop("CHESTER_CONFIG_PATH", None)

    # The pre-existing directory must have been deleted by fresh=True
    assert not existing_var.exists()
