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
