# tests/test_auto_pull.py
import subprocess
import unittest.mock as mock

import pytest

from chester.auto_pull import check_slurm_job_status


def _mock_sacct(stdout, returncode=0, side_effect=None):
    """Helper to mock subprocess.run for sacct calls."""
    if side_effect:
        return mock.patch(
            "chester.auto_pull.subprocess.run", side_effect=side_effect
        )
    return mock.patch(
        "chester.auto_pull.subprocess.run",
        return_value=mock.Mock(returncode=returncode, stdout=stdout, stderr=""),
    )


class TestCheckSlurmJobStatus:
    def test_completed(self):
        with _mock_sacct("COMPLETED\n"):
            assert check_slurm_job_status("gl", 12345) == "completed"

    def test_failed(self):
        with _mock_sacct("FAILED\n"):
            assert check_slurm_job_status("gl", 12345) == "failed"

    def test_timeout(self):
        with _mock_sacct("TIMEOUT\n"):
            assert check_slurm_job_status("gl", 12345) == "failed"

    def test_out_of_memory(self):
        with _mock_sacct("OUT_OF_MEMORY\n"):
            assert check_slurm_job_status("gl", 12345) == "failed"

    def test_cancelled_with_uid_suffix(self):
        with _mock_sacct("CANCELLED by 12345\n"):
            assert check_slurm_job_status("gl", 12345) == "failed"

    def test_running(self):
        with _mock_sacct("RUNNING\n"):
            assert check_slurm_job_status("gl", 12345) == "running"

    def test_pending(self):
        with _mock_sacct("PENDING\n"):
            assert check_slurm_job_status("gl", 12345) == "running"

    def test_ssh_timeout(self):
        with _mock_sacct(None, side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=30)):
            assert check_slurm_job_status("gl", 12345) == "unknown"

    def test_empty_output(self):
        with _mock_sacct(""):
            assert check_slurm_job_status("gl", 12345) == "unknown"

    def test_multi_line_uses_first_line(self):
        with _mock_sacct("COMPLETED\nCOMPLETED\nCOMPLETED\n"):
            assert check_slurm_job_status("gl", 12345) == "completed"
