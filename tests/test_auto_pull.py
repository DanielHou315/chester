# tests/test_auto_pull.py
import subprocess
import unittest.mock as mock

import pytest

from chester.auto_pull import check_slurm_job_status, check_job_status


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


class TestCheckJobStatusIntegration:
    """Integration tests for check_job_status with SLURM support."""

    def _slurm_job(self, **overrides):
        job = {
            'host': 'gl',
            'remote_log_dir': '/remote/logs/exp1',
            'local_log_dir': '/local/logs/exp1',
            'exp_name': 'test',
            'slurm_job_id': 99999,
        }
        job.update(overrides)
        return job

    def _ssh_job(self, **overrides):
        job = {
            'host': 'myserver',
            'remote_log_dir': '/remote/logs/exp1',
            'local_log_dir': '/local/logs/exp1',
            'exp_name': 'test',
        }
        job.update(overrides)
        return job

    def test_check_job_status_slurm_failed_detected(self):
        """No .done + sacct FAILED -> 'failed'."""
        job = self._slurm_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
             mock.patch("chester.auto_pull.check_slurm_job_status", return_value="failed"):
            assert check_job_status(job) == 'failed'

    def test_check_job_status_slurm_running(self):
        """No .done + sacct RUNNING -> 'running'."""
        job = self._slurm_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
             mock.patch("chester.auto_pull.check_slurm_job_status", return_value="running"):
            assert check_job_status(job) == 'running'

    def test_check_job_status_slurm_done_marker_present(self):
        """.done exists -> 'done' (SLURM job)."""
        job = self._slurm_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=True):
            assert check_job_status(job) == 'done'

    def test_check_job_status_slurm_completed_no_done_yet(self):
        """sacct COMPLETED but no .done -> 'running' (still waiting for marker)."""
        job = self._slurm_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
             mock.patch("chester.auto_pull.check_slurm_job_status", return_value="completed"):
            assert check_job_status(job) == 'running'

    def test_check_job_status_ssh_still_uses_pid(self):
        """SSH job (no slurm_job_id) uses PID tracking."""
        job = self._ssh_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
             mock.patch("chester.auto_pull.get_remote_pid", return_value=12345), \
             mock.patch("chester.auto_pull.check_process_running", return_value=False):
            assert check_job_status(job) == 'failed'

    def test_check_job_status_slurm_sacct_unknown_fallback(self):
        """sacct 'unknown' -> 'running' (conservative fallback)."""
        job = self._slurm_job()
        with mock.patch("chester.auto_pull.check_done_marker", return_value=False), \
             mock.patch("chester.auto_pull.check_slurm_job_status", return_value="unknown"):
            assert check_job_status(job) == 'running'
