# third_party/chester/tests/test_monitor.py
from unittest.mock import patch

import pytest

from chester.auto_pull import monitor_jobs
from chester.job_store import write_job_file, JOB_STATUS_PENDING


def _make_job(tmp_path, exp_name="test_job", host="gl", slurm_job_id=None):
    job = {
        "host": host,
        "remote_log_dir": f"/remote/data/{exp_name}",
        "local_log_dir": str(tmp_path / exp_name),
        "exp_name": exp_name,
        "exp_prefix": "test",
        "extra_pull_dirs": [],
        "status": JOB_STATUS_PENDING,
    }
    if slurm_job_id is not None:
        job["slurm_job_id"] = slurm_job_id
    job_id = write_job_file(tmp_path, job)
    return job_id


def test_monitor_jobs_returns_when_all_done(tmp_path):
    """monitor_jobs() returns after all jobs reach terminal state."""
    job_id = _make_job(tmp_path)

    with patch("chester.auto_pull.execute_pull_for_job", return_value="pulled") as mock_pull, \
         patch("chester.job_store.delete_job_file") as mock_del, \
         patch("chester.auto_pull.time.sleep"):
        monitor_jobs([job_id], tmp_path, poll_interval=0)

    mock_pull.assert_called_once()
    mock_del.assert_called_once_with(tmp_path, job_id)


def test_monitor_jobs_stays_pending_while_running(tmp_path):
    """monitor_jobs() keeps looping while execute_pull_for_job returns 'running'."""
    job_id = _make_job(tmp_path)
    call_count = {"n": 0}

    def fake_execute(job, bare=False):
        call_count["n"] += 1
        return "pulled" if call_count["n"] >= 3 else "running"

    with patch("chester.auto_pull.execute_pull_for_job", side_effect=fake_execute), \
         patch("chester.job_store.delete_job_file"), \
         patch("chester.auto_pull.time.sleep"):
        monitor_jobs([job_id], tmp_path, poll_interval=0)

    assert call_count["n"] == 3


def test_monitor_jobs_handles_failed(tmp_path):
    """monitor_jobs() treats 'failed' as terminal — does not hang."""
    job_id = _make_job(tmp_path)

    with patch("chester.auto_pull.execute_pull_for_job", return_value="failed"), \
         patch("chester.job_store.mark_job_failed") as mock_fail, \
         patch("chester.auto_pull.time.sleep"):
        monitor_jobs([job_id], tmp_path, poll_interval=0)

    mock_fail.assert_called_once_with(tmp_path, job_id)


def test_monitor_jobs_handles_done_orphans(tmp_path):
    """execute_pull_for_job returning 'pulled' (from done_orphans) terminates correctly."""
    job_id = _make_job(tmp_path)

    with patch("chester.auto_pull.execute_pull_for_job", return_value="pulled") as mock_pull, \
         patch("chester.job_store.delete_job_file") as mock_del, \
         patch("chester.auto_pull.time.sleep"):
        monitor_jobs([job_id], tmp_path, poll_interval=0)

    mock_pull.assert_called_once()
    mock_del.assert_called_once_with(tmp_path, job_id)
