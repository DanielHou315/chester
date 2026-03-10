"""
Chester job store — persistent per-job files for auto-pull tracking.

Each remote job submitted with auto_pull=True writes one JSON file:
    {job_store_dir}/{uuid}.json

The file is deleted after a successful pull.
Failed jobs are marked in-place (status='failed') but never pulled.
"""
import json
import uuid
from pathlib import Path
from typing import Optional

JOB_STATUS_PENDING = 'pending'
JOB_STATUS_FAILED = 'failed'


def get_default_job_store_dir() -> Path:
    """Return {PROJECT_PATH}/.chester/auto_pull_jobs/."""
    from chester import config
    return Path(config.PROJECT_PATH) / '.chester' / 'auto_pull_jobs'


def write_job_file(job_store_dir: Path, job: dict) -> str:
    """
    Write a job dict to a new UUID-named file in job_store_dir.

    A 'job_id' key is added automatically. Returns the job_id (UUID string).
    """
    job_store_dir = Path(job_store_dir)
    job_store_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    data = dict(job)
    data['job_id'] = job_id
    data['status'] = JOB_STATUS_PENDING
    (job_store_dir / f'{job_id}.json').write_text(json.dumps(data, indent=2))
    return job_id


def load_pending_jobs(job_store_dir: Path, prefix: Optional[str] = None) -> list:
    """
    Load all jobs with status='pending' from job_store_dir.

    Args:
        job_store_dir: Directory containing per-job JSON files.
        prefix: If given, only return jobs where exp_prefix == prefix.
    """
    job_store_dir = Path(job_store_dir)
    if not job_store_dir.exists():
        return []
    jobs = []
    for path in sorted(job_store_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get('status') != JOB_STATUS_PENDING:
            continue
        if prefix is not None and data.get('exp_prefix') != prefix:
            continue
        jobs.append(data)
    return jobs


def mark_job_failed(job_store_dir: Path, job_id: str):
    """Update job file status to 'failed' in-place. No pull will be attempted.

    Note: the read-then-write is not atomic. If the process crashes mid-write the
    file may be truncated and silently skipped by the loader. This is acceptable
    (the job is lost, not corrupted) but worth being aware of.
    """
    path = Path(job_store_dir) / f'{job_id}.json'
    if not path.exists():
        return
    data = json.loads(path.read_text())
    data['status'] = JOB_STATUS_FAILED
    path.write_text(json.dumps(data, indent=2))


def delete_job_file(job_store_dir: Path, job_id: str):
    """Delete a job file. No-op if already gone."""
    path = Path(job_store_dir) / f'{job_id}.json'
    if path.exists():
        path.unlink()
