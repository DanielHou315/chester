# tests/test_job_store.py
import json
import pytest
from pathlib import Path
from chester.job_store import write_job_file, load_pending_jobs, delete_job_file, mark_job_failed, JOB_STATUS_PENDING, JOB_STATUS_FAILED


def test_write_job_file_creates_file(tmp_path):
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/logs/exp1',
        'local_log_dir': '/local/logs/exp1',
        'exp_name': 'test_exp',
        'exp_prefix': 'my_prefix',
        'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 1
    assert job_id in files[0].name


def test_write_job_file_content(tmp_path):
    job = {
        'host': 'gl',
        'remote_log_dir': '/remote/logs/exp1',
        'local_log_dir': '/local/logs/exp1',
        'exp_name': 'test_exp',
        'exp_prefix': 'my_prefix',
        'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    file_path = tmp_path / f'{job_id}.json'
    data = json.loads(file_path.read_text())
    assert data['host'] == 'gl'
    assert data['job_id'] == job_id
    assert data['status'] == JOB_STATUS_PENDING


def test_load_pending_jobs_all(tmp_path):
    for i in range(3):
        write_job_file(tmp_path, {
            'host': 'gl', 'remote_log_dir': f'/r/{i}',
            'local_log_dir': f'/l/{i}', 'exp_name': f'exp{i}',
            'exp_prefix': 'prefix_a', 'status': JOB_STATUS_PENDING,
        })
    jobs = load_pending_jobs(tmp_path)
    assert len(jobs) == 3


def test_load_pending_jobs_prefix_filter(tmp_path):
    write_job_file(tmp_path, {
        'host': 'gl', 'remote_log_dir': '/r/1', 'local_log_dir': '/l/1',
        'exp_name': 'exp1', 'exp_prefix': 'alpha', 'status': JOB_STATUS_PENDING,
    })
    write_job_file(tmp_path, {
        'host': 'gl', 'remote_log_dir': '/r/2', 'local_log_dir': '/l/2',
        'exp_name': 'exp2', 'exp_prefix': 'beta', 'status': JOB_STATUS_PENDING,
    })
    jobs = load_pending_jobs(tmp_path, prefix='alpha')
    assert len(jobs) == 1
    assert jobs[0]['exp_prefix'] == 'alpha'


def test_load_pending_jobs_skips_non_pending(tmp_path):
    import uuid
    job_id = str(uuid.uuid4())
    path = tmp_path / f'{job_id}.json'
    path.write_text(json.dumps({'status': 'failed', 'job_id': job_id, 'exp_prefix': 'x'}))
    jobs = load_pending_jobs(tmp_path)
    assert len(jobs) == 0


def test_delete_job_file(tmp_path):
    job = {
        'host': 'gl', 'remote_log_dir': '/r/1', 'local_log_dir': '/l/1',
        'exp_name': 'exp1', 'exp_prefix': 'pfx', 'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    delete_job_file(tmp_path, job_id)
    assert not (tmp_path / f'{job_id}.json').exists()


def test_delete_job_file_missing_is_noop(tmp_path):
    delete_job_file(tmp_path, 'nonexistent-uuid')  # should not raise


def test_load_pending_jobs_empty_dir(tmp_path):
    jobs = load_pending_jobs(tmp_path)
    assert jobs == []


def test_mark_job_failed(tmp_path):
    job = {
        'host': 'gl', 'remote_log_dir': '/r/1', 'local_log_dir': '/l/1',
        'exp_name': 'exp1', 'exp_prefix': 'pfx', 'status': JOB_STATUS_PENDING,
    }
    job_id = write_job_file(tmp_path, job)
    mark_job_failed(tmp_path, job_id)
    file_path = tmp_path / f'{job_id}.json'
    assert file_path.exists()
    import json
    data = json.loads(file_path.read_text())
    assert data['status'] == JOB_STATUS_FAILED
    pending = load_pending_jobs(tmp_path)
    assert all(j['job_id'] != job_id for j in pending)
