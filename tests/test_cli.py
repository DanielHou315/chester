# tests/test_cli.py
import json
import pytest
from unittest import mock
from pathlib import Path


class TestPullRemoteCommand:

    def _pending_job(self, exp_prefix='pfx'):
        return {
            'host': 'gl',
            'remote_log_dir': '/remote/logs/exp1',
            'local_log_dir': '/local/logs/exp1',
            'exp_name': 'test_exp',
            'exp_prefix': exp_prefix,
            'extra_pull_dirs': [],
            'status': 'pending',
        }

    def test_no_jobs(self, tmp_path, capsys):
        from chester.cli import cmd_pull_remote
        cmd_pull_remote(job_store_dir=tmp_path, prefix=None, bare=False, dry_run=False)
        assert 'No pending jobs' in capsys.readouterr().out

    def test_deletes_file_on_pulled(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='pulled'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        assert not (tmp_path / f'{job_id}.json').exists()

    def test_marks_failed_file_no_delete(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file, load_pending_jobs
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='failed'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        path = tmp_path / f'{job_id}.json'
        assert path.exists()
        assert json.loads(path.read_text())['status'] == 'failed'
        assert load_pending_jobs(tmp_path) == []

    def test_keeps_file_on_running(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='running'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        assert (tmp_path / f'{job_id}.json').exists()

    def test_keeps_file_on_pull_failed(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='pull_failed'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        assert (tmp_path / f'{job_id}.json').exists()

    def test_prefix_filter(self, tmp_path):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        write_job_file(tmp_path, self._pending_job(exp_prefix='alpha'))
        write_job_file(tmp_path, self._pending_job(exp_prefix='beta'))
        calls = []
        def fake_pull(job, bare=False):
            calls.append(job['exp_prefix'])
            return 'running'
        with mock.patch('chester.cli.execute_pull_for_job', side_effect=fake_pull):
            cmd_pull_remote(tmp_path, prefix='alpha', bare=False, dry_run=False)
        assert calls == ['alpha']

    def test_dry_run_no_changes(self, tmp_path, capsys):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        job_id = write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job') as mock_pull:
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=True)
        mock_pull.assert_not_called()
        assert (tmp_path / f'{job_id}.json').exists()
        assert 'dry-run' in capsys.readouterr().out.lower()

    def test_summary_printed(self, tmp_path, capsys):
        from chester.cli import cmd_pull_remote
        from chester.job_store import write_job_file
        write_job_file(tmp_path, self._pending_job())
        with mock.patch('chester.cli.execute_pull_for_job', return_value='pulled'):
            cmd_pull_remote(tmp_path, prefix=None, bare=False, dry_run=False)
        out = capsys.readouterr().out
        assert 'pulled' in out.lower()


class TestBuildParser:
    def test_pull_remote_subcommand_exists(self):
        from chester.cli import build_parser
        p = build_parser()
        args = p.parse_args(['pull-remote'])
        assert args.command == 'pull-remote'
        assert args.prefix is None
        assert args.bare is False
        assert args.dry_run is False

    def test_pull_remote_with_prefix(self):
        from chester.cli import build_parser
        p = build_parser()
        args = p.parse_args(['pull-remote', '--prefix', 'my_exp'])
        assert args.prefix == 'my_exp'

    def test_pull_subcommand_does_not_exist(self):
        from chester.cli import build_parser
        import pytest
        p = build_parser()
        with pytest.raises(SystemExit):
            p.parse_args(['pull', 'gl', 'some/folder'])