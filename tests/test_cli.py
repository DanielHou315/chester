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

    def test_pull_subcommand_exists(self):
        from chester.cli import build_parser
        p = build_parser()
        args = p.parse_args(['pull', 'gl', 'some/folder'])
        assert args.command == 'pull'
        assert args.host == 'gl'
        assert args.folder == 'some/folder'


class TestCmdPull:
    def test_dry_run_prints_rsync_and_no_subprocess(self, capsys):
        """In dry mode, prints rsync command and does not call subprocess.run."""
        import chester.config as real_config_module
        from chester.cli import cmd_pull
        with mock.patch('chester.cli.subprocess.run') as mock_run, \
             mock.patch.object(real_config_module, 'PROJECT_PATH', '/home/user/project'), \
             mock.patch.object(real_config_module, 'REMOTE_DIR', {'gl': '/remote/gl'}):
            cmd_pull(host='gl', folder='myexp', bare=False, dry=True)
        mock_run.assert_not_called()
        out = capsys.readouterr().out
        assert 'rsync' in out
        assert 'gl:' in out

    def test_bare_includes_before_excludes(self):
        """bare=True: --include patterns appear before --exclude patterns."""
        import chester.config as real_config_module
        from chester.cli import cmd_pull
        captured = []
        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return mock.Mock(returncode=0)
        with mock.patch('chester.cli.subprocess.run', side_effect=fake_run), \
             mock.patch.object(real_config_module, 'PROJECT_PATH', '/home/user/project'), \
             mock.patch.object(real_config_module, 'REMOTE_DIR', {'gl': '/remote/gl'}):
            cmd_pull(host='gl', folder='myexp', bare=True, dry=False)
        include_pos = next(i for i, x in enumerate(captured) if x == '--include')
        exclude_pos = next(i for i, x in enumerate(captured) if x == '--exclude')
        assert include_pos < exclude_pos, "All --include rules must precede --exclude rules"

    def test_trailing_slashes(self):
        """Both rsync source and destination must have trailing slashes."""
        import chester.config as real_config_module
        from chester.cli import cmd_pull
        captured = []
        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return mock.Mock(returncode=0)
        with mock.patch('chester.cli.subprocess.run', side_effect=fake_run), \
             mock.patch.object(real_config_module, 'PROJECT_PATH', '/home/user/project'), \
             mock.patch.object(real_config_module, 'REMOTE_DIR', {'gl': '/remote/gl'}):
            cmd_pull(host='gl', folder='myexp', bare=False, dry=False)
        src = next(x for x in captured if x.startswith('gl:'))
        dst = next(x for x in captured if x.startswith('/home/user/project'))
        assert src.endswith('/'), f"rsync source must end with /: {src}"
        assert dst.endswith('/'), f"rsync dest must end with /: {dst}"

    def test_uses_project_path_not_cwd(self):
        """local_dir is rooted at config.PROJECT_PATH, not './data'."""
        import chester.config as real_config_module
        from chester.cli import cmd_pull
        captured = []
        def fake_run(cmd, **kwargs):
            captured.extend(cmd)
            return mock.Mock(returncode=0)
        with mock.patch('chester.cli.subprocess.run', side_effect=fake_run), \
             mock.patch.object(real_config_module, 'PROJECT_PATH', '/home/user/project'), \
             mock.patch.object(real_config_module, 'REMOTE_DIR', {'gl': '/remote/gl'}):
            cmd_pull(host='gl', folder='myexp', bare=False, dry=False)
        dst = next(x for x in captured if '/home/user/project' in x)
        assert dst.startswith('/home/user/project'), f"dst should be rooted at PROJECT_PATH: {dst}"
