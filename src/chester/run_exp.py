import os
from pathlib import Path
import subprocess
import base64
import os.path as osp
import pickle as pickle
import cloudpickle
import numpy as np
import inspect
import collections
import glob
import sys
import time
import datetime
import dateutil.tz
import json
from . import config


# Auto-pull manifest management
_auto_pull_manifest_path = None
_auto_pull_jobs = []

# Deprecated modes that are no longer supported
_DEPRECATED_MODES = frozenset({
    "ec2", "autobot", "singularity", "local_singularity",
})


def _map_local_to_remote_log_dir(local_log_dir: str, mode: str) -> str:
    """
    Map a local log directory to its remote equivalent.

    The user specifies a local log_dir, and chester automatically maps it to
    the equivalent path on remote based on project roots.

    Example:
        Local PROJECT_PATH: /home/user/project
        Local log_dir: /home/user/project/data/train/exp1
        Remote remote_dir: /home/remote/project
        -> Remote log_dir: /home/remote/project/data/train/exp1

    Args:
        local_log_dir: Local log directory (absolute or relative to PROJECT_PATH)
        mode: Execution mode (e.g., 'armfranka', 'gl')

    Returns:
        Remote log directory path

    Raises:
        ValueError: If local_log_dir is not within PROJECT_PATH
    """
    project_path = config.PROJECT_PATH
    remote_dir = config.REMOTE_DIR.get(mode)

    if not remote_dir:
        raise ValueError(f"No remote_dir configured for mode '{mode}'")

    # Resolve local_log_dir to absolute path
    if not os.path.isabs(local_log_dir):
        local_log_dir = os.path.join(project_path, local_log_dir)
    local_log_dir = os.path.normpath(local_log_dir)

    # Ensure local_log_dir is within PROJECT_PATH
    project_path_normalized = os.path.normpath(project_path)
    if not local_log_dir.startswith(project_path_normalized + os.sep) and local_log_dir != project_path_normalized:
        raise ValueError(
            f"log_dir must be within PROJECT_PATH for remote sync.\n"
            f"  log_dir: {local_log_dir}\n"
            f"  PROJECT_PATH: {project_path_normalized}\n"
            f"Log directory must be a subdirectory of the project root."
        )

    # Compute relative path from PROJECT_PATH
    relative_path = os.path.relpath(local_log_dir, project_path_normalized)

    # Map to remote
    remote_log_dir = os.path.join(remote_dir, relative_path)
    return remote_log_dir


def _resolve_extra_pull_dirs(extra_pull_dirs: list, mode: str) -> list:
    """
    Resolve extra_pull_dirs to (local, remote) path pairs.

    Relative paths are resolved against PROJECT_PATH (local) and REMOTE_DIR (remote).
    Absolute paths (starting with '/') are used as-is on both local and remote.

    Args:
        extra_pull_dirs: List of directory paths (strings)
        mode: Execution mode (e.g., 'armfranka', 'gl')

    Returns:
        List of dicts with 'local' and 'remote' keys
    """
    if not extra_pull_dirs:
        return []

    result = []
    remote_dir = config.REMOTE_DIR.get(mode, '')

    for path in extra_pull_dirs:
        if os.path.isabs(path):
            # Absolute path: same on both local and remote
            result.append({'local': path, 'remote': path})
        else:
            # Relative path: resolve against project roots
            local_path = os.path.join(config.PROJECT_PATH, path)
            remote_path = os.path.join(remote_dir, path)
            result.append({'local': local_path, 'remote': remote_path})

    return result


def _resolve_extra_pull_dirs_v2(extra_pull_dirs, project_path, remote_dir):
    """Resolve extra_pull_dirs using the new config system (no old config module).

    Args:
        extra_pull_dirs: List of directory paths (strings)
        project_path: Local project root path
        remote_dir: Remote project root path

    Returns:
        List of dicts with 'local' and 'remote' keys
    """
    if not extra_pull_dirs:
        return []

    result = []
    for path in extra_pull_dirs:
        if os.path.isabs(path):
            result.append({'local': path, 'remote': path})
        else:
            local_path = os.path.join(project_path, path)
            remote_path = os.path.join(remote_dir, path)
            result.append({'local': local_path, 'remote': remote_path})
    return result


def _map_local_to_remote_log_dir_v2(local_log_dir, project_path, remote_dir):
    """Map local log_dir to remote using the new config system.

    Args:
        local_log_dir: Local log directory (absolute path, already resolved)
        project_path: Local project root
        remote_dir: Remote project root

    Returns:
        Remote log directory path

    Raises:
        ValueError: If local_log_dir is not within project_path
    """
    local_log_dir = os.path.normpath(local_log_dir)
    project_path = os.path.normpath(project_path)

    if not local_log_dir.startswith(project_path + os.sep) and local_log_dir != project_path:
        raise ValueError(
            f"log_dir must be within project_path for remote sync.\n"
            f"  log_dir: {local_log_dir}\n"
            f"  project_path: {project_path}\n"
        )

    relative_path = os.path.relpath(local_log_dir, project_path)
    return os.path.join(remote_dir, relative_path)


def _init_auto_pull_manifest(exp_prefix: str, mode: str):
    """Initialize the manifest file path for this batch of experiments."""
    global _auto_pull_manifest_path, _auto_pull_jobs
    manifest_dir = os.path.join(config.LOG_DIR, '.chester_manifests')
    os.makedirs(manifest_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    _auto_pull_manifest_path = os.path.join(manifest_dir, f'{exp_prefix}_{mode}_{timestamp}.json')
    _auto_pull_jobs = []


def _init_auto_pull_manifest_v2(exp_prefix, mode, log_dir):
    """Initialize auto-pull manifest using the new config system."""
    global _auto_pull_manifest_path, _auto_pull_jobs
    manifest_dir = os.path.join(log_dir, '.chester_manifests')
    os.makedirs(manifest_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M%S')
    _auto_pull_manifest_path = os.path.join(manifest_dir, f'{exp_prefix}_{mode}_{timestamp}.json')
    _auto_pull_jobs = []


def _add_job_to_manifest(host: str, remote_log_dir: str, local_log_dir: str,
                         exp_name: str, extra_pull_dirs: list = None):
    """Add a job to the auto-pull manifest."""
    global _auto_pull_jobs
    _auto_pull_jobs.append({
        'host': host,
        'remote_log_dir': remote_log_dir,
        'local_log_dir': local_log_dir,
        'exp_name': exp_name,
        'extra_pull_dirs': extra_pull_dirs or [],
        'pid_file': os.path.join(remote_log_dir, '.chester_pid'),
        'status': 'pending',
        'submitted_at': datetime.datetime.now().isoformat()
    })


def _save_and_spawn_auto_pull(dry: bool = False, poll_interval: int = 60):
    """Save the manifest and spawn the auto-pull poller."""
    global _auto_pull_manifest_path, _auto_pull_jobs

    if not _auto_pull_jobs:
        print('[chester] No jobs to track for auto-pull')
        return

    assert _auto_pull_manifest_path is not None, "chester auto_pull: manifest path not initialized"

    # Save manifest
    with open(_auto_pull_manifest_path, 'w') as f:
        json.dump(_auto_pull_jobs, f, indent=2)
    print(f'[chester] Saved auto-pull manifest: {_auto_pull_manifest_path}')
    print(f'[chester] Tracking {len(_auto_pull_jobs)} jobs for auto-pull')

    if dry:
        print('[chester] Dry run - not spawning auto-pull poller')
        return

    # Spawn background poller
    log_file = _auto_pull_manifest_path.replace('.json', '.log')

    # Run as module to support relative imports
    cmd = (f'nohup python -m chester.auto_pull '
           f'--manifest {_auto_pull_manifest_path} '
           f'--poll-interval {poll_interval} '
           f'> {log_file} 2>&1 &')

    print(f'[chester] Spawning auto-pull poller: {cmd}')
    os.system(cmd)
    print(f'[chester] Auto-pull log: {log_file}')


def monitor_processes(active_processes, max_processes=2, sleep_time=1):
    """
    Monitor the number of running processes and wait if maximum is reached.

    Args:
        process_list: List of subprocess.Popen objects
        max_processes: Maximum number of concurrent processes
        sleep_time: Time to sleep between checks in seconds

    Returns:
        Updated list with only running processes
    """
    # Wait if we've reached the maximum number of processes
    while len(active_processes) >= max_processes:
        time.sleep(sleep_time)
        # Update the list of active processes
        active_processes = [p for p in active_processes if p.poll() is None]

    return active_processes

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")





class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class VariantDict(AttrDict):
    def __init__(self, d, hidden_keys):
        super(VariantDict, self).__init__(d)
        self._hidden_keys = hidden_keys

    def dump(self):
        return {k: v for k, v in self.items() if k not in self._hidden_keys}


class VariantGenerator(dict):
    """
    Usage:

    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", ['x', 'y'])
    vg.variants() => # all combinations of [1,2,3] x ['x','y']

    Supports noncyclic dependency among parameters:
    vg = VariantGenerator()
    vg.add("param1", [1, 2, 3])
    vg.add("param2", lambda param1: [param1+1, param1+2])
    vg.variants() => # ..
    """

    def __init__(self):
        self._variants = []
        self._populate_variants()
        self._hidden_keys = []
        for k, vs, cfg in self._variants:
            if cfg.get("hide", False):
                self._hidden_keys.append(k)

    @property
    def size(self):
        return len(self.variants())

    def __getitem__(self, item):
        for param in self.variations():
            if param[0] == item:
                return param[1]

    def add(self, key, vals, **kwargs):
        self._variants.append((key, vals, kwargs))

    def update(self, key, vals, **kwargs):
        for i, (k, _, _) in enumerate(self._variants):
            if k == key:
                self._variants[i] = (k, vals, kwargs)
                return
        self.add(key, vals, kwargs)

    def _populate_variants(self):
        methods = inspect.getmembers(
            self.__class__, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
        methods = [x[1].__get__(self, self.__class__)
                   for x in methods if getattr(x[1], '__is_variant', False)]
        for m in methods:
            self.add(m.__name__, m, **getattr(m, "__variant_config", dict()))

    def variations(self):
        ret = []
        for key, vals, _ in self._variants:
            if not isinstance(vals, list):
                continue
            if len(vals) > 1:
                ret.append(key)
        return ret

    def variants(self, randomized=False):
        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        # ret = list(map(self.variant_dict, ret))
        ret = list(map(AttrDict, ret))
        ret[0]['chester_first_variant'] = True
        ret[-1]['chester_last_variant'] = True
        return ret

    def variant_dict(self, variant):
        return VariantDict(variant, self._hidden_keys)

    def to_name_suffix(self, variant):
        suffix = []
        for k, vs, cfg in self._variants:
            if not cfg.get("hide", False):
                suffix.append(k + "_" + str(variant[k]))
        return "_".join(suffix)

    def ivariants(self):
        dependencies = list()
        for key, vals, cfg in self._variants:
            if cfg.get("hide", False):
                continue  # Skip hidden keys entirely
            if hasattr(vals, "__call__"):
                args = inspect.getfullargspec(vals).args
                if hasattr(vals, 'im_self') or hasattr(vals, "__self__"):
                    # remove the first 'self' parameter
                    args = args[1:]
                dependencies.append((key, set(args)))
            else:
                dependencies.append((key, set()))
        sorted_keys = []
        # topo sort all nodes
        while len(sorted_keys) < len(self._variants):
            # get all nodes with zero in-degree
            free_nodes = [k for k, v in dependencies if len(v) == 0]
            if len(free_nodes) == 0:
                error_msg = "Invalid parameter dependency: \n"
                for k, v in dependencies:
                    if len(v) > 0:
                        error_msg += k + " depends on " + " & ".join(v) + "\n"
                raise ValueError(error_msg)
            dependencies = [(k, v)
                            for k, v in dependencies if k not in free_nodes]
            # remove the free nodes from the remaining dependencies
            for _, v in dependencies:
                v.difference_update(free_nodes)
            sorted_keys += free_nodes
        return self._ivariants_sorted(sorted_keys)

    def _ivariants_sorted(self, sorted_keys):
        if len(sorted_keys) == 0:
            yield dict()
        else:
            first_keys = sorted_keys[:-1]
            first_variants = self._ivariants_sorted(first_keys)
            last_key = sorted_keys[-1]
            last_vals = [v for k, v, _ in self._variants if k == last_key][0]
            if hasattr(last_vals, "__call__"):
                last_val_keys = inspect.getfullargspec(last_vals).args
                if hasattr(last_vals, 'im_self') or hasattr(last_vals, '__self__'):
                    last_val_keys = last_val_keys[1:]
            else:
                last_val_keys = None
            for variant in first_variants:
                if hasattr(last_vals, "__call__"):
                    last_variants = last_vals(
                        **{k: variant[k] for k in last_val_keys})
                    for last_choice in last_variants:
                        yield AttrDict(variant, **{last_key: last_choice})
                else:
                    for last_choice in last_vals:
                        yield AttrDict(variant, **{last_key: last_choice})


def variant(*args, **kwargs):
    def _variant(fn):
        fn.__is_variant = True
        fn.__variant_config = kwargs
        return fn

    if len(args) == 1 and isinstance(args[0], collections.Callable):
        return _variant(args[0])
    return _variant


def rsync_code(remote_host, remote_dir):
    """
    Sync project code to remote host.

    Syncs from PROJECT_PATH to remote_dir.
    Requires rsync_include and rsync_exclude lists in chester.yaml.
    """
    project_path = config.PROJECT_PATH
    print(f'Ready to rsync code: {project_path} -> {remote_host}:{remote_dir}')

    yaml_include = config.RSYNC_INCLUDE
    yaml_exclude = config.RSYNC_EXCLUDE

    if not yaml_include and not yaml_exclude:
        raise ValueError("rsync_include and rsync_exclude must be defined in chester.yaml")

    include_args = ' '.join(f"--include='{p}'" for p in yaml_include)
    exclude_args = ' '.join(f"--exclude='{p}'" for p in yaml_exclude)
    cmd = f"rsync -avzh --delete {include_args} {exclude_args} {project_path}/ {remote_host}:{remote_dir}"
    print(cmd)
    os.system(cmd)


def rsync_code_v2(remote_host, remote_dir, project_path, rsync_include, rsync_exclude):
    """Sync project code to remote host using the new config system.

    Args:
        remote_host: SSH host identifier
        remote_dir: Remote directory to sync to
        project_path: Local project root path
        rsync_include: List of include patterns
        rsync_exclude: List of exclude patterns
    """
    print(f'[chester] rsync code: {project_path} -> {remote_host}:{remote_dir}')

    if not rsync_include and not rsync_exclude:
        print('[chester] Warning: no rsync_include/rsync_exclude defined, syncing entire project')

    cmd = ["rsync", "-avzh", "--delete"]
    for p in (rsync_include or []):
        cmd.append(f"--include={p}")
    for p in (rsync_exclude or []):
        cmd.append(f"--exclude={p}")
    cmd.append(f"{project_path}/")
    cmd.append(f"{remote_host}:{remote_dir}")
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


exp_count = -2
sub_process_popens = []
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%m_%d_%H_%M')
remote_confirmed = False


def run_experiment_lite(
        stub_method_call=None,
        batch_tasks=None,
        exp_prefix="experiment",
        exp_name=None,
        log_dir=None,
        sub_dir='train',
        script=None,
        python_command="python",
        mode="local",
        use_gpu=False,
        dry=False,
        env=None,
        variant=None,
        variations=None,
        use_cloudpickle=True,
        pre_commands=None,
        print_command=True,
        launch_with_subprocess=True,
        wait_subprocess=True,
        max_num_processes=10,
        use_singularity=None,
        slurm_overrides=None,
        hydra_enabled=False,
        hydra_flags=None,
        auto_pull=False,
        auto_pull_interval=60,
        extra_pull_dirs=None,
        sync_env=None,
        git_snapshot=True,
        **kwargs):
    """
    Serialize the stubbed method call and run the experiment using the
    specified backend.

    Uses the new backend system (.chester/config.yaml) for dispatch instead
    of hardcoded mode lists.

    Args:
        stub_method_call: A stubbed method call to serialize and run.
        batch_tasks: Pre-built list of tasks (advanced usage).
        exp_prefix: Name prefix for the experiments.
        exp_name: Explicit experiment name (auto-generated if None).
        log_dir: Log directory (auto-generated if None).
        sub_dir: Subdirectory under log_dir for organization.
        script: Python script to run (defaults to chester.run_exp_worker).
        python_command: Base python command.
        mode: Backend name from .chester/config.yaml (or "local" default).
        use_gpu: Whether to request GPU resources.
        dry: If True, print commands without executing.
        env: Extra environment variables (dict).
        variant: Dictionary of variant parameters.
        variations: List of variant keys used in experiment naming.
        use_cloudpickle: Use cloudpickle for serialization.
        pre_commands: Pre-execution commands.
        print_command: Print the generated command/script.
        launch_with_subprocess: Launch via subprocess (local only).
        wait_subprocess: Wait for subprocess to complete (local only).
        max_num_processes: Max concurrent local processes.
        use_singularity: Override singularity setting (None=use backend default).
        slurm_overrides: Dict of per-experiment SLURM parameter overrides.
        hydra_enabled: Use Hydra command line format.
        hydra_flags: Additional Hydra flags.
        auto_pull: Enable automatic result pulling from remote.
        auto_pull_interval: Poll interval in seconds for auto-pull.
        extra_pull_dirs: Extra directories to pull from remote.
        sync_env: Override sync_on_launch config.
        git_snapshot: Save git state to log_dir before running (default True).
        **kwargs: Additional parameters passed to the python script.
    """
    # Fix mutable defaults
    if env is None:
        env = {}
    if variations is None:
        variations = []

    # ----------------------------------------------------------------
    # 1. Reject deprecated modes
    # ----------------------------------------------------------------
    if mode in _DEPRECATED_MODES:
        raise ValueError(
            f"Mode '{mode}' is deprecated. Use backend names from "
            f".chester/config.yaml instead. "
            f"For singularity, configure it on the backend."
        )

    # ----------------------------------------------------------------
    # 2. Load config and create backend
    # ----------------------------------------------------------------
    from chester.config_v2 import load_config, get_backend
    from chester.backends import create_backend

    cfg = load_config()
    backend_config = get_backend(mode, cfg)
    backend = create_backend(backend_config, cfg)

    project_path = cfg["project_path"]
    cfg_log_dir = cfg["log_dir"]
    is_remote = backend_config.type in ("ssh", "slurm")

    # ----------------------------------------------------------------
    # 3. Handle singularity override
    # ----------------------------------------------------------------
    if use_singularity is False:
        backend.config.singularity = None
    elif use_singularity is True and not backend.config.singularity:
        raise ValueError(
            f"use_singularity=True but backend '{mode}' has no singularity config"
        )

    # ----------------------------------------------------------------
    # 4. Variant bookkeeping
    # ----------------------------------------------------------------
    last_variant = variant.pop('chester_last_variant', False)
    first_variant = variant.pop('chester_first_variant', False)

    local_batch_dir = os.path.join(cfg_log_dir, sub_dir, exp_prefix)

    # ----------------------------------------------------------------
    # 5. Task preparation (same logic as original)
    # ----------------------------------------------------------------
    assert stub_method_call is not None or batch_tasks is not None or script is not None, \
        "Must provide at least either stub_method_call or batch_tasks or script"
    if script is None:
        script = '-m chester.run_exp_worker'  # Use module syntax for installed package

    if batch_tasks is None:
        batch_tasks = [
            dict(
                kwargs,
                pre_commands=pre_commands,
                stub_method_call=stub_method_call,
                exp_name=exp_name,
                log_dir=log_dir,
                env=env,
                variant=variant,
                use_cloudpickle=use_cloudpickle
            )
        ]

    global exp_count
    global remote_confirmed
    global sub_process_popens
    sub_process_popens = monitor_processes(sub_process_popens, max_num_processes)

    for task in batch_tasks:
        call = task.pop("stub_method_call")
        if use_cloudpickle:
            data = base64.b64encode(cloudpickle.dumps(call)).decode("utf-8")
        else:
            data = base64.b64encode(pickle.dumps(call)).decode("utf-8")
        task["args_data"] = data
        exp_count += 1

        if task.get("exp_name", None) is None:
            exp_name = exp_prefix
            for v in variations:
                print(v)
                key_name = v.split('.')[-1]
                if isinstance(variant[v], (list, tuple)):
                    continue
                if isinstance(variant[v], str):
                    exp_name += '_{}'.format(variant[v].split('/')[-1])
                elif isinstance(variant[v], bool):
                    if variant[v]:
                        exp_name += '_{}'.format(key_name)
                elif variant[v] is not None:  # int or float
                    exp_name += '_{}_{:g}'.format(key_name, variant[v])
            ind = len(glob.glob(os.path.join(local_batch_dir, '[0-9]*_*')))
            if exp_count == -1:
                exp_count = ind + 1
            task["exp_name"] = "{}_{}".format(exp_count, exp_name)
            print('exp name ', task["exp_name"])

        # Handle log_dir: user specifies local path, chester maps to remote
        local_log_dir = task.get("log_dir", None)
        if local_log_dir is None:
            local_log_dir = os.path.join(cfg_log_dir, sub_dir, exp_prefix, task["exp_name"])
        elif not os.path.isabs(local_log_dir):
            local_log_dir = os.path.join(project_path, local_log_dir)
        local_log_dir = os.path.normpath(local_log_dir)
        task['_local_log_dir'] = local_log_dir

        # For remote backends, map local to remote; for local, use as-is
        if is_remote:
            task['log_dir'] = _map_local_to_remote_log_dir_v2(
                local_log_dir, project_path, backend_config.remote_dir
            )
        else:
            task['log_dir'] = local_log_dir

        if task.get("variant", None) is not None:
            variant = task.pop("variant")
            if "exp_name" not in variant:
                variant["exp_name"] = task["exp_name"]
            task["variant_data"] = base64.b64encode(pickle.dumps(variant)).decode("utf-8")
        elif "variant" in task:
            del task["variant"]
        task["env"] = task.get("env", dict()) or dict()

    # ----------------------------------------------------------------
    # 6. Git snapshot (before any execution)
    # ----------------------------------------------------------------
    if git_snapshot and first_variant:
        try:
            from chester.git_snapshot import save_git_snapshot
            first_log_dir = batch_tasks[0].get('_local_log_dir', local_batch_dir)
            os.makedirs(first_log_dir, exist_ok=True)
            save_git_snapshot(first_log_dir, repo_path=project_path)
        except Exception as e:
            print(f'[chester] Warning: git snapshot failed: {e}')

    # ----------------------------------------------------------------
    # 7. Confirm for remote (non-dry) runs
    # ----------------------------------------------------------------
    if is_remote and not remote_confirmed and not dry:
        remote_confirmed = query_yes_no(
            "Running in (non-dry) mode %s. Confirm?" % mode)
        if not remote_confirmed:
            sys.exit(1)

    # ----------------------------------------------------------------
    # 8. Rsync code for remote backends (first variant only)
    # ----------------------------------------------------------------
    if is_remote and first_variant and not dry:
        rsync_code_v2(
            remote_host=backend_config.host,
            remote_dir=backend_config.remote_dir,
            project_path=project_path,
            rsync_include=cfg.get("rsync_include", []),
            rsync_exclude=cfg.get("rsync_exclude", []),
        )

    # ----------------------------------------------------------------
    # 9. Initialize auto-pull manifest (first variant only)
    # ----------------------------------------------------------------
    if is_remote and first_variant and auto_pull:
        _init_auto_pull_manifest_v2(exp_prefix, mode, cfg_log_dir)

    # ----------------------------------------------------------------
    # 10. Generate script and submit via backend
    # ----------------------------------------------------------------
    for task in batch_tasks:
        task_env = task.pop("env", None)
        local_log_dir = task.pop('_local_log_dir')
        remote_log_dir = task.get('log_dir', '')

        # Merge env: caller env + task env
        merged_env = {}
        if env:
            merged_env.update(env)
        if task_env:
            merged_env.update(task_env)

        # Build task dict for backend (params sub-dict)
        backend_task = {
            "params": {k: v for k, v in task.items()},
            "exp_name": task.get("exp_name", exp_prefix),
            "_local_log_dir": local_log_dir,
        }

        # Dispatch to backend
        if backend_config.type == "local":
            # Local backend: generate command and run directly
            command = backend.generate_command(
                backend_task,
                script=script,
                python_command=python_command,
                env=merged_env or None,
            )

            if print_command:
                print(command)
            if dry:
                return

            if launch_with_subprocess:
                try:
                    run_env = dict(os.environ, **(merged_env or {}))
                    if wait_subprocess:
                        subprocess.call(command, shell=True, env=run_env)
                        popen_obj = None
                    else:
                        popen_obj = subprocess.Popen(command, shell=True, env=run_env)
                    sub_process_popens.append(popen_obj)
                except Exception as e:
                    print(e)
                    if isinstance(e, KeyboardInterrupt):
                        raise
            else:
                # For hydra debug mode
                from .hydra_utils import run_hydra_command
                assert hydra_enabled, "hydra_enabled must be True when launch_with_subprocess is False"
                run_hydra_command(command, task["log_dir"], stub_method_call)
                popen_obj = None

            return popen_obj

        else:
            # Remote backends (ssh, slurm) — unified dispatch
            gen_kwargs = dict(
                script=script,
                python_command=python_command,
                env=merged_env or None,
            )
            if backend_config.type == "slurm" and slurm_overrides:
                gen_kwargs["slurm_overrides"] = slurm_overrides

            script_content = backend.generate_script(backend_task, **gen_kwargs)

            if print_command:
                print(script_content)

            if not dry:
                # Save script locally for debugging
                local_exp_dir = os.path.join(local_batch_dir, task.get("exp_name", ""))
                os.makedirs(local_exp_dir, exist_ok=True)
                script_name = "slurm_launch.sh" if backend_config.type == "slurm" else "ssh_launch.sh"
                with open(os.path.join(local_exp_dir, script_name), 'w') as f:
                    f.write(script_content)

            # Add job to auto-pull manifest
            if auto_pull and not dry:
                _add_job_to_manifest(
                    host=backend_config.host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task.get('exp_name', ''),
                    extra_pull_dirs=_resolve_extra_pull_dirs_v2(
                        extra_pull_dirs, project_path, backend_config.remote_dir
                    ),
                )

            # Submit via backend (backend.submit handles dry=True internally)
            backend.submit(backend_task, script_content, dry=dry)

    # ----------------------------------------------------------------
    # 11. Spawn auto-pull poller (last variant only)
    # ----------------------------------------------------------------
    if is_remote and last_variant and auto_pull:
        _save_and_spawn_auto_pull(dry=dry, poll_interval=auto_pull_interval)
