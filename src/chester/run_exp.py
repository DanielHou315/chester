import os
from pathlib import Path
import re
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
from chester import config, config_ec2
from chester.hydra_utils import to_hydra_command, run_hydra_command
from chester.slurm import to_slurm_command, to_local_command, to_ssh_command
from chester.utils_s3 import launch_ec2, s3_sync_code


# Auto-pull manifest management
_auto_pull_manifest_path = None
_auto_pull_jobs = []


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


def _init_auto_pull_manifest(exp_prefix: str, mode: str):
    """Initialize the manifest file path for this batch of experiments."""
    global _auto_pull_manifest_path, _auto_pull_jobs
    manifest_dir = os.path.join(config.LOG_DIR, '.chester_manifests')
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
    chester_dir = os.path.dirname(os.path.abspath(__file__))
    auto_pull_script = os.path.join(chester_dir, 'auto_pull.py')
    log_dir = os.path.join(config.LOG_DIR, '.chester_manifests')
    log_file = _auto_pull_manifest_path.replace('.json', '.log')

    cmd = (f'nohup python {auto_pull_script} '
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


_find_unsafe = re.compile(r'[a-zA-Z0-9_^@%+=:,./-]').search


def _shellquote(s):
    """Return a shell-escaped version of the string *s*."""
    if not s:
        return "''"

    if _find_unsafe(s) is None:
        return s

    # use single quotes, and put single quotes into double quotes
    # the string $'b is then quoted as '$'"'"'b'

    return "'" + s.replace("'", "'\"'\"'") + "'"


def _to_param_val(v):
    if v is None:
        return ""
    elif isinstance(v, list):
        return " ".join(map(_shellquote, list(map(str, v))))
    else:
        return _shellquote(str(v))




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
        script=None,  # TODO: change this before making pip package
        python_command="python",
        mode="local",
        use_gpu=False,
        dry=False,
        env={},
        variant=None,
        variations=[],
        use_cloudpickle=True,
        pre_commands=None,
        print_command=True,
        launch_with_subprocess=True,
        wait_subprocess=True,
        max_num_processes=10,
        compile_script=None,
        wait_compile=None,
        use_singularity=False,
        hydra_enabled=False,  # New parameter for Hydra support
        hydra_flags=None,  # Additional hydra flags like multirun
        auto_pull=False,  # Enable automatic result pulling from remote
        auto_pull_interval=60,  # Poll interval in seconds for auto-pull
        sync_env=None,  # Override sync_on_launch config (None = use config)
        extra_pull_dirs=None,  # List of extra directories to pull (relative to PROJECT_PATH or absolute)
        **kwargs):
    """
    Serialize the stubbed method call and run the experiment using the specified mode.
    :param stub_method_call: A stubbed method call.
    :param script: The name of the entrance point python script
    :param mode: Where & how to run the experiment. Can be ['local', 'singularity', 'seuss', 'psc']
    :param dry: Whether to do a dry-run, which only prints the commands without executing them.
    :param exp_prefix: Name prefix for the experiments
    :param env: extra environment variables
    :param kwargs: All other parameters will be passed directly to the entrance python script.
    :param variant: If provided, should be a dictionary of parameters
    :param hydra_enabled: If True, will use Hydra command line override format instead of chester format
    :param hydra_flags: Optional dictionary of hydra flags to add to the command (e.g., {'multirun': True})
    """
    last_variant = variant.pop('chester_last_variant', False)
    first_variant = variant.pop('chester_first_variant', False)
    host = config.HOST_ADDRESS[mode]
    local_chester_queue_dir = config.LOG_DIR + '/queues'
    remote_sub_dir = os.path.join(config.REMOTE_LOG_DIR[mode], sub_dir)
    remote_batch_dir = os.path.join(remote_sub_dir, exp_prefix)
    local_batch_dir = os.path.join(config.LOG_DIR, sub_dir, exp_prefix)

    if first_variant:
        os.system(f'mkdir -p {local_chester_queue_dir}')
    if mode == 'singularity':
        mode = 'local_singularity'

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
        # Store original local log_dir for auto-pull, compute remote log_dir for execution
        local_log_dir = task.get("log_dir", None)
        if local_log_dir is None:
            # Default: data/{sub_dir}/{exp_prefix}/{exp_name}
            local_log_dir = os.path.join(config.LOG_DIR, sub_dir, exp_prefix, task["exp_name"])
        elif not os.path.isabs(local_log_dir):
            # Relative path - resolve relative to PROJECT_PATH
            local_log_dir = os.path.join(config.PROJECT_PATH, local_log_dir)
        local_log_dir = os.path.normpath(local_log_dir)
        task['_local_log_dir'] = local_log_dir  # Store for auto-pull

        # For remote modes, map local to remote; for local mode, use as-is
        if mode in ["local", "local_singularity"]:
            task['log_dir'] = local_log_dir
        else:
            task['log_dir'] = _map_local_to_remote_log_dir(local_log_dir, mode)

        if task.get("variant", None) is not None:
            variant = task.pop("variant")
            if "exp_name" not in variant:
                variant["exp_name"] = task["exp_name"]
                # variant["group_name"] = exp_prefix
            task["variant_data"] = base64.b64encode(pickle.dumps(variant)).decode("utf-8")
        elif "variant" in task:
            del task["variant"]
        task["env"] = task.get("env", dict()) or dict()
        local_exp_dir = os.path.join(local_batch_dir, task["exp_name"])

    if mode not in ["local", "local_singularity", "ec2"] and not remote_confirmed and not dry:
        remote_confirmed = query_yes_no(
            "Running in (non-dry) mode %s. Confirm?" % mode)
        if not remote_confirmed:
            sys.exit(1)

    if mode in ["local"]:
        for task in batch_tasks:
            env = task.pop("env", None)
            task.pop('_local_log_dir', None)  # Pop internal field, not for CLI
            # Generate command based on whether hydra is enabled or not
            if hydra_enabled:
                command = to_hydra_command(
                    params=task,
                    python_command=python_command,
                    script=script,
                    hydra_flags=hydra_flags,
                    env=env
                )
            else:
                command = to_local_command(
                    task,
                    python_command=python_command,
                    script=script,
                    env=env
                )
                
            if print_command:
                print(command)
            if dry:
                return
            
            if env is None:
                env = dict()
            if launch_with_subprocess:
                try:
                    if wait_subprocess:
                        subprocess.call(command, shell=True, env=dict(os.environ, **env))
                        popen_obj = None
                    else:
                        popen_obj = subprocess.Popen(command, shell=True, env=dict(os.environ, **env))
                    sub_process_popens.append(popen_obj)
                except Exception as e:
                    print(e)
                    if isinstance(e, KeyboardInterrupt):
                        raise
            else: # for debug, not need to catch subprocess
                assert hydra_enabled, "hydra_enabled must be True when launch_with_subprocess is False"
                run_hydra_command(command, task["log_dir"], stub_method_call)
                popen_obj = None

            return popen_obj
    elif mode == 'local_singularity':
        for task in batch_tasks:
            env = task.pop("env", None)
            task.pop('_local_log_dir', None)  # Pop internal field, not for CLI
            command = to_local_command(
                task,
                python_command=python_command,
                script=osp.join(config.PROJECT_PATH, script)
            )
            if print_command:
                print(command)
            if dry:
                return
            try:
                if env is None:
                    env = dict()
                # TODO add argument for specifying container
                singularity_header = f'singularity exec {config.SIMG_PATH[mode]}'
                command = singularity_header + ' ' + command
                subprocess.call(
                    command, shell=True, env=dict(os.environ, **env))
                popen_obj = None
            except Exception as e:
                print(e)
                if isinstance(e, KeyboardInterrupt):
                    raise
            return popen_obj
    elif mode in ['gl', 'seuss', 'psc', 'satori']:
        for task in batch_tasks:
            remote_dir = config.REMOTE_DIR[mode]
            simg_dir = config.SIMG_PATH[mode]
            if first_variant and not dry:
                rsync_code(remote_host=host, remote_dir=remote_dir)
            if first_variant and auto_pull:
                _init_auto_pull_manifest(exp_prefix, mode)
            remote_log_dir = task['log_dir']
            local_log_dir = task.pop('_local_log_dir')  # Pop internal field, not for CLI
            header = config.REMOTE_HEADER[mode]
            header = header + "\n#SBATCH -o " + os.path.join(remote_log_dir, 'slurm.out') + " # STDOUT"
            header = header + "\n#SBATCH -e " + os.path.join(remote_log_dir, 'slurm.err') + " # STDERR\n"
            gpus = str(variant.get("gpus", 1))
            header = header.replace("$gpus", gpus)
            python_command = "srun " + python_command
            command_list = to_slurm_command(
                task,
                use_gpu=use_gpu,
                modules=config.MODULES[mode],
                cuda_module=config.CUDA_MODULE[mode],
                header=header,
                python_command=python_command,
                script=script,
                use_singularity=use_singularity,
                simg_dir=simg_dir,
                remote_dir=remote_dir,
                mount_options=config.REMOTE_MOUNT_OPTION[mode],
                compile_script=compile_script,
                wait_compile=wait_compile,
                hydra_enabled=hydra_enabled,
                hydra_flags=hydra_flags,
                set_egl_gpu=False,
                env=env
            )
            if print_command:
                print("; ".join(command_list))
            command = "\n".join(command_list)
            os.system(f'mkdir -p {local_exp_dir}')
            local_script_name = os.path.join(local_exp_dir, "slurm_launch")
            with open(local_script_name, 'w') as f:
                f.write(command)

            # Add job to auto-pull manifest
            if auto_pull:
                _add_job_to_manifest(
                    host=host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task['exp_name'],
                    extra_pull_dirs=_resolve_extra_pull_dirs(extra_pull_dirs, mode)
                )

            if last_variant:
                print('Remote sub dir: ', remote_sub_dir)
                os.system('scp -r {f1} {host}:{f2}'.format(f1=local_batch_dir, f2=remote_sub_dir, host=mode))
                os.system(f'rm -rf {local_batch_dir}')
                print('Ready to execute the scheduler')
                remote_cmd = (f'cd {remote_dir} && . ./prepare.sh && '
                              f'python chester/scheduler/remote_slurm_launcher.py {remote_batch_dir} {int(dry)}')
                cmd = "ssh  {host} \'{cmd} \'".format(host=host,
                                                      cmd=remote_cmd
                                                      )
                print('Submit to slurm ', cmd)
                os.system(cmd)  # Launch

                # Save manifest and spawn auto-pull poller
                if auto_pull:
                    _save_and_spawn_auto_pull(dry=dry, poll_interval=auto_pull_interval)
            # Cleanup
    elif mode in ['autobot']:
        for task in batch_tasks:
            # TODO check remote directory
            remote_dir = config.REMOTE_DIR[mode]
            simg_dir = config.SIMG_PATH[mode]

            # query_yes_no('Confirm: Syncing code to {}:{}'.format(mode, remote_dir))
            if first_variant:
                rsync_code(remote_host=mode, remote_dir=remote_dir)
            if first_variant and auto_pull:
                _init_auto_pull_manifest(exp_prefix, mode)
            remote_log_dir = task['log_dir']
            local_log_dir = task.pop('_local_log_dir')  # Pop internal field, not for CLI
            remote_script_name = os.path.join(remote_log_dir, task['exp_name'])
            local_script_name = os.path.join(local_exp_dir, task['exp_name'])
            header = '#CHESTERNODE ' + ','.join(config.AUTOBOT_NODELIST)
            header = header + "\n#CHESTEROUT " + os.path.join(remote_log_dir, 'slurm.out')
            header = header + "\n#CHESTERERR " + os.path.join(remote_log_dir, 'slurm.err')
            header = header + "\n#CHESTERSCRIPT " + remote_script_name
            if simg_dir.find('$') == -1:
                simg_dir = osp.join(remote_dir, simg_dir)
            command_list = to_slurm_command(
                task,
                use_gpu=use_gpu,
                modules=config.MODULES[mode],
                cuda_module=config.CUDA_MODULE[mode],
                header=header,
                python_command=python_command,
                use_singularity=use_singularity,
                script=osp.join(remote_dir, script),
                simg_dir=simg_dir,
                remote_dir=remote_dir,
                mount_options=config.REMOTE_MOUNT_OPTION[mode],
                compile_script=compile_script,
                wait_compile=wait_compile,
                set_egl_gpu=True,
            )
            if print_command:
                print("; ".join(command_list))
            command = "\n".join(command_list)
            script_name = './data/tmp/' + task['exp_name']
            scheduler_script_name = os.path.join(local_chester_queue_dir, task['exp_name'])
            with open(script_name, 'w') as f:
                f.write(command)
            os.system(f'cp {script_name} {local_script_name}')
            os.system(f'cp {script_name} {scheduler_script_name}')

            # Add job to auto-pull manifest
            if auto_pull:
                _add_job_to_manifest(
                    host=mode,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task['exp_name'],
                    extra_pull_dirs=_resolve_extra_pull_dirs(extra_pull_dirs, mode)
                )

            # Cleanup
            os.remove(script_name)
            # Open scheduler if all jobs have been submitted
            # Remote end will only open another scheduler when there is not one running already
            # Redirect the output of the remote scheduler to the log file
            if last_variant and not dry:
                print('Syncing to remote')
                print('Remote batch dir: ', remote_batch_dir)
                os.system('scp -r {f1} {host}:{f2}'.format(f1=local_batch_dir, f2=remote_batch_dir, host=mode))
                os.system('scp -r {f1} {host}:{f2}'.format(f1=local_chester_queue_dir, f2=config.CHESTER_QUEUE_DIR,
                                                           host=mode))
                os.system(f'rm -rf {local_batch_dir}')
                os.system(f'rm -rf {local_chester_queue_dir}')
                # os.system("ssh {host} \'{cmd}\'".format(host=mode, cmd='mkdir -p ' + config.CHESTER_CHEDULER_LOG_DIR))
                t = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%m_%d_%H_%M')
                # log_file = os.path.join(config.CHESTER_CHEDULER_LOG_DIR, f'{t}.txt')
                log_file = os.path.join(config.CHESTER_CHEDULER_LOG_DIR, f'log_{t}.txt')
                print('Ready to execute the scheduler')
                cmd = "ssh  {host} \'{cmd} > {output}&\'".format(host=mode,
                                                                 cmd=f'cd {remote_dir} && . ./prepare.sh && nohup python chester/scheduler/remote_scheduler.py',
                                                                 output=log_file)
                if dry:
                    print(remote_script_name)
                    print(cmd)
                else:
                    print(cmd)
                    os.system(cmd)

                # Save manifest and spawn auto-pull poller
                if auto_pull:
                    _save_and_spawn_auto_pull(dry=dry, poll_interval=auto_pull_interval)

    elif mode in config.SSH_HOSTS:
        # SSH-based remote execution (no SLURM)
        # Package manager (uv/conda) determined by chester.yaml config
        for task in batch_tasks:
            remote_dir = config.REMOTE_DIR[mode]

            if first_variant and not dry:
                rsync_code(remote_host=host, remote_dir=remote_dir)
            if first_variant and auto_pull:
                _init_auto_pull_manifest(exp_prefix, mode)

            remote_log_dir = task['log_dir']
            local_log_dir = task.pop('_local_log_dir')  # Pop internal field, not for CLI

            # Pop env from task dict to pass separately to to_ssh_command
            task_env = task.pop("env", None)
            if task_env:
                # Merge with any env passed to run_experiment_lite
                merged_env = {**env, **task_env} if env else task_env
            else:
                merged_env = env

            # Generate SSH command script (python command wrapped by slurm.py based on config)
            command_list = to_ssh_command(
                task,
                python_command=python_command,
                remote_dir=remote_dir,
                script=script,
                env=merged_env,
                hydra_enabled=hydra_enabled,
                hydra_flags=hydra_flags,
                sync_env=sync_env,
            )

            if print_command:
                print("; ".join(command_list))

            command = "\n".join(command_list)

            # Save script locally
            os.system(f'mkdir -p {local_exp_dir}')
            local_script_name = os.path.join(local_exp_dir, "ssh_launch.sh")
            with open(local_script_name, 'w') as f:
                f.write(command)

            # Add job to auto-pull manifest
            if auto_pull:
                _add_job_to_manifest(
                    host=host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task['exp_name'],
                    extra_pull_dirs=_resolve_extra_pull_dirs(extra_pull_dirs, mode)
                )

            if not dry:
                # Copy script to remote
                remote_script_path = os.path.join(remote_log_dir, "ssh_launch.sh")
                os.system(f'ssh {host} "mkdir -p {remote_log_dir}"')
                os.system(f'scp {local_script_name} {host}:{remote_script_path}')

                # Execute via SSH with nohup, save PID for tracking
                stdout_log = os.path.join(remote_log_dir, 'stdout.log')
                stderr_log = os.path.join(remote_log_dir, 'stderr.log')
                pid_file = os.path.join(remote_log_dir, '.chester_pid')
                ssh_cmd = (f'ssh {host} "cd {remote_dir} && '
                           f'nohup bash {remote_script_path} > {stdout_log} 2> {stderr_log} & '
                           f'echo \\$! > {pid_file}"')

                print(f'[chester] Launching on {host}: {task["exp_name"]}')
                print(f'[chester] Remote log dir: {remote_log_dir}')
                os.system(ssh_cmd)

        # Save manifest and spawn auto-pull poller (after all jobs submitted)
        if last_variant and auto_pull:
            _save_and_spawn_auto_pull(dry=dry, poll_interval=auto_pull_interval)

    elif mode == 'ec2':
        # if docker_image is None:
        #     docker_image = config.DOCKER_IMAGE
        s3_code_path = s3_sync_code(config_ec2, dry=dry)
        for task in batch_tasks:
            task["remote_log_dir"] = osp.join(config_ec2.AWS_S3_PATH, exp_prefix.replace("_", "-"), task["exp_name"])
            if compile_script is None:
                task["pre_commands"] = [". ./prepare_ec2.sh", 'time ./compile.sh']
            else:
                task["pre_commands"] = [". ./prepare_ec2.sh", 'time ./' + compile_script]
        launch_ec2(batch_tasks,
                   exp_prefix=exp_prefix,
                   docker_image=None,  # Currently not using docker
                   python_command=python_command,
                   script=script,
                   aws_config=None,
                   dry=dry,
                   terminate_machine=True,
                   use_gpu=use_gpu,
                   code_full_path=s3_code_path,
                   sync_s3_pkl=True,
                   sync_s3_html=True,
                   sync_s3_png=True,
                   sync_s3_log=True,
                   sync_s3_gif=True,
                   sync_s3_mp4=True,
                   sync_s3_pth=True,
                   sync_s3_txt=True,
                   sync_log_on_termination=True,
                   periodic_sync=True,
                   periodic_sync_interval=15)
