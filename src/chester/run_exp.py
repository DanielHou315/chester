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
import shlex
import secrets as _secrets
from . import config
from chester.config_v2 import load_config, get_backend
from chester.backends import create_backend
from chester.job_store import write_job_file, get_default_job_store_dir, JOB_STATUS_PENDING


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


def _register_job_for_pull(
    host: str,
    remote_log_dir: str,
    local_log_dir: str,
    exp_name: str,
    exp_prefix: str,
    extra_pull_dirs: list = None,
    slurm_job_id: int = None,
    submodule_commits: dict = None,
    submodule_worktrees: dict = None,
):
    """Write a single job file to the persistent job store."""
    job_store_dir = get_default_job_store_dir()
    job = {
        'host': host,
        'remote_log_dir': remote_log_dir,
        'local_log_dir': local_log_dir,
        'exp_name': exp_name,
        'exp_prefix': exp_prefix,
        'extra_pull_dirs': extra_pull_dirs or [],
        'status': JOB_STATUS_PENDING,
    }
    if slurm_job_id is not None:
        job['slurm_job_id'] = slurm_job_id
    if submodule_commits:
        job['submodule_commits'] = submodule_commits
    if submodule_worktrees:
        job['submodule_worktrees'] = submodule_worktrees
    job_id = write_job_file(job_store_dir, job)
    print(f'[chester] Registered job for pull: {exp_name} -> {job_store_dir}/{job_id}.json')


def _validate_submodule_commits(
    submodule_commits: dict,
    project_path: str,
) -> dict:
    """Validate submodule refs locally and resolve to full 40-char SHAs.

    Args:
        submodule_commits: {submodule_path: ref} — user-provided refs (may be
            short SHA, branch name, or tag).
        project_path: Absolute local path of the project root.

    Returns:
        {submodule_path: full_sha} with resolved 40-char SHAs.

    Raises:
        ValueError: If a submodule path does not exist or a ref cannot be
            resolved.
    """
    resolved = {}
    for sub_path, ref in submodule_commits.items():
        abs_sub = os.path.join(project_path, sub_path)
        if not os.path.isdir(abs_sub):
            raise ValueError(
                f"[chester] submodule_commits: path not found: '{abs_sub}'\n"
                f"  Key '{sub_path}' must be a directory under project_path."
            )
        try:
            full_sha = subprocess.check_output(
                ["git", "-C", abs_sub, "rev-parse", "--verify", f"{ref}^{{commit}}"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            raise ValueError(
                f"[chester] submodule_commits: cannot resolve ref '{ref}' "
                f"in submodule '{sub_path}'.\n"
                f"  Run: git -C {abs_sub} rev-parse {ref}"
            )
        resolved[sub_path] = full_sha
    return resolved


def _build_worktree_paths(
    resolved_commits: dict,
    remote_dir: str,
    timestamp: str,
) -> dict:
    """Compute unique remote worktree paths for each pinned submodule.

    Names follow the pattern:
        {submodule_path}/.worktrees/{timestamp}_{random6hex}_{short_sha}/

    The timestamp reflects submission time (not execution time). The random
    6-hex suffix ensures uniqueness across concurrent same-minute launches.

    Args:
        resolved_commits: {submodule_path: full_40char_sha}
        remote_dir: Absolute remote project root path.
        timestamp: Submission timestamp string (e.g. "03_23_10_00").

    Returns:
        {submodule_path: abs_remote_worktree_path}
    """
    result = {}
    for sub_path, full_sha in resolved_commits.items():
        rand = _secrets.token_hex(3)       # 6 hex chars
        short_sha = full_sha[:12]
        wt_name = f"{timestamp}_{rand}_{short_sha}"
        wt_path = os.path.join(remote_dir, sub_path, ".worktrees", wt_name)
        result[sub_path] = wt_path
    return result


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

def confirm_action(message: str, default: str = "yes", skip: bool = False) -> bool:
    """Prompt the user for yes/no confirmation.

    Args:
        message: The question to display.
        default: Presumed answer on empty input ("yes", "no", or None for required).
        skip: If True, return True without prompting.

    Returns:
        True if confirmed, False if denied.
    """
    if skip:
        return True

    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError(f"invalid default answer: '{default}'")

    while True:
        choice = input(message + prompt).lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def query_yes_no(question, default="yes"):
    """Deprecated: use confirm_action() instead."""
    import warnings
    warnings.warn("query_yes_no() is deprecated, use confirm_action()", DeprecationWarning, stacklevel=2)
    return confirm_action(question, default=default)





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
        self._derivations = []
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
        order = kwargs.get("order")
        if order is not None:
            if order not in ("serial", "dependent"):
                raise ValueError(
                    f"order='{order}' on '{key}' is not valid. "
                    f"Use order='serial' or order='dependent'."
                )
            if callable(vals) and not isinstance(vals, list):
                raise ValueError(
                    f"order='{order}' on '{key}' cannot be used with callable values. "
                    f"Provide a concrete list instead."
                )
            if isinstance(vals, list) and len(vals) < 2:
                raise ValueError(
                    f"order='{order}' on '{key}' requires at least 2 values, got {len(vals)}"
                )
            if order == "serial":
                existing = [k for k, _, c in self._variants if c.get("order") == "serial"]
                if existing:
                    raise ValueError(
                        f"order='serial' on '{key}' conflicts with existing serial "
                        f"key '{existing[0]}'. Only one serial key is supported."
                    )
        self._variants.append((key, vals, kwargs))

    def derive(self, key, fn):
        """Register a derived parameter computed from the full variant dict.

        Unlike ``vg.add(key, lambda dep: [...])``, the function receives the
        entire variant dict and returns a **single concrete value** (not a list).
        This is the right tool when a parameter is fully determined by other
        parameters rather than being an independent sweep axis.

        The primary use case is bypassing OmegaConf ``${eval:...}`` expressions
        that fail when nested — derived keys are passed as concrete Hydra CLI
        overrides, so the YAML expression is never evaluated::

            vg.add("experiment.training.env.num_train_sim", [127, 1])
            vg.derive(
                "experiment.training.env.num_train_real",
                lambda v: 128 - v["experiment.training.env.num_train_sim"],
            )
            vg.derive(
                "experiment.training.env.sim_fixed_asset_scale",
                lambda v: 2.0 if v["experiment.training.env.num_train_sim"]
                               > v["experiment.training.env.num_train_real"]
                          else 1.0,
            )

        Dotted keys (e.g. ``"experiment.training.env.num_train_sim"``) work
        naturally because ``fn`` indexes the dict with string keys rather than
        using Python attribute access.

        **Ordering**: derivations are applied in registration order after all
        base variants are expanded.  A derivation can safely reference any key
        set by ``vg.add()`` or by an *earlier* ``vg.derive()`` call.

        **Limitations**:

        - *No cycle detection*. Two patterns produce incorrect results silently
          or loudly:

          1. ``derive("A", fn_A)`` references key ``"B"`` before ``derive("B",
             fn_B)`` is registered → ``KeyError`` at ``vg.variants()`` time if
             ``"B"`` was not also set by ``vg.add()``.

          2. ``derive("A", ...)`` registered *after* ``derive("B", ...)`` where
             ``fn_B`` reads ``"A"`` → ``fn_B`` sees the *old* value of ``"A"``
             (from ``vg.add()``), then ``fn_A`` overwrites it with a new value.
             No error is raised; the final variant is silently inconsistent.

          Rule of thumb: register derivations in dependency order — if B
          depends on A, call ``vg.derive("A", ...)`` before
          ``vg.derive("B", ...)``.

        - *Derived keys do not appear in* ``vg.variations()``, so they are not
          factored into the experiment name hash. If two variants differ only in
          derived values they will get the same experiment directory name.
          Override ``exp_name`` explicitly in the launcher loop if this matters.
        """
        self._derivations.append((key, fn))

    def get_dependent_keys(self) -> list:
        """Return keys marked with order='dependent'."""
        return [k for k, _, cfg in self._variants if cfg.get("order") == "dependent"]

    def get_serial_keys(self) -> list:
        """Return keys marked with order='serial'."""
        return [k for k, _, cfg in self._variants if cfg.get("order") == "serial"]

    def get_dependency_map(self, variants: list) -> dict:
        """Compute inter-variant dependency map based on order='dependent' fields.

        For each dependent key, a variant's predecessor is the variant that is
        identical except the dependent field has the previous value in the list.

        Args:
            variants: List of variant dicts (from self.variants()).

        Returns:
            Dict mapping variant index -> list of predecessor variant indices.
            Variants with no predecessors are omitted.
        """
        dep_keys = self.get_dependent_keys()
        if not dep_keys:
            return {}

        # Build value ordering for each dependent key
        seq_val_order = {}
        seq_val_list = {}
        for key, vals, cfg in self._variants:
            if cfg.get("order") == "dependent" and isinstance(vals, list):
                seq_val_order[key] = {v: i for i, v in enumerate(vals)}
                seq_val_list[key] = vals

        # Index variants by their identity tuple (all field values)
        # Convert mutable values (lists) to tuples for hashability
        def _hashable(val):
            if isinstance(val, list):
                return tuple(val)
            return val

        all_keys = [k for k, _, _ in self._variants]
        variant_index = {}
        for i, v in enumerate(variants):
            identity = tuple(_hashable(v.get(k)) for k in all_keys)
            variant_index[identity] = i

        dep_map = {}
        for i, v in enumerate(variants):
            predecessors = []
            identity = list(_hashable(v.get(k)) for k in all_keys)

            for seq_key in dep_keys:
                val = v[seq_key]
                order = seq_val_order[seq_key]
                val_idx = order.get(val, 0)
                if val_idx == 0:
                    continue  # first value, no predecessor for this key

                # Find the previous value
                prev_val = seq_val_list[seq_key][val_idx - 1]

                # Build predecessor identity: same as current but with prev_val for this key
                key_pos = all_keys.index(seq_key)
                pred_identity = list(identity)
                pred_identity[key_pos] = prev_val
                pred_idx = variant_index.get(tuple(_hashable(x) for x in pred_identity))
                if pred_idx is not None:
                    predecessors.append(pred_idx)

            if predecessors:
                dep_map[i] = predecessors

        return dep_map

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
        for key, vals, cfg in self._variants:
            if not isinstance(vals, list):
                continue
            if cfg.get("order") == "serial":
                continue  # serial steps share a directory, don't affect exp_name
            if len(vals) > 1:
                ret.append(key)
        return ret

    def variants(self, randomized=False):
        dep_keys = self.get_dependent_keys()
        serial_keys = self.get_serial_keys()
        ordered_keys = dep_keys + serial_keys

        if ordered_keys and randomized:
            raise ValueError(
                "variants(randomized=True) cannot be used with ordered fields "
                "(order='serial' or order='dependent'). "
                "Ordered execution requires deterministic variant ordering."
            )

        ret = list(self.ivariants())
        if randomized:
            np.random.shuffle(ret)
        # ret = list(map(self.variant_dict, ret))
        ret = list(map(AttrDict, ret))
        for variant in ret:
            for key, fn in self._derivations:
                variant[key] = fn(variant)

        # --- order="dependent": inject identity + predecessor metadata ---
        if dep_keys:
            dep_map = self.get_dependency_map(ret)
            all_keys = [k for k, _, _ in self._variants]

            def _hashable(val):
                return tuple(val) if isinstance(val, list) else val

            for i, v in enumerate(ret):
                identity = tuple((k, _hashable(v.get(k))) for k in all_keys)
                v["_chester_seq_identity"] = identity
                v["_chester_pred_identities"] = [
                    tuple((k, _hashable(ret[j].get(k))) for k in all_keys)
                    for j in dep_map.get(i, [])
                ]

        # --- order="serial": collapse into single variants with serial steps ---
        if serial_keys:
            non_serial_keys = [k for k, _, cfg in self._variants
                               if cfg.get("order") != "serial"]

            def _hashable_s(val):
                return tuple(val) if isinstance(val, list) else val

            # Group variants that are identical except for serial fields
            groups = {}  # group_id -> [variant_indices] (in order)
            for i, v in enumerate(ret):
                group_id = tuple(_hashable_s(v.get(k)) for k in non_serial_keys)
                groups.setdefault(group_id, []).append(i)

            # Collapse each group into a single variant with _chester_serial_steps
            collapsed = []
            for group_id, indices in groups.items():
                base = ret[indices[0]]  # Use first variant as base
                serial_steps = []
                for sk in serial_keys:
                    steps = [ret[idx][sk] for idx in indices]
                    serial_steps.append((sk, steps))
                base["_chester_serial_steps"] = serial_steps
                collapsed.append(base)

            ret = collapsed

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
    cmd = f"rsync -avzhK --info=progress2 --delete {include_args} {exclude_args} {project_path}/ {remote_host}:{remote_dir}"
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

    cmd = ["rsync", "-avzhK", "--info=progress2", "--delete", "--skip-compress=sif/img/iso/gz/bz2/xz/zst/zip/7z"]
    for p in (rsync_include or []):
        cmd.append(f"--include={p}")
    for p in (rsync_exclude or []):
        cmd.append(f"--exclude={p}")
    cmd.append(f"{project_path}/")
    cmd.append(f"{remote_host}:{remote_dir}")
    print(' '.join(cmd))
    result = subprocess.run(cmd)
    # Exit code 24 = "some files vanished before transfer" — harmless race condition
    if result.returncode not in (0, 24):
        raise subprocess.CalledProcessError(result.returncode, cmd)


def _format_size(size_bytes: int) -> str:
    """Format a byte count as a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            if unit == 'B':
                return f"{size_bytes} B"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _scan_local_batch_dir(batch_dir: str) -> list:
    """List and scan all variant subdirectories under batch_dir.

    Returns a list of dicts sorted by name:
        [{'name': str, 'path': str, 'size_bytes': int, 'pt_count': int}]
    Returns [] if batch_dir does not exist.
    """
    if not os.path.isdir(batch_dir):
        return []

    rows = []
    for entry in sorted(os.listdir(batch_dir)):
        full = os.path.join(batch_dir, entry)
        if not os.path.isdir(full):
            continue
        size_bytes = 0
        pt_count = 0
        for dirpath, _dirnames, filenames in os.walk(full):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    size_bytes += os.path.getsize(fp)
                except OSError:
                    pass
                if fn.endswith('.pt') or fn.endswith('.pth'):
                    pt_count += 1
        rows.append({
            'name': entry,
            'path': full,
            'size_bytes': size_bytes,
            'pt_count': pt_count,
        })
    return rows


def _scan_remote_batch_dir(host: str, remote_batch_dir: str) -> list | None:
    """Scan all variant subdirs under remote_batch_dir via a single SSH call.

    Returns a list of dicts sorted by name:
        [{'name': str, 'exists': bool, 'size_str': str, 'pt_count': int}]
    Returns None if the SSH call fails.
    Returns [] if the remote directory does not exist or is empty.
    """
    script = (
        f"if [ -d {shlex.quote(remote_batch_dir)} ]; then "
        f"for dir in {shlex.quote(remote_batch_dir)}/*/; do "
        f"[ -d \"$dir\" ] || continue; "
        f"name=$(basename \"$dir\"); "
        f"size=$(du -sh \"$dir\" 2>/dev/null | cut -f1); "
        f"count=$(find \"$dir\" \\( -name \"*.pt\" -o -name \"*.pth\" \\) "
        f"2>/dev/null | wc -l | tr -d ' '); "
        f"echo \"$name|$size|$count\"; "
        f"done; fi"
    )
    try:
        result = subprocess.run(
            ['ssh', host, script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        rows = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split('|', 2)
            if len(parts) != 3:
                continue
            name, size_str, count_str = parts
            exists = size_str.strip() != '—'
            try:
                pt_count = int(count_str.strip())
            except ValueError:
                pt_count = 0
            rows.append({
                'name': name.strip(),
                'exists': exists,
                'size_str': size_str.strip() if exists else '—',
                'pt_count': pt_count,
            })
        return sorted(rows, key=lambda r: r['name'])
    except Exception:
        return None


def _fresh_start_v2(
    exp_prefix: str,
    sub_dir: str,
    cfg_log_dir: str,
    backend_config,
    project_path: str,
    is_remote: bool,
    mode: str,
) -> None:
    """Scan, display, confirm, and delete all variant dirs under exp_prefix.

    Called before any submission when fresh=True.  Exits the process if the
    user does not confirm.
    """
    import shutil as _shutil

    local_batch_dir = os.path.join(cfg_log_dir, sub_dir, exp_prefix)
    local_rows = _scan_local_batch_dir(local_batch_dir)

    # Remote scan
    remote_rows = []
    remote_scan_failed = False
    if is_remote:
        remote_batch_dir = _map_local_to_remote_log_dir_v2(
            local_batch_dir, project_path, backend_config.remote_dir
        )
        result = _scan_remote_batch_dir(backend_config.host, remote_batch_dir)
        if result is None:
            remote_scan_failed = True
        else:
            remote_rows = result

    any_local = bool(local_rows)
    any_remote = any(r.get('exists') for r in remote_rows)

    if not any_local and not any_remote and not remote_scan_failed:
        print(f'[chester] fresh=True — no existing directories found under '
              f'"{exp_prefix}". Proceeding.')
        return

    if remote_scan_failed and not any_local:
        print(f'[chester] fresh=True — remote scan of {backend_config.host} failed '
              f'and no local directories exist. Proceeding without cleanup.')
        return

    # Build unified name set (local names + remote names)
    local_by_name = {r['name']: r for r in local_rows}
    remote_by_name = {r['name']: r for r in remote_rows}
    all_names = sorted(set(local_by_name) | set(remote_by_name))

    print(f'\n[chester] fresh=True — scanning directories for '
          f'exp_prefix=\'{exp_prefix}\' mode={mode}\n')

    name_col = max((len(n) for n in all_names), default=30)
    name_col = max(name_col, 30)

    if is_remote:
        hdr = (f"  {'#':>3}  {'Experiment':<{name_col}}  "
               f"{'Local':<22}  {'Remote':<22}")
        sep = (f"  {'─':>3}  {'─' * name_col}  {'─' * 22}  {'─' * 22}")
    else:
        hdr = f"  {'#':>3}  {'Experiment':<{name_col}}  {'Local':<22}"
        sep = f"  {'─':>3}  {'─' * name_col}  {'─' * 22}"

    print(hdr)
    print(sep)

    total_local_bytes = 0
    total_local_pt = 0
    total_local_dirs = 0
    total_remote_pt = 0
    total_remote_dirs = 0

    for i, name in enumerate(all_names, 1):
        lr = local_by_name.get(name)
        if lr:
            local_str = f"{_format_size(lr['size_bytes'])}  {lr['pt_count']} pt"
            total_local_bytes += lr['size_bytes']
            total_local_pt += lr['pt_count']
            total_local_dirs += 1
        else:
            local_str = '—'

        if is_remote:
            rr = remote_by_name.get(name)
            if remote_scan_failed:
                remote_str = '[scan failed]'
            elif rr and rr.get('exists'):
                remote_str = f"{rr['size_str']}  {rr['pt_count']} pt"
                total_remote_pt += rr['pt_count']
                total_remote_dirs += 1
            else:
                remote_str = '—'
            print(f"  {i:>3}  {name:<{name_col}}  {local_str:<22}  {remote_str:<22}")
        else:
            print(f"  {i:>3}  {name:<{name_col}}  {local_str:<22}")

    print()
    print(f"  Local total:   {total_local_dirs} dirs   "
          f"{_format_size(total_local_bytes)}   {total_local_pt} pt/pth files")
    if is_remote:
        if remote_scan_failed:
            print("  Remote total:  [scan failed]")
        else:
            print(f"  Remote total:  {total_remote_dirs} dirs   "
                  f"{total_remote_pt} pt/pth files")

    print()
    if not confirm_action("WARNING: This will permanently delete ALL directories listed above."):
        print("Aborted.")
        sys.exit(0)

    print()

    # Delete local dirs
    for name in all_names:
        lr = local_by_name.get(name)
        if lr:
            try:
                _shutil.rmtree(lr['path'])
                print(f"[chester] Deleted local: {lr['path']}")
            except Exception as e:
                print(f"[chester] Warning: could not delete {lr['path']}: {e}")

    # Delete remote dirs (batched into one SSH call)
    if is_remote:
        if remote_scan_failed:
            print(f"[chester] Warning: remote scan of {backend_config.host} failed — "
                  f"remote directories were NOT deleted. Clean them manually.")
        else:
            existing_remote = [
                os.path.join(remote_batch_dir, r['name'])
                for r in remote_rows if r.get('exists')
            ]
            if existing_remote:
                quoted = ' '.join(shlex.quote(p) for p in existing_remote)
                try:
                    subprocess.run(
                        ['ssh', backend_config.host, f'rm -rf {quoted}'],
                        check=True,
                    )
                    n = len(existing_remote)
                    print(f"[chester] Deleted {n} remote "
                          f"{'directory' if n == 1 else 'directories'} "
                          f"on {backend_config.host}")
                except Exception as e:
                    print(f"[chester] Warning: remote deletion failed: {e}")

    print()


exp_count = -2
sub_process_popens = []
# Module-level registry: (exp_prefix, seq_identity) -> slurm_job_id
_slurm_job_registry: dict = {}
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%m_%d_%H_%M')
remote_confirmed = False


def _iter_serial_overrides(serial_steps):
    """Yield one override dict per serial step.

    serial_steps is a list of (key, [val1, val2, ...]) tuples.
    For a single serial key with N values, yields N dicts.
    """
    if not serial_steps:
        return
    # Currently only one serial key is supported
    key, values = serial_steps[0]
    for val in values:
        yield {key: val}


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
        auto_pull=True,
        extra_pull_dirs=None,
        sync_env=None,
        git_snapshot=True,
        confirm=False,
        fresh=False,
        skip_dependency_check=False,
        submodule_commits=None,
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
        extra_pull_dirs: Extra directories to pull from remote.
        sync_env: Override sync_on_launch config.
        git_snapshot: Save git state to log_dir before running (default True).
        confirm: If True, skip the remote execution confirmation prompt.
        fresh: If True, scan and delete existing exp_prefix dirs before launching;
               always prompts for confirmation regardless of confirm flag.
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
    elif use_singularity is True:
        if not backend.config.singularity:
            raise ValueError(
                f"use_singularity=True but backend '{mode}' has no singularity config"
            )
        # Force enable even if config says enabled: false
        backend.config.singularity.enabled = True
    elif use_singularity is None and backend.config.singularity:
        # Default: respect the enabled flag in config
        if not backend.config.singularity.enabled:
            backend.config.singularity = None

    # ----------------------------------------------------------------
    # 3.5. Validate submodule commit pinning
    # ----------------------------------------------------------------
    resolved_commits = {}
    submodule_worktrees = {}
    if submodule_commits:
        if backend.config.singularity is None:
            raise ValueError(
                f"[chester] submodule_commits requires singularity to be active for "
                f"backend '{mode}'. The current backend has no singularity config, or "
                f"singularity was disabled via use_singularity=False."
            )
        resolved_commits = _validate_submodule_commits(submodule_commits, project_path)
        remote_dir_for_wt = backend.config.remote_dir or project_path
        submodule_worktrees = _build_worktree_paths(resolved_commits, remote_dir_for_wt, timestamp)
        print(f"[chester] Submodule commit pinning:")
        for sub, sha in resolved_commits.items():
            wt = submodule_worktrees[sub]
            wt_rel = os.path.relpath(wt, remote_dir_for_wt)
            print(f"  {sub}: {submodule_commits[sub]} -> {sha}")
            print(f"      worktree: {wt_rel}")

    # ----------------------------------------------------------------
    # 4. Variant bookkeeping
    # ----------------------------------------------------------------
    last_variant = variant.pop('chester_last_variant', False)
    first_variant = variant.pop('chester_first_variant', False)
    seq_identity = variant.pop("_chester_seq_identity", None)
    pred_identities = variant.pop("_chester_pred_identities", None)
    serial_steps = variant.pop("_chester_serial_steps", None)

    # ----------------------------------------------------------------
    # 4.1. Clear dependency registry on first variant
    # ----------------------------------------------------------------
    if first_variant:
        _slurm_job_registry.clear()

    # ----------------------------------------------------------------
    # 4.2. Dependency check (non-SLURM guard for order="dependent")
    # ----------------------------------------------------------------
    has_dependent = seq_identity is not None

    if has_dependent and backend_config.type != "slurm" and not skip_dependency_check:
        raise ValueError(
            "[chester] order='dependent' dependencies are only enforced on "
            f"SLURM backends. Current mode: '{mode}'.\n"
            "Pass skip_dependency_check=True to run_experiment_lite() "
            "to suppress this check when you deliberately want unordered execution."
        )

    local_batch_dir = os.path.join(cfg_log_dir, sub_dir, exp_prefix)

    # ----------------------------------------------------------------
    # 4.5. Fresh start — delete existing dirs before ID generation
    # ----------------------------------------------------------------
    if fresh and first_variant:
        _fresh_start_v2(
            exp_prefix=exp_prefix,
            sub_dir=sub_dir,
            cfg_log_dir=cfg_log_dir,
            backend_config=backend_config,
            project_path=project_path,
            is_remote=is_remote,
            mode=mode,
        )

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
        remote_confirmed = confirm_action(
            f"Running in (non-dry) mode {mode}. Confirm?",
            skip=confirm,
        )
        if not remote_confirmed:
            sys.exit(1)

    # ----------------------------------------------------------------
    # 8. Rsync code for remote backends (first variant only)
    # ----------------------------------------------------------------
    if is_remote and first_variant and not dry:
        rsync_exclude = list(cfg.get("rsync_exclude", []))
        rsync_code_v2(
            remote_host=backend_config.host,
            remote_dir=backend_config.remote_dir,
            project_path=project_path,
            rsync_include=cfg.get("rsync_include", []),
            rsync_exclude=rsync_exclude,
        )

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
            # Local backend: generate command(s) and run directly
            if serial_steps:
                # Build one command per serial step, chain with &&
                commands = []
                for step_overrides in _iter_serial_overrides(serial_steps):
                    cmd = backend.generate_command(
                        backend_task,
                        script=script,
                        python_command=python_command,
                        env=merged_env or None,
                        hydra_enabled=hydra_enabled,
                        hydra_flags=hydra_flags,
                        extra_overrides=step_overrides,
                    )
                    commands.append(cmd)
                command = " && ".join(commands)
            else:
                command = backend.generate_command(
                    backend_task,
                    script=script,
                    python_command=python_command,
                    env=merged_env or None,
                    hydra_enabled=hydra_enabled,
                    hydra_flags=hydra_flags,
                )

            if print_command:
                print(command)
            if dry:
                return

            # Singularity wrapping always requires subprocess — in-process
            # hydra execution cannot run inside a container.
            use_subprocess = launch_with_subprocess or backend.config.singularity
            if use_subprocess:
                try:
                    run_env = dict(os.environ, **(merged_env or {}))
                    if wait_subprocess:
                        subprocess.call(command, shell=True, env=run_env,
                                        executable="/bin/bash")
                        popen_obj = None
                    else:
                        popen_obj = subprocess.Popen(command, shell=True, env=run_env,
                                                     executable="/bin/bash")
                    sub_process_popens.append(popen_obj)
                except Exception as e:
                    print(e)
                    if isinstance(e, KeyboardInterrupt):
                        raise
            else:
                # For hydra debug mode (in-process, no singularity)
                from .hydra_utils import run_hydra_command
                assert hydra_enabled, "hydra_enabled must be True when launch_with_subprocess is False"
                if serial_steps:
                    for step_overrides in _iter_serial_overrides(serial_steps):
                        step_cmd = backend.generate_command(
                            backend_task,
                            script=script, python_command=python_command,
                            env=merged_env or None,
                            hydra_enabled=hydra_enabled, hydra_flags=hydra_flags,
                            extra_overrides=step_overrides,
                        )
                        run_hydra_command(step_cmd, task["log_dir"], stub_method_call)
                else:
                    run_hydra_command(command, task["log_dir"], stub_method_call)
                popen_obj = None

            return popen_obj

        else:
            # Remote backends (ssh, slurm) — unified dispatch
            gen_kwargs = dict(
                script=script,
                python_command=python_command,
                env=merged_env or None,
                hydra_enabled=hydra_enabled,
                hydra_flags=hydra_flags,
            )
            if backend_config.type == "slurm" and slurm_overrides:
                gen_kwargs["slurm_overrides"] = slurm_overrides

            # Pass worktree info to backend for script generation (singularity only)
            if submodule_worktrees and backend.config.singularity:
                gen_kwargs["submodule_worktrees"] = submodule_worktrees
                gen_kwargs["submodule_resolved_commits"] = resolved_commits

            # For order="serial", pass serial step overrides to the backend
            if serial_steps:
                gen_kwargs["serial_steps"] = serial_steps

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

            # Resolve sequential dependency job IDs
            dependency_job_ids = None
            if backend_config.type == "slurm" and pred_identities and not dry:
                dependency_job_ids = []
                for pred_identity in pred_identities:
                    jid = _slurm_job_registry.get((exp_prefix, pred_identity))
                    if jid is not None:
                        dependency_job_ids.append(jid)
                    else:
                        print(f"[chester] Warning: predecessor job not found in registry "
                              f"for identity {pred_identity}. Submitting without this dependency.")
                if not dependency_job_ids:
                    dependency_job_ids = None

            # Submit via backend
            submit_kwargs = {"dry": dry}
            if dependency_job_ids:
                submit_kwargs["dependency_job_ids"] = dependency_job_ids
            submit_result = backend.submit(backend_task, script_content, **submit_kwargs)

            # Register job ID in sequential dependency registry
            if backend_config.type == "slurm" and seq_identity is not None:
                if submit_result is not None:
                    _slurm_job_registry[(exp_prefix, seq_identity)] = submit_result
                elif dry:
                    # Register placeholder so dry-run validation doesn't warn
                    _slurm_job_registry[(exp_prefix, seq_identity)] = "dry"

            # Register job in persistent job store
            if auto_pull and not dry:
                slurm_job_id = submit_result if backend_config.type == "slurm" else None
                resolved_extra_pull_dirs = _resolve_extra_pull_dirs_v2(
                    extra_pull_dirs, project_path, backend_config.remote_dir
                )
                _register_job_for_pull(
                    host=backend_config.host,
                    remote_log_dir=remote_log_dir,
                    local_log_dir=local_log_dir,
                    exp_name=task.get('exp_name', ''),
                    exp_prefix=exp_prefix,
                    extra_pull_dirs=resolved_extra_pull_dirs,
                    slurm_job_id=slurm_job_id,
                    submodule_commits=resolved_commits or None,
                    submodule_worktrees=submodule_worktrees or None,
                )

