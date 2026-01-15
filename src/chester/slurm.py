import os
import os.path as osp
import re
from subprocess import run
from tempfile import NamedTemporaryFile
from chester import config
from chester.hydra_utils import to_hydra_command

# TODO remove the singularity part

slurm_dir = './'


def get_package_manager_setup_commands(sync=True, source_prepare=True):
    """
    Generate shell commands for package manager setup based on config.

    Args:
        sync: Whether to sync/update the environment
        source_prepare: Whether to source prepare.sh for custom project setup

    Returns list of shell commands to set up the environment.
    """
    pkg_manager = config.PACKAGE_MANAGER
    sync_on_launch = config.SYNC_ON_LAUNCH if sync else False
    commands = []

    if pkg_manager == 'uv':
        # Add common uv installation paths to PATH
        commands.append('export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"')
        # Auto-install uv if not present
        commands.append(
            'if ! command -v uv &> /dev/null; then '
            'echo "[chester] Installing uv..."; '
            'curl -LsSf https://astral.sh/uv/install.sh | sh; '
            'export PATH="$HOME/.local/bin:$PATH"; '
            'fi'
        )
        # Set CUDA_HOME for packages that need it
        commands.append(
            'export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))'
        )
        commands.append('echo "[chester] CUDA_HOME=$CUDA_HOME"')
        # Set fake versions for setuptools-scm packages
        commands.append('export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.7.0')
        # Set MAX_JOBS for parallel CUDA compilation
        commands.append('export MAX_JOBS=16')
        # Install ninja globally for faster builds
        commands.append('uv tool install ninja 2>/dev/null || true')
        # Sync environment
        if sync_on_launch:
            commands.append('echo "[chester] Running uv sync..."')
            commands.append('uv sync')

    elif pkg_manager == 'conda':
        conda_cmd = config.CONDA_COMMAND or 'conda'
        conda_env = config.CONDA_ENV
        if not conda_env:
            raise ValueError("chester config: conda_env is required when package_manager is 'conda'")

        # Source bashrc to get conda in PATH
        commands.append('source ~/.bashrc')
        # Activate conda environment
        commands.append(f'{conda_cmd} activate {conda_env}')
        # Set CUDA_HOME
        commands.append(
            'export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))'
        )
        commands.append('echo "[chester] CUDA_HOME=$CUDA_HOME"')
        # Sync environment (update from environment.yml if exists)
        if sync_on_launch:
            commands.append(
                f'if [ -f environment.yml ]; then '
                f'echo "[chester] Running {conda_cmd} env update..."; '
                f'{conda_cmd} env update -n {conda_env} -f environment.yml --prune; '
                f'fi'
            )
    else:
        raise ValueError(f"chester config: unknown package_manager '{pkg_manager}' (expected 'uv' or 'conda')")

    # Source prepare.sh for custom project setup (if exists)
    if source_prepare:
        commands.append('if [ -f ./prepare.sh ]; then . ./prepare.sh; fi')

    return commands


def get_python_command(base_command="python"):
    """
    Get the python command wrapped for the configured package manager.

    Args:
        base_command: Base python command (e.g., "python", "srun python")

    Returns:
        Wrapped command string (e.g., "uv run python", "conda run -n env python")
    """
    pkg_manager = config.PACKAGE_MANAGER

    if pkg_manager == 'uv':
        # For uv, prefix with "uv run"
        return f'uv run {base_command}'
    elif pkg_manager == 'conda':
        # For conda, the environment is already activated, no wrapper needed
        return base_command
    else:
        return base_command


def slurm_run_scripts(scripts):
    """this is another function that those _sub files should call. this actually execute files"""
    # TODO support running multiple scripts

    assert isinstance(scripts, str)

    os.chdir(slurm_dir)

    # make sure it will run.
    assert scripts.startswith('#!/usr/bin/env bash\n')
    file_temp = NamedTemporaryFile(delete=False)
    file_temp.write(scripts.encode('utf-8'))
    file_temp.close()
    run(['sbatch', file_temp.name], check=True)
    os.remove(file_temp.name)


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

def to_local_command(
    params, 
    python_command="python", 
    script=None,
    env={}):
    command = python_command + " " + script

    for k, v in env.items():
        command = ("%s=%s " % (k, v)) + command
    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)
    if pre_commands is not None or post_commands is not None:
        print("Not executing the pre_commands: ", pre_commands, ", nor post_commands: ", post_commands)

    for k, v in params.items():
        if isinstance(v, dict):
            for nk, nv in v.items():
                if str(nk) == "_name":
                    command += "  --%s %s" % (k, _to_param_val(nv))
                else:
                    command += \
                        "  --%s_%s %s" % (k, nk, _to_param_val(nv))
        else:
            command += "  --%s %s" % (k, _to_param_val(v))
    return command

def to_slurm_command(params, header, python_command="srun python", remote_dir='~/',
                     script=None,  # Defer config access to runtime
                     simg_dir=None, use_gpu=False, modules=None, cuda_module=None, use_singularity=True,
                     mount_options=None, compile_script=None, wait_compile=None, set_egl_gpu=False,
                     env=None, hydra_enabled=False, hydra_flags=None, sync_env=None):
    """
    Transfer the commands to the format that can be run by slurm.

    Package manager (uv/conda) is determined by chester.yaml config.

    Args:
        params: Dictionary of parameters
        header: SLURM header with #SBATCH directives
        python_command: Base python command (will be wrapped based on package_manager)
        remote_dir: Remote directory
        script: Script to run
        simg_dir: Singularity image directory
        use_gpu: Whether to use GPU
        modules: List of modules to load
        cuda_module: CUDA module to load
        use_singularity: Whether to use singularity container
        mount_options: Singularity mount options
        compile_script: Optional compile script
        wait_compile: Time to wait after compile
        set_egl_gpu: Whether to set EGL_GPU env var
        env: Environment variables
        hydra_enabled: Whether to use Hydra
        hydra_flags: Hydra flags
        sync_env: Override for sync_on_launch config (None = use config)
    """
    # Resolve default script path at runtime (not import time)
    if script is None:
        script = osp.join(config.PROJECT_PATH, 'scripts/run_experiment.py')

    command_list = list()
    command_list.append(header)
    sing_commands = list()

    command_list.append('set -x')  # echo commands to stdout
    command_list.append('set -u')  # throw an error if unset variable referenced
    command_list.append('set -e')  # exit on errors
    command_list.append('srun hostname')
    command_list.append('cd {}'.format(remote_dir))
    for remote_module in modules:
        command_list.append('module load ' + remote_module)
    if use_gpu:
        assert cuda_module is not None
        command_list.append('module load ' + cuda_module)

    # Log into singularity shell
    if use_singularity:
        if mount_options is not None:
            options = '-B ' + mount_options
        else:
            options = ''
        sing_prefix = 'singularity exec {} {} {} /bin/bash -c'.format(options, '--nv' if use_gpu else '', simg_dir)
    else:
        sing_prefix = '/bin/bash -c'

    # Package manager setup (replaces hardcoded prepare.sh sourcing)
    do_sync = sync_env if sync_env is not None else config.SYNC_ON_LAUNCH
    setup_commands = get_package_manager_setup_commands(sync=do_sync)
    sing_commands.extend(setup_commands)

    # NCCL settings for distributed training
    sing_commands.append("export NCCL_DEBUG=INFO")
    sing_commands.append("export PYTHONFAULTHANDLER=1")
    sing_commands.append("export NCCL_SOCKET_IFNAME=^docker0,lo")

    if set_egl_gpu:
        sing_commands.append('export EGL_GPU=$SLURM_JOB_GRES')
        sing_commands.append('echo $EGL_GPU')
    if compile_script is not None:
        sing_commands.append(compile_script)
    if wait_compile is not None:
        sing_commands.append('sleep ' + str(int(wait_compile)))

    pre_commands = params.pop("pre_commands", None)
    post_commands = params.pop("post_commands", None)
    if pre_commands is not None:
        command_list.extend(pre_commands)

    # Wrap python command for package manager
    wrapped_python = get_python_command(python_command)

    if hydra_enabled:
        command = to_hydra_command(params, wrapped_python, script, hydra_flags, env)
    else:
        command = to_local_command(params, wrapped_python, script, env)
    sing_commands.append(command)

    # Add .done marker file on successful completion (for auto_pull to detect)
    assert 'log_dir' in params, "chester auto_pull: log_dir must be in params"
    log_dir = params['log_dir']
    sing_commands.append(f'touch {log_dir}/.done')

    all_sing_cmds = ' && '.join(sing_commands)
    command_list.append(sing_prefix + ' \'{}\''.format(all_sing_cmds))
    if post_commands is not None:
        command_list.extend(post_commands)
    return command_list

def to_ssh_command(params, python_command="python", remote_dir='./',
                   script='main.py', env=None, hydra_enabled=False, hydra_flags=None,
                   sync_env=None):
    """
    Generate a bash script for SSH-based remote execution (no SLURM).

    This creates a simpler script than to_slurm_command, suitable for
    running on remote machines via SSH + nohup.

    Package manager (uv/conda) is determined by chester.yaml config.

    Args:
        params: Dictionary of parameters to pass to the script
        python_command: Base python command (will be wrapped based on package_manager config)
        remote_dir: Remote directory to cd into
        script: Script to run
        env: Environment variables to set
        hydra_enabled: Whether to use Hydra command format
        hydra_flags: Hydra flags if hydra_enabled
        sync_env: Override for sync_on_launch config (None = use config)
    """
    command_list = []

    # Bash header
    command_list.append('#!/usr/bin/env bash')
    command_list.append('set -x')  # echo commands to stdout
    command_list.append('set -u')  # throw an error if unset variable referenced
    command_list.append('set -e')  # exit on errors

    # Change to remote directory
    command_list.append(f'cd {remote_dir}')
    # Add project root to PYTHONPATH for local modules
    command_list.append('export PYTHONPATH="$PWD:${PYTHONPATH:-}"')

    # Package manager setup (uv or conda based on config)
    do_sync = sync_env if sync_env is not None else config.SYNC_ON_LAUNCH
    setup_commands = get_package_manager_setup_commands(sync=do_sync)
    command_list.extend(setup_commands)

    # Wrap python command for package manager
    wrapped_python = get_python_command(python_command)

    # Generate the main command
    if hydra_enabled:
        command = to_hydra_command(params, wrapped_python, script, hydra_flags, env)
    else:
        command = to_local_command(params, wrapped_python, script, env if env else {})
    command_list.append(command)

    # Add .done marker file on successful completion (for auto_pull to detect)
    assert 'log_dir' in params, "chester auto_pull: log_dir must be in params"
    log_dir = params['log_dir']
    command_list.append(f'touch {log_dir}/.done')

    return command_list


# if __name__ == '__main__':
#     slurm_run_scripts(header)
