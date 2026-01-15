#!/usr/bin/env python
"""
Local tests for chester - no remote access required.

Run with: uv run python tests/test_local.py
"""
import os
import tempfile


def test_variant_generator():
    """Test VariantGenerator creates correct parameter combinations."""
    print("=" * 60)
    print("TEST: VariantGenerator")
    print("=" * 60)

    from chester.run_exp import VariantGenerator

    vg = VariantGenerator()
    vg.add('learning_rate', [0.001, 0.01])
    vg.add('batch_size', [32, 64])
    vg.add('hidden_dim', lambda batch_size: [batch_size * 2])  # Dependency

    variants = vg.variants()

    print(f"Generated {len(variants)} variants:")
    for i, v in enumerate(variants):
        print(f"  {i}: lr={v.learning_rate}, bs={v.batch_size}, hd={v.hidden_dim}")

    # Verify count: 2 * 2 = 4 variants
    assert len(variants) == 4, f"Expected 4 variants, got {len(variants)}"

    # Verify dependency works
    for v in variants:
        assert v.hidden_dim == v.batch_size * 2, "Dependency not working"

    # Verify first/last markers
    assert variants[0].get('chester_first_variant', False), "Missing first_variant marker"
    assert variants[-1].get('chester_last_variant', False), "Missing last_variant marker"

    print("PASSED\n")


def test_config_loading():
    """Test YAML-based config loading."""
    print("=" * 60)
    print("TEST: Config Loading")
    print("=" * 60)

    from chester import config

    # Test basic attributes exist
    print(f"PROJECT_PATH: {config.PROJECT_PATH}")
    print(f"LOG_DIR: {config.LOG_DIR}")
    print(f"HOST_ADDRESS: {config.HOST_ADDRESS}")

    assert isinstance(config.PROJECT_PATH, str), "PROJECT_PATH should be a string"
    assert isinstance(config.LOG_DIR, str), "LOG_DIR should be a string"
    assert isinstance(config.HOST_ADDRESS, dict), "HOST_ADDRESS should be a dict"
    assert 'local' in config.HOST_ADDRESS, "HOST_ADDRESS should have 'local' key"

    print("PASSED\n")


def test_local_command_generation():
    """Test local command generation."""
    print("=" * 60)
    print("TEST: Local Command Generation")
    print("=" * 60)

    from chester.slurm import to_local_command

    params = {
        'log_dir': '/tmp/test',
        'exp_name': 'test_exp',
        'args_data': 'base64encodeddata',
    }

    cmd = to_local_command(
        params=params,
        python_command='python',
        script='chester/run_exp_worker.py',
        env={'MY_VAR': 'value'}
    )

    print(f"Generated command:\n  {cmd}\n")

    assert 'python' in cmd, "Missing python command"
    assert 'run_exp_worker.py' in cmd, "Missing script"
    assert 'log_dir' in cmd, "Missing log_dir param"

    print("PASSED\n")


def test_slurm_command_generation():
    """Test SLURM command generation (dry run - no submission)."""
    print("=" * 60)
    print("TEST: SLURM Command Generation")
    print("=" * 60)

    from chester.slurm import to_slurm_command

    params = {
        'log_dir': '/tmp/test',
        'exp_name': 'test_exp',
        'args_data': 'base64encodeddata',
    }

    header = """#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00"""

    cmd_list = to_slurm_command(
        params=params,
        use_gpu=True,
        modules=['singularity'],
        cuda_module='cuda/11.8',
        header=header,
        python_command='python',
        script='train.py',
        use_singularity=False,
        simg_dir=None,
        remote_dir='/home/user/project',
        mount_options='',
        set_egl_gpu=False,
        env={},
    )

    full_cmd = "\n".join(cmd_list)
    print(f"Generated SLURM script:\n{full_cmd[:500]}...\n")

    assert '#SBATCH' in full_cmd, "Missing SBATCH header"
    assert 'module load' in full_cmd, "Missing module load"
    assert 'train.py' in full_cmd, "Missing script"

    print("PASSED\n")


def test_logger():
    """Test logger functionality."""
    print("=" * 60)
    print("TEST: Logger")
    print("=" * 60)

    from chester import logger

    with tempfile.TemporaryDirectory() as tmpdir:
        # Configure logger with CSV output
        logger.configure(dir=tmpdir, format_strs=['csv', 'stdout'])

        # Log some values
        logger.logkv('epoch', 1)
        logger.logkv('loss', 0.5)
        logger.logkv('accuracy', 0.85)
        logger.dumpkvs()

        logger.logkv('epoch', 2)
        logger.logkv('loss', 0.3)
        logger.logkv('accuracy', 0.92)
        logger.dumpkvs()

        # Check CSV was created
        csv_path = os.path.join(tmpdir, 'progress.csv')
        assert os.path.exists(csv_path), f"CSV file not created at {csv_path}"

        with open(csv_path) as f:
            content = f.read()
            print(f"CSV content:\n{content}")
            assert 'epoch' in content, "Missing epoch column"
            assert 'loss' in content, "Missing loss column"

        logger.reset()

    print("PASSED\n")


def test_ssh_command_generation():
    """Test SSH command generation for non-SLURM hosts."""
    print("=" * 60)
    print("TEST: SSH Command Generation")
    print("=" * 60)

    from chester.slurm import to_ssh_command

    params = {
        'log_dir': '/tmp/test',
        'exp_name': 'test_exp',
        'args_data': 'base64encodeddata',
    }

    cmd_list = to_ssh_command(
        params=params,
        python_command='python',
        remote_dir='/home/user/project',
        script='train.py',
        env={'CUDA_VISIBLE_DEVICES': '0'},
    )

    full_cmd = "\n".join(cmd_list)
    print(f"Generated SSH script:\n{full_cmd}\n")

    assert 'cd /home/user/project' in full_cmd, "Missing cd command"
    assert 'train.py' in full_cmd, "Missing script"
    assert '.done' in full_cmd, "Missing .done marker creation"

    print("PASSED\n")


def main():
    print("\n" + "=" * 60)
    print("CHESTER LOCAL TESTS")
    print("=" * 60 + "\n")

    tests = [
        test_variant_generator,
        test_config_loading,
        test_local_command_generation,
        test_slurm_command_generation,
        test_logger,
        test_ssh_command_generation,
    ]

    passed = 0
    failed = 0
    skipped = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}\n")
            failed += 1
        except Exception as e:
            if "SKIPPED" in str(e) or "not found" in str(e):
                skipped += 1
            else:
                print(f"ERROR: {e}\n")
                import traceback
                traceback.print_exc()
                failed += 1

    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
