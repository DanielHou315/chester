import sys

def pytest_configure(config):
    """Strip Isaac Sim paths from sys.path to avoid numpy conflicts."""
    sys.path[:] = [
        p for p in sys.path
        if '_isaac_sim' not in p and 'IsaacLab' not in p
    ]
