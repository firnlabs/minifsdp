"""Root conftest: propagate coverage config to spawned subprocesses."""

import os
import subprocess


def pytest_configure(config: object) -> None:
    """Set COVERAGE_PROCESS_START so mp.spawn child processes record coverage."""
    os.environ.setdefault("COVERAGE_PROCESS_START", "pyproject.toml")


def pytest_sessionfinish(session: object, exitstatus: int) -> None:
    """Combine subprocess coverage files after pytest-cov finishes reporting."""
    subprocess.run(["coverage", "combine", "--append"], capture_output=True)
