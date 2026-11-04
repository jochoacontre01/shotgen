import subprocess
import sys
from pathlib import Path


def test_ruff_linter_clean():
    """Verify that the ruff linter passes with zero errors across all repository Python directories."""
    repo_root = Path(__file__).resolve().parent.parent
    targets = ["packages", "examples", "scripts", "tests"]

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *targets],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Ruff linter failed with errors:\n{result.stdout}\n{result.stderr}"
