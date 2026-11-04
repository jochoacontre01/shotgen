import shutil
import stat
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_ruff_linter_clean():
    """Verify that the ruff linter passes with zero errors across all repository Python directories."""
    targets = ["packages", "examples", "scripts", "tests"]

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *targets],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Ruff linter failed with errors:\n{result.stdout}\n{result.stderr}"


def test_git_hooks_configured():
    """Verify .githooks/pre-commit and .githooks/pre-push exist and core.hooksPath is set."""
    pre_commit = REPO_ROOT / ".githooks" / "pre-commit"
    pre_push = REPO_ROOT / ".githooks" / "pre-push"

    assert pre_commit.is_file(), "Missing .githooks/pre-commit"
    assert pre_push.is_file(), "Missing .githooks/pre-push"

    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    assert hooks_path == ".githooks", f"Expected core.hooksPath to be '.githooks', got '{hooks_path}'"


def test_pre_commit_and_pre_push_hooks_auto_fix_and_pytest_gate(tmp_path):
    """Verify pre-commit and pre-push hooks auto-fix lint issues and gate pushes on pytest results."""
    # Initialize an isolated temporary git repository
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)

    hooks_dir = tmp_path / ".githooks"
    hooks_dir.mkdir()
    for hook_name in ("pre-commit", "pre-push"):
        src = REPO_ROOT / ".githooks" / hook_name
        dst = hooks_dir / hook_name
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=tmp_path, check=True)

    for folder in ("packages", "examples", "scripts", "tests"):
        (tmp_path / folder).mkdir()

    # Add a passing unit test in tmp_path/tests/
    dummy_test = tmp_path / "tests" / "test_dummy.py"
    dummy_test.write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_dummy.py"], cwd=tmp_path, check=True)

    # 1. Test pre-commit hook: stage a file with an unused import (F401)
    sample_script = tmp_path / "scripts" / "sample.py"
    sample_script.write_text("import sys\nimport os\n\nprint(sys.version)\n", encoding="utf-8")

    subprocess.run(["git", "add", "scripts/sample.py"], cwd=tmp_path, check=True)
    commit_res = subprocess.run(
        ["git", "commit", "-m", "feat: add sample script"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert commit_res.returncode == 0, f"pre-commit failed:\n{commit_res.stdout}\n{commit_res.stderr}"

    # Verify unused 'import os' was automatically removed and staged into the commit
    committed_content = sample_script.read_text(encoding="utf-8")
    assert "import os" not in committed_content
    assert "import sys" in committed_content

    # 2. Test pre-push hook (Option B + pytest gate passing): bypass pre-commit with --no-verify, then run pre-push
    sample_script.write_text("import sys\nimport math\n\nprint(sys.version)\n", encoding="utf-8")
    subprocess.run(["git", "add", "scripts/sample.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "--no-verify", "-m", "chore: bypass pre-commit with unused import"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    push_hook_res = subprocess.run(
        [str(hooks_dir / "pre-push")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert push_hook_res.returncode == 0, f"pre-push failed:\n{push_hook_res.stdout}\n{push_hook_res.stderr}"
    assert "import math" not in sample_script.read_text(encoding="utf-8")

    # Verify pre-push automatically created the 'style(lint): auto-apply ruff lint fixes' commit
    last_commit_msg = subprocess.run(
        ["git", "log", "-1", "--pretty=%s"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert last_commit_msg == "style(lint): auto-apply ruff lint fixes"

    # 3. Test pre-push hook aborts when a pytest test fails
    dummy_test.write_text("def test_broken():\n    assert False\n", encoding="utf-8")
    subprocess.run(["git", "add", "tests/test_dummy.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "test: introduce broken test"], cwd=tmp_path, check=True, capture_output=True)

    failed_push_res = subprocess.run(
        [str(hooks_dir / "pre-push")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_push_res.returncode != 0, "Expected pre-push to abort when pytest fails"
