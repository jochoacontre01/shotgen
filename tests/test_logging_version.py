import re
from pathlib import Path
from shotgen.pylogger import get_code_version, setup_file_logging, get_logger


def test_get_code_version():
    """Verify that get_code_version returns a valid version or commit string."""
    version = get_code_version()
    assert version is not None
    assert len(version) > 0
    # In a git repository, it should contain a hex commit or branch
    assert isinstance(version, str)


def test_setup_file_logging_header_and_format(tmp_path):
    """Verify setup_file_logging writes code version at the top and formats name as %Y%m%d_%H%M%S_<script>.log."""
    test_logger = get_logger("test_run")
    log_file = setup_file_logging(tmp_path, "my_script", logger=test_logger)

    assert log_file.exists()

    # Naming format: %Y%m%d_%H%M%S_my_script.log
    filename = log_file.name
    pattern = r"^\d{8}_\d{6}_my_script\.log$"
    assert re.match(pattern, filename) is not None, f"Filename '{filename}' does not match pattern {pattern}"

    # Verify code version is written at the very top of the file
    with open(log_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        second_line = f.readline()
        third_line = f.readline()

    assert first_line.startswith("# Code Version:"), f"First line does not start with '# Code Version:': {first_line}"
    code_version = get_code_version()
    assert code_version in first_line or len(first_line.strip()) > len("# Code Version:")
    assert second_line.startswith("# Started:")
    assert third_line.startswith("# Script: my_script")
