import re

import numpy as np
from shotgen.pylogger import (
    get_code_version,
    get_invoked_command,
    get_logger,
    log_array_stats,
    setup_file_logging,
)


def test_get_code_version():
    """Verify that get_code_version returns a valid version or commit string."""
    version = get_code_version()
    assert version is not None
    assert len(version) > 0
    # In a git repository, it should contain a hex commit or branch
    assert isinstance(version, str)


def test_get_invoked_command():
    """Verify that get_invoked_command returns non-empty command string."""
    cmd = get_invoked_command()
    assert cmd is not None
    assert len(cmd) > 0
    assert isinstance(cmd, str)


def test_setup_file_logging_header_and_format(tmp_path):
    """Verify setup_file_logging writes code version and command at the top and formats name as %Y%m%d_%H%M%S_<script>.log."""
    test_logger = get_logger("test_run")
    log_file = setup_file_logging(tmp_path, "my_script", logger=test_logger)

    assert log_file.exists()

    # Naming format: %Y%m%d_%H%M%S_my_script.log
    filename = log_file.name
    pattern = r"^\d{8}_\d{6}_my_script\.log$"
    assert re.match(pattern, filename) is not None, f"Filename '{filename}' does not match pattern {pattern}"

    # Verify code version and command are written at the very top of the file
    with open(log_file, "r", encoding="utf-8") as f:
        first_line = f.readline()
        second_line = f.readline()
        third_line = f.readline()
        fourth_line = f.readline()

    assert first_line.startswith("# Code Version:"), f"First line does not start with '# Code Version:': {first_line}"
    code_version = get_code_version()
    assert code_version in first_line or len(first_line.strip()) > len("# Code Version:")
    assert second_line.startswith("# Command:"), f"Second line does not start with '# Command:': {second_line}"
    assert third_line.startswith("# Started:")
    assert fourth_line.startswith("# Script: my_script")


def test_log_array_stats_in_file(tmp_path):
    """Verify log_array_stats writes shape, dimensions, min, max, and mean to the log file."""
    test_logger = get_logger("test_array_stats")
    log_file = setup_file_logging(tmp_path, "array_script", logger=test_logger)

    arr = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    log_array_stats(test_logger, "sample_velocity", arr)

    # Also test via logger method
    empty_arr = np.empty((0, 5), dtype=np.float32)
    test_logger.log_array("empty_test", empty_arr)

    # Flush handlers to ensure file content is synced
    for h in test_logger.handlers:
        h.flush()

    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Array 'sample_velocity' stats" in content
    assert "Shape: (2, 2)" in content
    assert "Dimensions: 2" in content
    assert "Min: 10" in content
    assert "Max: 40" in content
    assert "Mean: 25" in content
    assert "Array 'empty_test' stats" in content
    assert "Empty (size=0)" in content


