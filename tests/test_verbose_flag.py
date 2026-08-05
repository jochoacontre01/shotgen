import os
import json
import pytest
from pathlib import Path

from shotgen_gpu.config import configure_devito_device
from shotgen_gpu.cli import _run_simulation_cli


def test_configure_devito_device_verbose():
    """Verify that configure_devito_device toggles NVCOMPILER_ACC_TIME environment variable based on verbose flag."""
    configure_devito_device("cpu", verbose=False)
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "0"

    configure_devito_device("cpu", verbose=True)
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "1"

    # Reset back to default (verbose=False)
    configure_devito_device("cpu", verbose=False)
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "0"


def test_simulation_cli_verbose_output(tmp_path, capsys):
    """Verify that GPU runtime verbose info is displayed when verbose=True and hidden when verbose=False."""
    out_dir_quiet = tmp_path / "quiet_sim"
    cfg_quiet = {
        "task": "simulation",
        "velocity_model": "marmousi",
        "dx": 10.0,
        "dz": 10.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 50.0,
        "f0": 25.0,
        "engine": "devito",
        "device": "cpu",
        "output_dir": str(out_dir_quiet),
    }

    _run_simulation_cli(cfg_quiet, verbose=False)
    captured_quiet = capsys.readouterr()
    assert "GPU runtime:" not in captured_quiet.out
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "0"

    out_dir_verbose = tmp_path / "verbose_sim"
    cfg_verbose = {
        "task": "simulation",
        "velocity_model": "marmousi",
        "dx": 10.0,
        "dz": 10.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 50.0,
        "f0": 25.0,
        "engine": "devito",
        "device": "cpu",
        "output_dir": str(out_dir_verbose),
    }

    _run_simulation_cli(cfg_verbose, verbose=True)
    captured_verbose = capsys.readouterr()
    assert "GPU runtime:" in captured_verbose.out
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "1"


def test_cli_main_verbose(tmp_path, monkeypatch, capsys):
    """Verify that shotgen-gpu CLI main parses -v flag properly."""
    from shotgen_gpu.cli import main

    cfg_file = tmp_path / "test_cfg.json"
    out_dir = tmp_path / "main_out"
    cfg = {
        "task": "simulation",
        "velocity_model": "marmousi",
        "dx": 10.0,
        "dz": 10.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 50.0,
        "f0": 25.0,
        "engine": "devito",
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    # Test quiet run
    monkeypatch.setattr("sys.argv", ["shotgen-gpu", "--config", str(cfg_file)])
    main()
    cap_quiet = capsys.readouterr()
    assert "GPU runtime:" not in cap_quiet.out
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "0"

    # Test verbose run with -v
    monkeypatch.setattr("sys.argv", ["shotgen-gpu", "--config", str(cfg_file), "-v"])
    main()
    cap_verbose = capsys.readouterr()
    assert "GPU runtime:" in cap_verbose.out
    assert os.environ.get("NVCOMPILER_ACC_TIME") == "1"
