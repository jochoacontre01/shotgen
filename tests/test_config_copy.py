import json
from pathlib import Path
from shotgen.cli import main as core_main
from shotgen_gpu.cli import main as gpu_main
from shotgen.sampleshot import ShotRecord


def test_shotgen_core_cli_copies_config(tmp_path, monkeypatch):
    """Verify that shotgen-core CLI copies the source config file into the dataset directory."""
    cfg_file = tmp_path / "custom_experiment_core.json"
    out_dir = tmp_path / "core_dataset"
    cfg = {
        "velocity_model": "simple_sinusoidal",
        "dx": 20.0,
        "dz": 20.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 20.0,
        "f0": 20.0,
        "engine": "pylops",
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    monkeypatch.setattr("sys.argv", ["shotgen-core", "--config", str(cfg_file)])
    core_main()

    copied_cfg = out_dir / "custom_experiment_core.json"
    assert copied_cfg.exists(), f"Expected {copied_cfg} to exist in dataset directory"
    with open(copied_cfg, "r") as f:
        loaded = json.load(f)
    assert loaded["n_sources"] == 1


def test_shotgen_gpu_cli_copies_config(tmp_path, monkeypatch):
    """Verify that shotgen-gpu CLI copies the source config file into the dataset directory."""
    cfg_file = tmp_path / "custom_experiment_gpu.json"
    out_dir = tmp_path / "gpu_dataset"
    cfg = {
        "velocity_model": "simple_sinusoidal",
        "dx": 20.0,
        "dz": 20.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 20.0,
        "f0": 20.0,
        "engine": "devito",
        "device": "cpu",
        "output_dir": str(out_dir),
    }
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    monkeypatch.setattr("sys.argv", ["shotgen-gpu", "--config", str(cfg_file)])
    gpu_main()

    copied_cfg = out_dir / "custom_experiment_gpu.json"
    assert copied_cfg.exists(), f"Expected {copied_cfg} to exist in dataset directory"
    with open(copied_cfg, "r") as f:
        loaded = json.load(f)
    assert loaded["n_sources"] == 1


def test_shot_record_save_shot_copies_config(tmp_path):
    """Verify that ShotRecord.save_shot copies the config file if specified."""
    import numpy as np
    cfg_file = tmp_path / "manual_run.json"
    cfg_file.write_text('{"test": true}')
    out_dir = tmp_path / "shot_dataset"

    shot = ShotRecord(nx=20, nz=20, dx=10.0, dz=10.0, n_sources=1, n_receivers=2, engine="pylops", device="cpu", config_file=str(cfg_file))
    shot.shot_run = np.zeros((1, 2, 20), dtype=np.float32)
    shot.time_vector = np.linspace(0, 0.05, 20)
    shot.save_shot(str(out_dir))

    copied_cfg = out_dir / "manual_run.json"
    assert copied_cfg.exists()
    assert copied_cfg.read_text() == '{"test": true}'

