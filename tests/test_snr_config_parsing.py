import json
import tempfile
import numpy as np
from pathlib import Path
import pytest

from shotgen.sampleshot import ShotRecord, LoadShotRecord
from shotgen.utils import generate_simulation_dir_name
from shotgen_gpu.cli import _run_simulation_cli


def test_snr_effect_pylops():
    """Verify that setting snr adds noise to the simulation output in pylops engine."""
    nx, nz = 30, 30
    dx, dz = 10.0, 10.0
    vel = 2000.0 * np.ones((nx, nz), dtype=np.float32)
    vel[:, 10:] = 3000.0

    # Simulation without SNR (noiseless)
    shot_clean = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz, n_sources=1, n_receivers=5, f0=25.0, engine="pylops", snr=None
    )
    shot_clean.set_model(vel)
    shot_clean.run(ms=300.0)

    # Simulation with SNR = 2.0 (noisy)
    shot_noisy = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz, n_sources=1, n_receivers=5, f0=25.0, engine="pylops", snr=2.0
    )
    shot_noisy.set_model(vel)
    shot_noisy.run(ms=300.0)

    assert shot_clean.snr is None
    assert shot_noisy.snr == 2.0

    # Traces should differ due to noise
    diff = shot_noisy.shot_run - shot_clean.shot_run
    assert np.linalg.norm(diff) > 0.0

    # Test apply_noise method directly
    shot_test = ShotRecord(nx=nx, nz=nz, dx=dx, dz=dz, n_sources=1, n_receivers=5, f0=25.0, engine="pylops")
    shot_test.set_model(vel)
    shot_test.run(ms=300.0)
    orig_data = shot_test.shot_run.copy()
    shot_test.apply_noise(snr=5.0)
    assert shot_test.snr == 5.0
    assert not np.allclose(orig_data, shot_test.shot_run)


def test_snr_save_and_load(tmp_path):
    """Verify that snr attribute is saved in metadata.h5 and loaded correctly by LoadShotRecord."""
    nx, nz = 30, 30
    dx, dz = 10.0, 10.0
    vel = 2500.0 * np.ones((nx, nz), dtype=np.float32)

    shot = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz, n_sources=1, n_receivers=5, f0=25.0, engine="pylops", snr=12.5
    )
    shot.set_model(vel)
    shot.run(ms=100.0)

    out_dir = tmp_path / "test_snr_save"
    shot.save_shot(str(out_dir))

    loaded = LoadShotRecord(str(out_dir))
    assert loaded.snr is not None
    assert np.isclose(loaded.snr, 12.5)


def test_full_config_parsing_gpu(tmp_path):
    """Verify that all config parameters, especially snr, are parsed and handled in shotgen-gpu CLI."""
    out_dir = tmp_path / "gpu_cfg_output"
    cfg = {
        "task": "simulation",
        "velocity_model": "marmousi",
        "dx": 10.0,
        "dz": 10.0,
        "nx": 40,
        "nz": 40,
        "n_sources": 1,
        "n_receivers": 6,
        "ms": 100.0,
        "f0": 20.0,
        "group_offset": 10.0,
        "shot_offset": 20.0,
        "gather": "common shot",
        "smooth": 3.0,
        "snr": 8.0,
        "fd_order": 4,
        "n_damping": 20,
        "engine": "devito",
        "device": "cpu",
        "fs_ms": 1.0,
        "output_dir": str(out_dir),
    }

    _run_simulation_cli(cfg)

    assert out_dir.exists()
    assert (out_dir / "traces.segy").exists()
    assert (out_dir / "metadata.h5").exists()

    loaded = LoadShotRecord(str(out_dir))
    assert loaded.snr is not None
    assert np.isclose(loaded.snr, 8.0)


def test_dir_name_generation_with_snr():
    """Verify that generate_simulation_dir_name includes snr correctly."""
    cfg = {
        "gather": "common shot",
        "nx": 100,
        "nz": 100,
        "dx": 5.0,
        "dz": 5.0,
        "n_sources": 2,
        "n_receivers": 10,
        "ms": 300.0,
        "f0": 25.0,
        "group_offset": 12.5,
        "shot_offset": 50.0,
        "snr": 0.1,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir="data")
    assert "0_1snr" in str(dir_name)
