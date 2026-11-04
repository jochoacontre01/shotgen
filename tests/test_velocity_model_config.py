from pathlib import Path

import numpy as np
from shotgen.cli import _load_velocity_model_core, _run_simulation_core
from shotgen.io import SegyIO
from shotgen_gpu.cli import _run_simulation_cli


def test_simple_sinusoidal_segy_io():
    """Verify simple_sinusoidal.segy exists in assets/ and can be loaded with SegyIO."""
    asset_path = Path(__file__).resolve().parent.parent / "assets" / "simple_sinusoidal.segy"
    assert asset_path.is_file(), f"simple_sinusoidal.segy not found at {asset_path}"

    vel, dx, dz = SegyIO.read_model(str(asset_path), verbose=False)
    assert vel.shape == (2000, 1000), f"Expected shape (2000, 1000), got {vel.shape}"
    assert dx > 0.0 and dz > 0.0
    # Top velocity must be lower than bottom layer velocity
    v_top = vel[1000, 100]
    v_bottom = vel[1000, 900]
    assert v_bottom > v_top, f"Expected v_bottom > v_top, got {v_bottom} <= {v_top}"


def test_velocity_model_parsing_by_path(tmp_path):
    """Verify config parsing and simulation execution using a file path for velocity_model."""
    asset_path = Path(__file__).resolve().parent.parent / "assets" / "simple_sinusoidal.segy"

    out_dir_core = tmp_path / "sim_by_path_core"
    cfg_core = {
        "velocity_model": str(asset_path),
        "dx": 20.0,
        "dz": 20.0,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 20.0,
        "f0": 20.0,
        "max_size": [200.0, 100.0],
        "output_dir": str(out_dir_core),
    }

    vel, nx, nz, _dx, _dz = _load_velocity_model_core(cfg_core)
    assert vel is not None
    assert nx > 0 and nz > 0

    _shot, _saved_dir = _run_simulation_core(cfg_core)
    assert (out_dir_core / "traces.segy").exists()
    assert (out_dir_core / "metadata.h5").exists()

    # Also test shotgen-gpu CLI config parsing and simulation by path
    out_dir_gpu = tmp_path / "sim_by_path_gpu"
    cfg_gpu = dict(cfg_core)
    cfg_gpu["output_dir"] = str(out_dir_gpu)
    cfg_gpu["device"] = "cpu"
    cfg_gpu["engine"] = "devito"
    _run_simulation_cli(cfg_gpu)
    assert (out_dir_gpu / "traces.segy").exists()


def test_velocity_model_parsing_by_keyword(tmp_path):
    """Verify config parsing and simulation execution using an asset keyword for velocity_model."""
    out_dir_core = tmp_path / "sim_by_keyword_core"
    cfg_core = {
        "velocity_model": "simple_sinusoidal",
        "dx": 20.0,
        "dz": 20.0,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 20.0,
        "f0": 20.0,
        "max_size": [200.0, 100.0],
        "output_dir": str(out_dir_core),
    }

    vel, nx, nz, _dx, _dz = _load_velocity_model_core(cfg_core)
    assert vel is not None
    assert nx > 0 and nz > 0

    _shot, _saved_dir = _run_simulation_core(cfg_core)
    assert (out_dir_core / "traces.segy").exists()
    assert (out_dir_core / "metadata.h5").exists()

    # Also test shotgen-gpu CLI config parsing and simulation by keyword
    out_dir_gpu = tmp_path / "sim_by_keyword_gpu"
    cfg_gpu = dict(cfg_core)
    cfg_gpu["output_dir"] = str(out_dir_gpu)
    cfg_gpu["device"] = "cpu"
    cfg_gpu["engine"] = "devito"
    _run_simulation_cli(cfg_gpu)
    assert (out_dir_gpu / "traces.segy").exists()


def test_wavefield_animation_velocity_model_parsing():
    """Verify that wavefield_animation's _load_velocity_model correctly resolves asset keywords."""
    import sys
    examples_dir = Path(__file__).resolve().parent.parent / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    from wavefield_animation import _load_velocity_model

    cfg = {"velocity_model": "simple_sinusoidal", "dx": 10.0, "dz": 10.0}
    vel, nx, nz, dx, dz = _load_velocity_model(cfg)
    assert vel is not None
    assert vel.shape == (nx, nz)
    assert np.isclose(dx, 10.0)
    assert np.isclose(dz, 10.0)


def test_simple_sinusoidal_fault_segy_io():
    """Verify simple_sinusoidal_fault.segy exists, loads correctly, and contains a 45-deg normal fault."""
    asset_path = Path(__file__).resolve().parent.parent / "assets" / "simple_sinusoidal_fault.segy"
    assert asset_path.is_file(), f"simple_sinusoidal_fault.segy not found at {asset_path}"

    vel, dx, dz = SegyIO.read_model(str(asset_path), verbose=False)
    assert vel.shape == (2000, 1000)
    assert np.isclose(dx, 1.0) and np.isclose(dz, 1.0)

    # Check normal fault juxtaposition across the 45-degree fault plane (x + z = 1500)
    # At z = 550, fault is at x = 950. Hanging wall is to left (x=940), footwall to right (x=960).
    assert vel[940, 550] == 2000.0, "Hanging wall should contain downthrown lower velocity top layer at z=550"
    assert vel[960, 550] == 3000.0, "Footwall should contain higher velocity bottom layer at z=550"
