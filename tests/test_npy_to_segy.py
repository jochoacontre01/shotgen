import subprocess
import sys

import numpy as np
import pytest
from shotgen.io import SegyIO, convert_npy_to_segy


def test_convert_npy_to_segy_roundtrip(tmp_path):
    """Test converting a (nz, nx) array with dx=50, dz=50 and reading back via SegyIO."""
    nz, nx = 40, 100
    dx, dz = 50.0, 50.0
    arr = np.linspace(1500.0, 4500.0, nz * nx, dtype=np.float32).reshape((nz, nx))

    npy_path = tmp_path / "test_model.npy"
    segy_path = tmp_path / "test_model.segy"
    np.save(npy_path, arr)

    model, out_dx, out_dz = convert_npy_to_segy(
        npy_path, segy_path, dx=dx, dz=dz, order="auto", verbose=False
    )

    assert model.shape == (nx, nz)
    assert out_dx == dx
    assert out_dz == dz
    np.testing.assert_allclose(model, arr.T, atol=1e-5)

    # Read back via SegyIO.read_model
    read_data, read_dx, read_dz = SegyIO.read_model(str(segy_path), verbose=False)
    assert read_data.shape == (nx, nz)
    assert read_dx == dx
    assert read_dz == dz
    np.testing.assert_allclose(read_data, arr.T, atol=1e-5)


def test_order_handling(tmp_path):
    """Test explicit 'zx', 'xz', and 'auto' order modes."""
    nz, nx = 30, 80
    dx, dz = 25.0, 50.0
    arr_zx = np.ones((nz, nx), dtype=np.float32) * 2000.0
    arr_xz = np.ones((nx, nz), dtype=np.float32) * 2000.0

    out_auto = tmp_path / "auto.segy"
    out_zx = tmp_path / "zx.segy"
    out_xz = tmp_path / "xz.segy"

    m_auto, _, _ = convert_npy_to_segy(arr_zx, out_auto, dx, dz, order="auto", verbose=False)
    assert m_auto.shape == (nx, nz)

    m_zx, _, _ = convert_npy_to_segy(arr_zx, out_zx, dx, dz, order="zx", verbose=False)
    assert m_zx.shape == (nx, nz)

    m_xz, _, _ = convert_npy_to_segy(arr_xz, out_xz, dx, dz, order="xz", verbose=False)
    assert m_xz.shape == (nx, nz)


def test_validation_errors(tmp_path):
    """Test validation errors for invalid dimensions, spacings, and values."""
    segy_path = tmp_path / "err.segy"

    # 1D array
    with pytest.raises(ValueError, match="Expected a 2D array"):
        convert_npy_to_segy(np.zeros(10), segy_path, dx=10.0, dz=10.0)

    # 3D array
    with pytest.raises(ValueError, match="Expected a 2D array"):
        convert_npy_to_segy(np.zeros((10, 10, 10)), segy_path, dx=10.0, dz=10.0)

    # Negative spacing
    with pytest.raises(ValueError, match="strictly positive"):
        convert_npy_to_segy(np.zeros((10, 10)), segy_path, dx=-5.0, dz=10.0)

    # NaN values
    bad_arr = np.ones((10, 10))
    bad_arr[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        convert_npy_to_segy(bad_arr, segy_path, dx=10.0, dz=10.0)

    # Missing file
    with pytest.raises(FileNotFoundError):
        convert_npy_to_segy(tmp_path / "nonexistent.npy", segy_path, dx=10.0, dz=10.0)

    # Invalid order
    with pytest.raises(ValueError, match="Supported orders are"):
        convert_npy_to_segy(np.ones((10, 10)), segy_path, dx=10.0, dz=10.0, order="invalid")


def test_cli_execution(tmp_path):
    """Test the scripts/npy_to_segy.py CLI entrypoint."""
    arr = np.full((50, 150), 3000.0, dtype=np.float32)
    npy_path = tmp_path / "cli_test.npy"
    np.save(npy_path, arr)

    cmd = [
        sys.executable,
        "scripts/npy_to_segy.py",
        str(npy_path),
        "--dx",
        "50",
        "--dz",
        "50",
    ]
    res = subprocess.run(cmd, check=False, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"

    expected_segy = tmp_path / "cli_test.segy"
    assert expected_segy.exists()

    data, dx, dz = SegyIO.read_model(str(expected_segy), verbose=False)
    assert data.shape == (150, 50)
    assert dx == 50.0
    assert dz == 50.0


def test_shotgen_gpu_loader_compatibility(tmp_path):
    """Test that shotgen-gpu's velocity model loader loads the generated SEGY correctly."""
    from shotgen_gpu.cli import _load_velocity_model

    arr = np.full((40, 80), 2200.0, dtype=np.float32)
    npy_path = tmp_path / "model.npy"
    segy_path = tmp_path / "model.segy"
    np.save(npy_path, arr)

    convert_npy_to_segy(npy_path, segy_path, dx=50.0, dz=50.0, order="auto", verbose=False)

    cfg = {"velocity_model": str(segy_path)}
    vel, nx, nz, dx, dz = _load_velocity_model(cfg)

    assert vel.shape == (80, 40)
    assert (nx, nz) == (80, 40)
    assert dx == 50.0
    assert dz == 50.0
