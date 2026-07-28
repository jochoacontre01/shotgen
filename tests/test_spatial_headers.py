import numpy as np
import pytest
import segyio
import pathlib
from shotgen.sampleshot import (
    load_marmousi,
    load_sigsbee,
    load_bpsalt,
    load_overthrust,
    map_coordinate_to_index,
    _find_assets_dir,
)

overthrust_file = _find_assets_dir() / "marine_overthrust_3d.segy"
skip_if_no_overthrust = pytest.mark.skipif(
    not overthrust_file.exists(),
    reason="marine_overthrust_3d.segy asset not present"
)


def test_marmousi_spatial_headers():
    data, metadata = load_marmousi()
    assert data.shape == (13601, 2801)
    assert metadata["dx"] == 1.0
    assert metadata["dz"] == 1.0
    assert metadata["nx"] == 13601
    assert metadata["nz"] == 2801
    assert metadata["origin_x"] == 0.0
    assert metadata["origin_z"] == 0.0


def test_sigsbee_spatial_headers():
    data, metadata = load_sigsbee()
    assert data.shape == (3201, 1201)
    assert metadata["dx"] == 1.0
    assert metadata["dz"] == 1.0
    assert metadata["nx"] == 3201
    assert metadata["nz"] == 1201


def test_bpsalt_spatial_headers():
    data, metadata = load_bpsalt()
    assert data.shape == (5395, 1911)
    assert metadata["dx"] == 12.5
    assert metadata["dz"] == 6.5
    assert metadata["nx"] == 5395
    assert metadata["nz"] == 1911


@skip_if_no_overthrust
def test_overthrust_spatial_headers():
    data, metadata = load_overthrust()
    assert data.shape == (801, 185)
    assert metadata["dx"] == 25.0
    assert metadata["dz"] == 25.0
    assert metadata["nx"] == 801
    assert metadata["nz"] == 185


def test_header_extraction_accuracy():
    bpsalt_path = _find_assets_dir() / "vel_z6.25m_x12.5m_exact.segy"
    with segyio.open(bpsalt_path, "r", ignore_geometry=True) as f:
        assert f.bin[segyio.BinField.Interval] == 6500
        assert f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL] == 6500
        assert f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX] == 12500

    if overthrust_file.exists():
        with segyio.open(overthrust_file, "r", ignore_geometry=True) as f:
            assert f.bin[segyio.BinField.Interval] == 25000
            assert f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL] == 25000
            assert f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX] == 25000


def test_coordinate_mapping():
    assert map_coordinate_to_index(0.0, spacing=12.5, origin=0.0, max_cells=100) == 0
    assert map_coordinate_to_index(12.5, spacing=12.5, origin=0.0, max_cells=100) == 1
    assert map_coordinate_to_index(6.25, spacing=12.5, origin=0.0, max_cells=100) == 1
    assert map_coordinate_to_index(6.24, spacing=12.5, origin=0.0, max_cells=100) == 0

    positions = np.array([0.0, 12.5, 25.0])
    indices = map_coordinate_to_index(positions, spacing=12.5, origin=0.0, max_cells=100)
    assert np.array_equal(indices, [0, 1, 2])

    with pytest.raises(ValueError):
        map_coordinate_to_index(-0.1, spacing=12.5, origin=0.0, max_cells=100)

    with pytest.raises(ValueError):
        map_coordinate_to_index(1250.0, spacing=12.5, origin=0.0, max_cells=100)
