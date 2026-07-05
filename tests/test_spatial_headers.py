import numpy as np
import pytest
import segyio
import pathlib
from shotgen.sampleshot import (
    load_marmousi,
    load_sigsbee,
    load_bpsalt,
    load_overthrust,
    map_coordinate_to_index
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

def test_overthrust_spatial_headers():
    data, metadata = load_overthrust()
    assert data.shape == (801, 185)
    assert metadata["dx"] == 25.0
    assert metadata["dz"] == 25.0
    assert metadata["nx"] == 801
    assert metadata["nz"] == 185

def test_header_extraction_accuracy():
    # Verify that the headers in Bpsalt and Overthrust indeed contain the physical values
    bpsalt_path = pathlib.Path(__file__).resolve().parents[1] / "assets/vel_z6.25m_x12.5m_exact.segy"
    with segyio.open(bpsalt_path, "r", ignore_geometry=True) as f:
        assert f.bin[segyio.BinField.Interval] == 6500 # dz * 1000
        assert f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL] == 6500
        assert f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX] == 12500 # dx * 1000

    overthrust_path = pathlib.Path(__file__).resolve().parents[1] / "assets/marine_overthrust_3d.segy"
    with segyio.open(overthrust_path, "r", ignore_geometry=True) as f:
        assert f.bin[segyio.BinField.Interval] == 25000 # dz * 1000
        assert f.header[0][segyio.TraceField.TRACE_SAMPLE_INTERVAL] == 25000
        assert f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX] == 25000 # dx * 1000

def test_coordinate_mapping():
    # Grid: dx = 12.5, origin = 0.0, max_cells = 100
    # Coordinates in meters:
    # 0.0 -> index 0
    # 12.5 -> index 1
    # 6.25 -> index 1 (nearest neighbor rounding)
    # 6.24 -> index 0 (nearest neighbor rounding)
    assert map_coordinate_to_index(0.0, spacing=12.5, origin=0.0, max_cells=100) == 0
    assert map_coordinate_to_index(12.5, spacing=12.5, origin=0.0, max_cells=100) == 1
    assert map_coordinate_to_index(6.25, spacing=12.5, origin=0.0, max_cells=100) == 1
    assert map_coordinate_to_index(6.24, spacing=12.5, origin=0.0, max_cells=100) == 0
    
    # Array input mapping
    positions = np.array([0.0, 12.5, 25.0])
    indices = map_coordinate_to_index(positions, spacing=12.5, origin=0.0, max_cells=100)
    assert np.array_equal(indices, [0, 1, 2])
    
    # Out of bounds cases (negative)
    with pytest.raises(ValueError):
        map_coordinate_to_index(-0.1, spacing=12.5, origin=0.0, max_cells=100)
        
    # Out of bounds cases (exceeding maximum bounds)
    # Max valid index is max_cells - 1 = 99. Max position is 99 * 12.5 = 1237.5
    # 100 * 12.5 = 1250 should be out of bounds
    with pytest.raises(ValueError):
        map_coordinate_to_index(1250.0, spacing=12.5, origin=0.0, max_cells=100)
