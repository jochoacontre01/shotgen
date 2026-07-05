import numpy as np
import pytest
from shotgen.sampleshot import load_overthrust, ShotRecord

def test_load_overthrust():
    # 1. Load the overthrust model
    vp = load_overthrust()
    
    # 2. Check dimensions
    assert vp.shape == (801, 185)
    
    # 3. Check values are physically valid velocity range
    assert vp.min() >= 1480.0
    assert vp.max() <= 4000.0
    assert np.allclose(vp[:, 0], 1480.0) # top boundary is water layer

def test_shotrecord_with_overthrust():
    # Load and slice to a very small size for fast testing
    vp_full = load_overthrust()
    vp = vp_full[300:350, :40] # shape (50, 40)
    
    nx, nz = vp.shape
    dx = 25.0
    dz = 25.0
    
    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=2,
        n_receivers=6,
        f0=25.0,
        src_origin=(100.0, 50.0),
        rec_origin=(100.0, 50.0),
        group_offset=50.0,
        shot_offset=100.0,
        gather="common shot",
        smooth=2,
        engine="pylops"
    )
    
    shot_rec.set_model(vp)
    
    # Run a short simulation
    data = shot_rec.run(ms=100)
    
    # Check that output is not None and has the correct shape (n_sources, n_receivers, n_time)
    assert data is not None
    assert len(data.shape) == 3
    assert data.shape[0] == 2
    assert data.shape[1] == 6
    assert not np.isnan(data).any()
