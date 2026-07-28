import numpy as np
import pytest
import pathlib
from shotgen.sampleshot import load_overthrust, ShotRecord, _find_assets_dir

overthrust_file = _find_assets_dir() / "marine_overthrust_3d.segy"
skip_if_no_overthrust = pytest.mark.skipif(
    not overthrust_file.exists(),
    reason="marine_overthrust_3d.segy asset not present"
)


@skip_if_no_overthrust
def test_load_overthrust():
    vp, _ = load_overthrust()
    assert vp.shape == (801, 185)
    assert vp.min() >= 1480.0
    assert vp.max() <= 4000.0
    assert np.allclose(vp[:, 0], 1480.0)


@skip_if_no_overthrust
def test_shotrecord_with_overthrust():
    vp_full, _ = load_overthrust()
    vp = vp_full[300:350, :40]

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
    data = shot_rec.run(ms=100)
    assert data is not None
    assert len(data.shape) == 3
    assert data.shape[0] == 2
    assert data.shape[1] == 6
    assert not np.isnan(data).any()
