import numpy as np
import pytest
from shotgen.sampleshot import ShotRecord
from shotgen.models import GeoModel


def test_shotrecord_nonempty_simulation_pylops():
    nx, nz = 100, 80
    dx, dz = 10.0, 10.0
    geo = GeoModel(nx=nx, nz=nz, v_base=2500.0)
    vp = geo.layered()

    shot = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=2,
        n_receivers=10,
        f0=25.0,  # Hz
        src_origin=(100.0, 10.0),
        rec_origin=(50.0, 10.0),
        group_offset=20.0,
        shot_offset=40.0,
        gather="common shot",
        smooth=5,
        engine="pylops",
        device="auto"
    )
    shot.set_model(vp)
    data = shot.run(ms=300.0)

    assert data is not None, "Shot simulation returned None"
    assert data.size > 0, "Shot simulation returned an empty array"
    assert not np.all(data == 0.0), "Shot simulation returned an all-zero array"
    assert np.max(np.abs(data)) > 1e-15, f"Shot simulation signal magnitude too small: {np.max(np.abs(data))}"


def test_shotrecord_nonempty_simulation_devito():
    nx, nz = 100, 80
    dx, dz = 10.0, 10.0
    geo = GeoModel(nx=nx, nz=nz, v_base=2500.0)
    vp = geo.layered()

    shot = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=2,
        n_receivers=10,
        f0=25.0,  # Hz
        src_origin=(100.0, 10.0),
        rec_origin=(50.0, 10.0),
        group_offset=20.0,
        shot_offset=40.0,
        gather="common shot",
        smooth=5,
        engine="devito",
        device="auto"
    )
    shot.set_model(vp)
    data = shot.run(ms=300.0)

    assert data is not None, "Shot simulation returned None"
    assert data.size > 0, "Shot simulation returned an empty array"
    assert not np.all(data == 0.0), "Shot simulation returned an all-zero array"
    assert np.max(np.abs(data)) > 1e-15, f"Shot simulation signal magnitude too small: {np.max(np.abs(data))}"
