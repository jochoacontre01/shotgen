import numpy as np
import pytest
from shotgen.sampleshot import ShotRecord

def test_cmp_parallel_simulation():
    nx, nz = 50, 40
    dx, dz = 10.0, 10.0
    n_sources = 4
    n_receivers = 8
    ms = 100.0

    sr = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=n_sources,
        n_receivers=n_receivers,
        gather="common midpoint",
        engine="pylops"
    )

    vel_model = np.ones((nx, nz), dtype=np.float32) * 2000.0
    vel_model[:, 20:] = 2500.0
    sr.set_model(vel_model)

    shot_run = sr.run(ms=ms)

    # Check that output shape is (n_sources, n_receivers, n_time)
    assert shot_run.ndim == 3
    assert shot_run.shape[0] == n_sources
    assert shot_run.shape[1] == n_receivers

    # Check metadata properties
    assert sr.aop is not None
    assert sr.src is not None
    assert sr.dt is not None
    assert sr.shot_run is not None
