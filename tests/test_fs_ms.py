import os
import shutil
import tempfile
import numpy as np
import pytest
from shotgen import ShotRecord, SegyIO
from shotgen.migration.kirchhoff import KirchhoffMigration, load_dataset_dir


def test_fs_ms_resampling_and_segy_header():
    nx, nz = 40, 40
    dx, dz = 10.0, 10.0
    n_receivers = 8
    n_sources = 2
    ms = 100.0
    fs_ms = 2.0  # 2 ms sampling rate

    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_receivers=n_receivers,
        n_sources=n_sources,
        gather="common shot",
        group_offset=10.0,
        shot_offset=20.0,
        origin=(0.0, 0.0),
        engine="pylops",
        fs_ms=fs_ms,
    )

    dummy_vel = np.ones((nx, nz)) * 2000.0
    shot_rec.set_model(dummy_vel)
    shot_rec.run(ms=ms)

    # Check resampled trace length: 100 ms / 2 ms + 1 = 51 samples
    expected_samples = int(np.round(ms / fs_ms)) + 1
    assert shot_rec.shot_run.shape[-1] == expected_samples, f"Expected {expected_samples} samples, got {shot_rec.shot_run.shape[-1]}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        shot_rec.save_shot(tmp_dir)

        traces_segy = os.path.join(tmp_dir, "traces.segy")
        segy_data = SegyIO.read(traces_segy)

        # Verify SEG-Y dt header in seconds (2 ms -> 0.002 s)
        assert np.isclose(segy_data["dt"], fs_ms / 1000.0), f"Expected dt={fs_ms/1000.0}, got {segy_data['dt']}"
        assert segy_data["data"].shape[1] == expected_samples, f"Expected {expected_samples} samples in SEG-Y, got {segy_data['data'].shape[1]}"

        # Test Kirchhoff migration script on the resampled dataset
        migrator = KirchhoffMigration(dataset_dir=tmp_dir)
        mig_image = migrator.run()

        assert mig_image is not None, "Kirchhoff migration returned None"
        assert mig_image.shape == (nx, nz), f"Migrated image shape mismatch: {mig_image.shape} vs ({nx}, {nz})"
        assert np.any(mig_image != 0), "Kirchhoff migration returned an all-zero array"


def test_fs_ms_null_behavior():
    nx, nz = 30, 30
    dx, dz = 10.0, 10.0
    ms = 50.0

    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_receivers=5,
        n_sources=2,
        engine="pylops",
        fs_ms=None,
    )
    dummy_vel = np.ones((nx, nz)) * 2000.0
    shot_rec.set_model(dummy_vel)
    shot_rec.run(ms=ms)

    assert shot_rec.fs_ms is None
    dt_orig = shot_rec.dt
    expected_dt_s = dt_orig

    with tempfile.TemporaryDirectory() as tmp_dir:
        shot_rec.save_shot(tmp_dir)
        segy_data = SegyIO.read(os.path.join(tmp_dir, "traces.segy"))
        assert np.isclose(segy_data["dt"], expected_dt_s, rtol=1e-4)


if __name__ == "__main__":
    test_fs_ms_resampling_and_segy_header()
    test_fs_ms_null_behavior()
    print("All fs_ms tests passed successfully!")
