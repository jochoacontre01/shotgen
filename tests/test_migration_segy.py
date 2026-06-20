import os
import shutil
import pytest
import numpy as np
from shotgen.sampleshot import ShotRecord
from shotgen.migration import KirchhoffMigration, ReverseTimeMigration, load_dataset_dir

@pytest.fixture
def dummy_dataset_dir(tmpdir):
    """Creates a small dummy dataset in a temporary directory."""
    dataset_dir = os.path.join(tmpdir, "test_migration_dataset")
    
    nx, nz = 50, 50
    dx, dz = 10.0, 10.0
    
    shot_rec = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz,
        n_receivers=5,
        n_sources=1,
        gather="common shot",
        origin=(0, 0),
        engine="pylops"
    )
    dummy_vel = np.ones((nx, nz)) * 2000.0
    shot_rec.set_model(dummy_vel)
    
    # Run short simulation
    shot_rec.run(ms=10)
    
    shot_rec.save_shot(dataset_dir)
    return dataset_dir

def test_kirchhoff_migration_segy(dummy_dataset_dir):
    """Tests if KirchhoffMigration properly loads SEGY dataset directly."""
    migrator = KirchhoffMigration(
        dataset_dir=dummy_dataset_dir,
        spacing=(10.0, 10.0)
    )
    
    assert migrator.vp is not None
    assert migrator.shots is not None
    assert migrator.shots.shape == (1, 5, migrator.time.shape[0])
    
    image = migrator.run()
    assert image.shape == migrator.vp.shape

def test_reverse_time_migration_segy(dummy_dataset_dir):
    """Tests if ReverseTimeMigration properly loads SEGY dataset directly."""
    rtm = ReverseTimeMigration(
        dataset_dir=dummy_dataset_dir,
        spacing=(10.0, 10.0),
        nbl=5,
        space_order=2,
    )
    
    assert rtm.vp is not None
    assert rtm.shots is not None
    assert rtm.f0 == 0.025  # 25 Hz converted to kHz in Devito
    
    # Do a quick run
    image = rtm.run(save_wavefield=False)
    assert image is not None
    
def test_edge_case_missing_velocity(dummy_dataset_dir):
    """Tests error handling when velocity files are missing."""
    v0_path = os.path.join(dummy_dataset_dir, "smooth_velocity.segy")
    vel_path = os.path.join(dummy_dataset_dir, "velocity_model.segy")
    
    os.remove(v0_path)
    os.remove(vel_path)
    
    with pytest.raises(FileNotFoundError, match="Neither smooth_velocity.segy nor velocity_model.segy found"):
        load_dataset_dir(dummy_dataset_dir, require_f0=False)

def test_edge_case_missing_f0(dummy_dataset_dir):
    """Tests error handling when f0 is missing and required."""
    meta_path = os.path.join(dummy_dataset_dir, "metadata.h5")
    
    # Delete metadata to force f0 missing
    os.remove(meta_path)
    
    with pytest.raises(ValueError, match="f0 is required for RTM"):
        load_dataset_dir(dummy_dataset_dir, require_f0=True)
