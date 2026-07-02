import os
import pytest
import numpy as np
from shotgen.sampleshot import ShotRecord, load_marmousi
from shotgen.migration import KirchhoffMigration, ReverseTimeMigration, load_dataset_dir

@pytest.fixture(scope="module")
def small_migration_dataset(tmp_path_factory):
    """Generates a small dataset using a sliced Marmousi model and ShotRecord."""
    # Create a unique temporary directory for this module's run
    dataset_dir = tmp_path_factory.mktemp("test_migration_dataset")
    
    # 1. Load Marmousi model and slice it to be small
    vp_full = load_marmousi()
    
    # Slice a small segment of Marmousi
    # Shape nx = 60, nz = 40, spacing = 10m
    dx_val = 10.0
    vp = vp_full[3000:3600:int(dx_val), 250:650:int(dx_val)]
    nx, nz = vp.shape
    
    # 2. Configure a small acquisition geometry
    # We want a small number of sources (2) and receivers (6)
    # The source and receiver origins should fit well within the model
    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx_val,
        dz=dx_val,
        n_sources=2,
        n_receivers=6,
        f0=30.0,
        src_origin=(100.0, 20.0),
        rec_origin=(100.0, 20.0),
        group_offset=40.0,  # 6 receivers * 40m = 240m span
        shot_offset=100.0,  # 2 sources * 100m = 200m span
        gather="common shot",
        smooth=5,
        snr=10,
        fd_order=4,         # Space order 4
        n_damping=20,
        engine="pylops"
    )
    
    shot_rec.set_model(vp)
    
    # Run simulation for a short time (e.g. 150 ms)
    shot_rec.run(ms=150)
    
    # Save the generated shot to the temporary directory
    shot_rec.save_shot(str(dataset_dir))
    
    return str(dataset_dir)


def test_kirchhoff_migration_success(small_migration_dataset):
    """Tests if KirchhoffMigration successfully processes the generated dataset directly

    from the directory and yields a non-empty image matching the model dimensions.
    """
    # Initialize the Kirchhoff PSDM migrator directly using the dataset directory
    migrator = KirchhoffMigration(
        dataset_dir=small_migration_dataset
    )
    
    # Verify properties loaded successfully
    assert migrator.vp is not None
    assert migrator.shots is not None
    assert migrator.time is not None
    
    # Run the Kirchhoff migration
    image = migrator.run()
    
    # 1. Output is not None and matches shape of velocity model
    assert image is not None
    assert image.shape == migrator.vp.shape
    
    # 2. Image is non-empty (has non-zero values due to reflections)
    assert np.any(image != 0.0)
    assert not np.all(np.isnan(image))
    
    # Verify some quantitative non-emptiness
    print(f"Kirchhoff Image shape: {image.shape}, min: {image.min()}, max: {image.max()}")
    assert np.max(np.abs(image)) > 1e-12


def test_rtm_migration_success(small_migration_dataset):
    """Tests if ReverseTimeMigration successfully processes the generated dataset directly

    from the directory and yields a non-empty image matching the model dimensions after cropping.
    """
    nbl = 10
    
    # Initialize ReverseTimeMigration directly using the dataset directory
    rtm = ReverseTimeMigration(
        dataset_dir=small_migration_dataset,
        nbl=nbl,
        smooth_sigma=3.0,
        space_order=2, # Fast order for testing
    )
    
    # Verify properties loaded successfully
    assert rtm.vp is not None
    assert rtm.shots is not None
    
    # Run Reverse Time Migration (alignment is handled internally in the class now)
    migrated_image = rtm.run(save_wavefield=False)
    
    # 1. Output image is not None
    assert migrated_image is not None
    
    # 2. Crop boundary layers (nbl) to compare shape
    plotted_image = migrated_image[nbl:-nbl, nbl:-nbl]
    assert plotted_image.shape == rtm.vp.shape
    
    # 3. Image is non-empty (has non-zero values due to reflections)
    assert np.any(plotted_image != 0.0)
    assert not np.all(np.isnan(plotted_image))
    
    # Validate non-emptiness
    print(f"RTM Image shape: {plotted_image.shape}, min: {plotted_image.min()}, max: {plotted_image.max()}")
    assert np.max(np.abs(plotted_image)) > 1e-12
