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
    vp_full, _ = load_marmousi()
    
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
    
    # Verify migrator.model exists and matches expected properties
    assert migrator.model is not None
    assert migrator.model.shape == migrator.vp.shape
    assert migrator.model.origin == (0.0, 0.0)
    
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
        device="cpu",
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


def test_origin_immutability():
    """Asserts that providing/setting a second 'origin' value to an already instantiated

    ShotRecord does not change the initial origin.
    """
    shot_rec = ShotRecord(
        nx=50, nz=40, dx=10.0, dz=10.0,
        n_receivers=5, n_sources=1,
        origin=(100.0, 200.0),
        engine="pylops"
    )
    assert shot_rec.origin == (100.0, 200.0)
    
    # Attempt to change origin directly via attribute assignment
    shot_rec.origin = (500.0, 600.0)
    assert shot_rec.origin == (100.0, 200.0)  # Should remain unchanged


def test_spatial_boundary_enforcement():
    """Asserts that ShotRecord raises a ValueError when given geometry

    that falls outside the model bounds.
    """
    # 1. Source falls outside bounds (x coordinate exceeds nx * dx)
    with pytest.raises(ValueError, match="Source coordinate X is outside model bounds"):
        ShotRecord(
            nx=50, nz=40, dx=10.0, dz=10.0,
            n_receivers=5, n_sources=1,
            src_origin=(600.0, 20.0),  # Max valid x is 50 * 10 = 500
            origin=(0, 0),
            engine="pylops"
        )

    # 2. Receiver falls outside bounds (z coordinate exceeds nz * dz)
    with pytest.raises(ValueError, match="Receiver coordinate Z is outside model bounds"):
        ShotRecord(
            nx=50, nz=40, dx=10.0, dz=10.0,
            n_receivers=5, n_sources=1,
            rec_origin=(10.0, 500.0),  # Max valid z is 40 * 10 = 400
            origin=(0, 0),
            engine="pylops"
        )


def test_origin_persistence_and_roundtrip(tmp_path):
    """Verifies that an arbitrary origin (x, z) provided to ShotRecord is

    correctly written to and read back from h5/segy files, and the full
    round-trip migration uses the intended origin without user intervention.
    """
    dataset_dir = os.path.join(tmp_path, "roundtrip_dataset")
    arbitrary_origin = (350.0, 450.0)
    
    # 1. Generate record with arbitrary origin
    nx, nz = 50, 40
    dx, dz = 10.0, 10.0
    shot_rec = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz,
        n_receivers=4, n_sources=2,
        f0=25.0,
        src_origin=(50.0, 10.0),
        rec_origin=(50.0, 10.0),
        group_offset=30.0,
        shot_offset=50.0,
        origin=arbitrary_origin,
        engine="pylops"
    )
    dummy_vel = np.ones((nx, nz)) * 2000.0
    dummy_vel[:, nz//2:] = 2500.0  # Layered model with reflector
    shot_rec.set_model(dummy_vel)
    shot_rec.run(ms=150)
    shot_rec.save_shot(dataset_dir)
    
    # 2. Initialize Kirchhoff PSDM directly using the dataset directory (no origin parameter passed)
    migrator = KirchhoffMigration(
        dataset_dir=dataset_dir
    )
    
    # 3. Assert origin was correctly read back and anchors geometry
    assert migrator.origin == arbitrary_origin
    assert migrator.model.origin == arbitrary_origin
    
    # Check that model shape matches grid dimensions
    assert migrator.model.shape == (nx, nz)
    
    # Verify migration runs successfully and anchors geometry
    image = migrator.run()
    assert image is not None
    assert image.shape == (nx, nz)
    assert np.any(image != 0.0)
    assert np.max(np.abs(image)) > 1e-12
    
    # 4. Check RTM parsed origin read-back and successful run
    rtm = ReverseTimeMigration(
        dataset_dir=dataset_dir,
        nbl=5,
        space_order=2,
        device="cpu",
    )
    
    # Note: RTM reads origin from dataset_dir if not overridden via kwargs
    assert rtm.origin == arbitrary_origin
    assert rtm.model.origin == arbitrary_origin
    
    rtm_image = rtm.run(save_wavefield=False)
    assert rtm_image is not None
    assert np.any(rtm_image != 0.0)
    assert np.max(np.abs(rtm_image)) > 1e-12


def test_rtm_source_coordinates_updating(small_migration_dataset):
    """Verifies that the RTM loop correctly updates the underlying Devito

    geometry.src.coordinates.data buffer with the moving source locations.
    """
    nbl = 10
    rtm = ReverseTimeMigration(
        dataset_dir=small_migration_dataset,
        nbl=nbl,
        smooth_sigma=3.0,
        space_order=2,
        device="cpu",
    )
    
    original_forward = rtm.solver.forward
    recorded_coords = []
    
    def dummy_forward(*args, **kwargs):
        # Record the current source coordinates from the Devito geometry object
        current_coord = rtm.geometry.src.coordinates.data.copy()
        recorded_coords.append(current_coord)
        return original_forward(*args, **kwargs)
        
    rtm.solver.forward = dummy_forward
    
    # Run the migration
    rtm.run(save_wavefield=False)
    
    # Check that we have recorded coordinates for each source
    assert len(recorded_coords) == rtm.n_sources
    for i in range(rtm.n_sources):
        # The coordinates in recorded_coords should match rtm.sources[i, :]
        assert np.allclose(recorded_coords[i][0], rtm.sources[i], atol=1e-3)

