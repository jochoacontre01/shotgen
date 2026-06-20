import os
import shutil
import numpy as np
from shotgen.sampleshot import ShotRecord, LoadShotRecord

def test_save_load():
    test_folder = "test_dataset"
    
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
        
    nx, nz = 50, 50
    dx, dz = 10.0, 10.0
    
    # 1. Create a ShotRecord
    shot_rec = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz,
        n_receivers=10,
        n_sources=2,
        gather="common shot",
        group_offset=10,
        shot_offset=20,
        origin=(0, 0),
        engine="pylops"
    )
    
    dummy_vel = np.ones((nx, nz)) * 2000.0
    shot_rec.set_model(dummy_vel)
    
    print("Running simulation...")
    shot_rec.run(ms=100)
    
    # 2. Save it
    print(f"Saving to {test_folder}...")
    shot_rec.save_shot(test_folder)
    
    # 3. Load it
    print(f"Loading from {test_folder}...")
    loaded_data = LoadShotRecord(test_folder)
    
    # 4. Verify
    assert loaded_data.shots.shape == shot_rec.shot_run.shape, "Shots shape mismatch"
    assert loaded_data.velocity_model is not None, "Velocity model not loaded"
    assert loaded_data.smooth_velocity is not None, "Smooth velocity not loaded"
    assert np.allclose(loaded_data.velocity_model, shot_rec.vel), "Velocity model mismatch"
    
    np.testing.assert_allclose(loaded_data.shots, shot_rec.shot_run, rtol=1e-5, err_msg="Trace data mismatch")
    
    print("Test passed! Directory saving and loading is working perfectly.")
    
    # Cleanup
    shutil.rmtree(test_folder)

if __name__ == "__main__":
    test_save_load()
