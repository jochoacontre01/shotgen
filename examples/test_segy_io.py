import numpy as np
import os
from shotgen import ShotRecord, SegyIO

def test_segy_io():
    # 1. Setup a simple ShotRecord
    nx, nz = 50, 50
    dx, dz = 10, 10
    n_receivers = 10
    n_sources = 2
    
    shot_rec = ShotRecord(
        nx=nx, nz=nz, dx=dx, dz=dz,
        n_receivers=n_receivers,
        n_sources=n_sources,
        gather="common shot",
        group_offset=10,
        shot_offset=20,
        origin=(0, 0),
        engine="pylops"
    )
    
    # 2. Set dummy model and run simulation
    dummy_vel = np.ones((nx, nz)) * 2000.0 # 2000 m/s constant velocity
    shot_rec.set_model(dummy_vel)
    print("Running simulation...")
    shot_rec.run(ms=100) # Short simulation
    
    # 3. Write SEGY
    test_filepath = "test_output.segy"
    print(f"Writing to {test_filepath}...")
    SegyIO.write(test_filepath, shot_rec)
    
    # 4. Read SEGY
    print(f"Reading from {test_filepath}...")
    segy_data = SegyIO.read(test_filepath)
    
    # 5. Verify
    n_traces_expected = n_sources * n_receivers
    assert segy_data["data"].shape[0] == n_traces_expected, "Trace count mismatch"
    assert segy_data["data"].shape[1] == shot_rec.shot_run.shape[2], "Time sample count mismatch"
    
    # Validate trace 0 data
    trace_0_expected = shot_rec.shot_run[0, 0, :]
    trace_0_actual = segy_data["data"][0, :]
    np.testing.assert_allclose(trace_0_actual, trace_0_expected, rtol=1e-5, err_msg="Trace data mismatch")
    
    # Validate first CDP geometry
    # source 0 is at src_origin=0 (by default), 1st rec is at 0
    s0x = segy_data["sources"][0, 0]
    r0x = segy_data["receivers"][0, 0]
    cdp0 = segy_data["cdp"][0]
    dt = segy_data["dt"]
    
    print("Test passed! Verification successful.")
    print("Geometry snapshot trace 0:")
    print(f"  Source X: {s0x}")
    print(f"  Group X: {r0x}")
    print(f"  CDP: {cdp0}")
    print(f"  dt: {dt} s")
    
    # Cleanup
    if os.path.exists(test_filepath):
        os.remove(test_filepath)

if __name__ == "__main__":
    test_segy_io()
