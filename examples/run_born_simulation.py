import os
import json
from pathlib import Path
from shotgen.runner import run_gpu_simulation
from shotgen import ShotRecord, GeoModel


def main():
    print("=== Running Born Wavefield Simulation ===")

    # 1. Option A: Run via shotgen-core ShotRecord (which bridges to shotgen-gpu via isolated subprocess)
    nx, nz = 80, 50
    dx, dz = 10.0, 10.0
    geo = GeoModel(nx=nx, nz=nz, v_base=2500.0)
    vp = geo.layered()

    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=2,
        n_receivers=8,
        f0=25.0,
        src_origin=(50.0, 0.0),
        rec_origin=(10.0, 0.0),
        group_offset=20.0,
        shot_offset=40.0,
        gather="common shot",
        smooth=5,
        engine="devito",
        device="auto"
    )
    shot_rec.set_model(vp)
    shots = shot_rec.run(ms=150)
    print(f"Born simulation completed via ShotRecord (engine=devito). Output shape: {shots.shape}")

    # 2. Option B: Run via shotgen-gpu CLI directly using test_config.json
    config_file = Path(__file__).resolve().parent / "test_config.json"
    if config_file.exists():
        print(f"Launching shotgen-gpu CLI using config: {config_file}")
        run_gpu_simulation(str(config_file))


if __name__ == "__main__":
    main()
