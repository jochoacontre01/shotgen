import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from shotgen_gpu.config import configure_devito_device
from shotgen_gpu.engine.solver import AcousticWaveSolverWrapper
from shotgen.io import SegyIO
from shotgen.sampleshot import (
    load_marmousi,
    load_bpsalt,
    load_overthrust,
    load_sigsbee,
    _find_assets_dir,
    ShotRecord
)


def parse_velocity_model(cfg: dict):
    """
    Loads velocity model from string identifier ('marmousi', 'bpsalt', 'overthrust', 'sigsbee')
    or file path string (.segy, .sgy, .npy).
    """
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")

    default_dx = float(cfg.get("dx", 10.0))
    default_dz = float(cfg.get("dz", 10.0))
    default_nx = int(cfg.get("nx", 100))
    default_nz = int(cfg.get("nz", 100))

    if isinstance(vel_spec, str):
        name_lower = vel_spec.lower().strip()
        benchmark_map = {
            "marmousi": load_marmousi,
            "bpsalt": load_bpsalt,
            "overthrust": load_overthrust,
            "sigsbee": load_sigsbee,
        }

        if name_lower in benchmark_map:
            res = benchmark_map[name_lower]()
            if isinstance(res, tuple) and len(res) >= 2:
                vel = res[0]
                meta = res[1] if isinstance(res[1], dict) else {}
            else:
                vel = res
                meta = {}

            nx_val, nz_val = vel.shape
            dx = float(cfg.get("dx", meta.get("dx", default_dx)))
            dz = float(cfg.get("dz", meta.get("dz", default_dz)))
            return vel, nx_val, nz_val, dx, dz

        filepath = Path(vel_spec)
        if not filepath.exists():
            assets_file = _find_assets_dir() / vel_spec
            if assets_file.exists():
                filepath = assets_file
            else:
                raise FileNotFoundError(f"Velocity model file or benchmark '{vel_spec}' not found.")

        if filepath.suffix.lower() in [".segy", ".sgy"]:
            vel, dx_m, dz_m = SegyIO.read_model(str(filepath))
            nx_val, nz_val = vel.shape
            dx = float(cfg.get("dx", dx_m))
            dz = float(cfg.get("dz", dz_m))
            nx = int(cfg.get("nx", nx_val))
            nz = int(cfg.get("nz", nz_val))
            return vel, nx, nz, dx, dz
        elif filepath.suffix.lower() == ".npy":
            vel = np.load(str(filepath))
            nx_val, nz_val = vel.shape
            return vel, nx_val, nz_val, default_dx, default_dz

    # Fallback synthetic model
    v_base = float(cfg.get("v_base", 2500.0))
    vel = v_base * np.ones((default_nx, default_nz), dtype=np.float32)
    vel[:, default_nz // 2:] = v_base + 500.0
    return vel, default_nx, default_nz, default_dx, default_dz


def main():
    parser = argparse.ArgumentParser(description="shotrecord.py: Generate seismic shot records from config.json")
    parser.add_argument("--config", "-c", type=str, default="examples/config.json", help="Path to configuration JSON file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    device = cfg.get("device", "auto")
    configured_device = configure_devito_device(device)
    print(f"[shotrecord.py] Running simulation on device: {configured_device.upper()}")

    vel, nx, nz, dx, dz = parse_velocity_model(cfg)
    n_sources = int(cfg.get("n_sources", 4))
    n_receivers = int(cfg.get("n_receivers", 20))
    f0 = float(cfg.get("f0", 25.0))
    ms = float(cfg.get("ms", cfg.get("ntime", 300.0)))
    fd_order = int(cfg.get("fd_order", 4))
    n_damping = int(cfg.get("n_damping", 40))
    smooth = float(cfg.get("smooth", 5.0))
    origin = tuple(cfg.get("origin", [0.0, 0.0]))

    if "sources" in cfg:
        sources = np.array(cfg["sources"], dtype=np.float32)
    else:
        sx = origin[0] + np.linspace(dx * 2, (nx - 2) * dx, n_sources, dtype=np.float32)
        sz = np.ones(n_sources, dtype=np.float32) * origin[1]
        sources = np.vstack([sx, sz]).T

    if "receivers" in cfg:
        receivers = np.array(cfg["receivers"], dtype=np.float32)
    else:
        rx = origin[0] + np.linspace(dx, (nx - 1) * dx, n_receivers, dtype=np.float32)
        rz = np.ones(n_receivers, dtype=np.float32) * origin[1]
        receivers = np.vstack([rx, rz]).T

    print(f"[shotrecord.py] Grid size: ({nx}, {nz}), dx={dx}m, dz={dz}m | {n_sources} sources, {n_receivers} receivers")

    solver = AcousticWaveSolverWrapper(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        vel=vel,
        sources=sources,
        receivers=receivers,
        f0=f0,
        origin=origin,
        fd_order=fd_order,
        n_damping=n_damping,
        smooth=smooth,
        device=configured_device,
    )

    shot_run, us, wavelet = solver.run(ms=ms, save_wavefield=cfg.get("save_wavefield", False))

    output_dir = Path(cfg.get("output_dir", "data/example_simulation_output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rec_actual = receivers.shape[1] if receivers.ndim == 3 else len(receivers)
    n_src_actual = len(sources)

    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_receivers=n_rec_actual,
        n_sources=n_src_actual,
        f0=f0,
        origin=origin,
        smooth=smooth,
        engine="devito",
        device=configured_device,
    )
    shot_rec.vel = vel
    shot_rec.v0 = gaussian_filter(vel, sigma=smooth)
    shot_rec.shot_run = shot_run
    shot_rec.sources = sources
    shot_rec.recs = receivers
    shot_rec.src = wavelet
    shot_rec.time_vector = np.linspace(0, ms / 1000.0, shot_run.shape[-1])
    shot_rec.save_shot(str(output_dir))

    # Save output to HDF5 container for backwards compatibility
    out_h5 = output_dir / "simulation_results.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("shots", data=shot_run)
        f.create_dataset("sources", data=sources)
        f.create_dataset("receivers", data=receivers)
        f.create_dataset("vel", data=vel)
        if wavelet is not None:
            f.create_dataset("wavelet", data=wavelet)
        f.create_dataset("nx", data=nx)
        f.create_dataset("nz", data=nz)
        f.create_dataset("dx", data=dx)
        f.create_dataset("dz", data=dz)
        f.create_dataset("ms", data=ms)
        f.create_dataset("f0", data=f0)

    print(f"[shotrecord.py] Wavefield simulation complete! Records saved to: {output_dir}")


if __name__ == "__main__":
    main()
