import os
import sys
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter

from shotgen_gpu.config import configure_devito_device, detect_device
from shotgen_gpu.engine.solver import AcousticWaveSolverWrapper
from shotgen_gpu.engine.rtm import ReverseTimeMigration
from shotgen.io import SegyIO
from shotgen.models import GeoModel, resample_velocity_model
from shotgen.sampleshot import (
    load_marmousi,
    load_bpsalt,
    load_overthrust,
    load_sigsbee,
    _find_assets_dir,
    ShotRecord
)
from shotgen.utils import generate_simulation_dir_name


def _load_velocity_model(cfg: dict):
    """
    Parses the velocity model from the configuration dictionary.
    Supports benchmark model string names ('marmousi', 'bpsalt', 'overthrust', 'sigsbee')
    as well as file path strings (.segy, .sgy, .npy).
    Automatically resamples the model to target dx and dz specified in config if different from original.
    """
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file") or cfg.get("vel_model") or cfg.get("model")

    target_dx = cfg.get("dx")
    target_dz = cfg.get("dz")
    if target_dx is not None:
        target_dx = float(target_dx)
    if target_dz is not None:
        target_dz = float(target_dz)

    default_dx = float(target_dx) if target_dx is not None else 10.0
    default_dz = float(target_dz) if target_dz is not None else 10.0
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

            dx_orig = float(meta.get("dx", default_dx))
            dz_orig = float(meta.get("dz", default_dz))

            if target_dx is not None or target_dz is not None:
                vel, nx_new, nz_new, dx_out, dz_out = resample_velocity_model(
                    vel, dx_orig, dz_orig, target_dx=target_dx, target_dz=target_dz
                )
                return vel, nx_new, nz_new, dx_out, dz_out

            nx_val, nz_val = vel.shape
            return vel, nx_val, nz_val, dx_orig, dz_orig

        # Check if vel_spec is a file path
        filepath = Path(vel_spec)
        if not filepath.exists():
            assets_file = _find_assets_dir() / vel_spec
            if assets_file.exists():
                filepath = assets_file
            else:
                raise FileNotFoundError(f"Velocity model file or benchmark '{vel_spec}' not found.")

        if filepath.suffix.lower() in [".segy", ".sgy"]:
            vel, dx_m, dz_m = SegyIO.read_model(str(filepath))
            if target_dx is not None or target_dz is not None:
                vel, nx_new, nz_new, dx_out, dz_out = resample_velocity_model(
                    vel, dx_m, dz_m, target_dx=target_dx, target_dz=target_dz
                )
                return vel, nx_new, nz_new, dx_out, dz_out

            nx_val, nz_val = vel.shape
            return vel, nx_val, nz_val, dx_m, dz_m
        elif filepath.suffix.lower() == ".npy":
            vel = np.load(str(filepath))
            dx_orig = float(cfg.get("dx_orig", default_dx))
            dz_orig = float(cfg.get("dz_orig", default_dz))
            if target_dx is not None or target_dz is not None:
                vel, nx_new, nz_new, dx_out, dz_out = resample_velocity_model(
                    vel, dx_orig, dz_orig, target_dx=target_dx, target_dz=target_dz
                )
                return vel, nx_new, nz_new, dx_out, dz_out

            nx_val, nz_val = vel.shape
            return vel, nx_val, nz_val, dx_orig, dz_orig

    # Fallback synthetic model
    v_base = float(cfg.get("v_base", 2500.0))
    vel = v_base * np.ones((default_nx, default_nz), dtype=np.float32)
    vel[:, default_nz // 2:] = v_base + 500.0
    return vel, default_nx, default_nz, default_dx, default_dz



def main():
    parser = argparse.ArgumentParser(description="shotgen-gpu: Devito/OpenACC wavefield propagation & RTM CLI")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to JSON configuration file")
    parser.add_argument("--cli", action="store_true", help="Display the first shot record on the terminal using chafa after simulation")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    device = cfg.get("device", "auto")
    configured_device = configure_devito_device(device)
    print(f"[shotgen-gpu CLI] Configured device: {configured_device.upper()}")

    task_type = cfg.get("task", "simulation")

    cli_enabled = args.cli or cfg.get("cli", False)

    if task_type == "simulation":
        _run_simulation_cli(cfg, cli_enabled=cli_enabled)
    elif task_type == "rtm":
        _run_rtm_cli(cfg)
    else:
        print(f"Error: Unknown task type '{task_type}' in config", file=sys.stderr)
        sys.exit(1)


def _run_simulation_cli(cfg: dict, cli_enabled: bool = False):
    vel, nx, nz, dx, dz = _load_velocity_model(cfg)

    n_sources = int(cfg.get("n_sources", 2))
    n_receivers = int(cfg.get("n_receivers", 10))
    f0 = float(cfg.get("f0", 25.0))
    ms = float(cfg.get("ms", 300.0))
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
        device=cfg.get("device", "auto"),
    )

    shot_run, us, wavelet = solver.run(ms=ms, save_wavefield=cfg.get("save_wavefield", False))

    cfg.update({"nx": nx, "nz": nz, "dx": dx, "dz": dz, "n_sources": n_sources, "n_receivers": n_receivers, "f0": f0, "ms": ms})
    if "output_dir" in cfg:
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = generate_simulation_dir_name(cfg, base_dir="data")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rec_actual = receivers.shape[1] if receivers.ndim == 3 else len(receivers)
    n_src_actual = len(sources)

    fs_ms = cfg.get("fs_ms", None)
    if fs_ms is not None:
        fs_ms = float(fs_ms)

    # Save SEGY and metadata using SegyIO via ShotRecord
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
        device=cfg.get("device", "auto"),
        fs_ms=fs_ms,
    )
    shot_rec.vel = vel
    shot_rec.v0 = gaussian_filter(vel, sigma=smooth)
    shot_rec.shot_run = shot_run

    if fs_ms is not None and fs_ms > 0:
        new_nt = int(np.round(ms / fs_ms)) + 1
        t_orig = np.linspace(0, ms, shot_run.shape[-1])
        t_new = np.linspace(0, ms, new_nt)
        from scipy.interpolate import interp1d
        f_interp = interp1d(t_orig, shot_run, axis=-1, kind="cubic", fill_value="extrapolate")
        resampled_shot_run = f_interp(t_new).astype(shot_run.dtype)
        del shot_run
        import gc
        gc.collect()
        shot_run = resampled_shot_run
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
        if fs_ms is not None:
            f.create_dataset("fs_ms", data=fs_ms)
        f.create_dataset("dx", data=dx)
        f.create_dataset("dz", data=dz)
        f.create_dataset("ms", data=ms)
        f.create_dataset("f0", data=f0)

    print(f"[shotgen-gpu CLI] Wavefield simulation complete! SEGY and HDF5 written to: {output_dir}")

    if cli_enabled:
        shot_rec.show_shot(cli=True, source_idx=0)



def _run_rtm_cli(cfg: dict):
    dataset_dir = cfg.get("dataset_dir")
    nbl = cfg.get("nbl", 40)
    space_order = cfg.get("space_order", 4)
    device = cfg.get("device", "auto")

    rtm = ReverseTimeMigration(
        dataset_dir=dataset_dir,
        nbl=nbl,
        space_order=space_order,
        device=device,
    )
    image = rtm.run(save_wavefield=False)

    output_dir = Path(cfg.get("output_dir", "data/gpu_rtm_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "rtm_image.npy"
    np.save(out_file, image)
    print(f"[shotgen-gpu CLI] RTM complete! Image written to: {out_file}")


if __name__ == "__main__":
    main()

