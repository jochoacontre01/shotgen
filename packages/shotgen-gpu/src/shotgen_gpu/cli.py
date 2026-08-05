import os
import sys
import json
import argparse
import time
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
    parser.add_argument("-v", "--verbose", action="store_true", help="Display GPU runtime and verbose simulation output when complete")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r") as f:
        cfg = json.load(f)

    verbose_enabled = args.verbose or cfg.get("verbose", False)

    device = cfg.get("device", "auto")
    configured_device = configure_devito_device(device, verbose=verbose_enabled)
    print(f"[shotgen-gpu CLI] Configured device: {configured_device.upper()}")

    task_type = cfg.get("task", "simulation")

    cli_enabled = args.cli or cfg.get("cli", False)

    if task_type == "simulation":
        _run_simulation_cli(cfg, cli_enabled=cli_enabled, verbose=verbose_enabled)
    elif task_type == "rtm":
        _run_rtm_cli(cfg, verbose=verbose_enabled)
    else:
        print(f"Error: Unknown task type '{task_type}' in config", file=sys.stderr)
        sys.exit(1)


def _parse_val(val, default, val_type=float):
    if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
        return default
    return val_type(val)


def _run_simulation_cli(cfg: dict, cli_enabled: bool = False, verbose: bool = False):
    t_start = time.time()
    vel, nx, nz, dx, dz = _load_velocity_model(cfg)

    n_sources = _parse_val(cfg.get("n_sources"), 2, int)
    n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
    f0 = _parse_val(cfg.get("f0"), 25.0, float)
    ms = _parse_val(cfg.get("ms", cfg.get("ntime")), 300.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 40, int)
    smooth = _parse_val(cfg.get("smooth"), 5.0, float)
    origin = tuple(cfg.get("origin", [0.0, 0.0]))
    src_origin = tuple(cfg.get("src_origin", [0.0, 0.0]))
    rec_origin = tuple(cfg.get("rec_origin", [0.0, 0.0]))
    group_offset = _parse_val(cfg.get("group_offset"), 1.0, float)
    shot_offset = _parse_val(cfg.get("shot_offset"), 1.0, float)
    gather = cfg.get("gather", "cmp")
    meters_per_cell = _parse_val(cfg.get("meters_per_cell"), 1.0, float)
    snr = _parse_val(cfg.get("snr"), None, float)
    fs_ms = _parse_val(cfg.get("fs_ms"), None, float)

    float_type_str = str(cfg.get("float_type", "float32")).lower()
    float_type = np.float64 if "64" in float_type_str else np.float32

    # Crop model if max_size is specified
    max_size = cfg.get("max_size")
    if max_size is not None:
        max_x, max_z = float(max_size[0]), float(max_size[1])
        x_start = max(0, min(int(np.round(origin[0] / dx)), vel.shape[0]))
        x_end = max(x_start, min(x_start + int(np.round(max_x / dx)), vel.shape[0]))
        z_start = max(0, min(int(np.round(origin[1] / dz)), vel.shape[1]))
        z_end = max(z_start, min(z_start + int(np.round(max_z / dz)), vel.shape[1]))
        if x_end > x_start and z_end > z_start:
            vel = vel[x_start:x_end, z_start:z_end]
            nx = vel.shape[0]
            nz = vel.shape[1]

    # Honor src_origin and rec_origin as absolute physical positions

    if "sources" in cfg and cfg["sources"] is not None:
        sources = np.array(cfg["sources"], dtype=float_type)
    else:
        sources = None

    if "receivers" in cfg and cfg["receivers"] is not None:
        receivers = np.array(cfg["receivers"], dtype=float_type)
    else:
        receivers = None

    shot_rec = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_receivers=n_receivers,
        n_sources=n_sources,
        f0=f0,
        src_origin=src_origin,
        rec_origin=rec_origin,
        origin=origin,
        meters_per_cell=meters_per_cell,
        fd_order=fd_order,
        n_damping=n_damping,
        gather=gather,
        group_offset=group_offset,
        shot_offset=shot_offset,
        smooth=smooth,
        snr=snr,
        engine="devito",
        float_type=float_type,
        device=cfg.get("device", "auto"),
        fs_ms=fs_ms,
        verbose=verbose,
    )
    shot_rec.set_model(vel)

    if sources is not None:
        shot_rec.sources = sources
    else:
        sources = shot_rec.sources

    if receivers is not None:
        shot_rec.recs = receivers
    else:
        receivers = shot_rec.recs

    # Display velocity model before starting simulation if cli is enabled
    if cli_enabled:
        shot_rec.show_model(cli=True)

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
        float_type=float_type,
        verbose=verbose,
    )

    shot_run, us, wavelet = solver.run(ms=ms, save_wavefield=cfg.get("save_wavefield", False))

    cfg.update({
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "n_sources": n_sources,
        "n_receivers": n_receivers,
        "f0": f0,
        "ms": ms,
        "snr": snr,
        "fs_ms": fs_ms,
        "group_offset": group_offset,
        "shot_offset": shot_offset,
        "gather": gather,
        "smooth": smooth,
        "fd_order": fd_order,
        "n_damping": n_damping,
        "meters_per_cell": meters_per_cell,
    })
    if "output_dir" in cfg:
        output_dir = Path(cfg["output_dir"])
    else:
        output_dir = generate_simulation_dir_name(cfg, base_dir="data")
    output_dir.mkdir(parents=True, exist_ok=True)

    n_rec_actual = receivers.shape[1] if receivers.ndim == 3 else len(receivers)
    n_src_actual = len(sources)

    shot_rec.n_receivers = n_rec_actual
    shot_rec.n_sources = n_src_actual
    shot_rec.v0 = gaussian_filter(vel, sigma=smooth)
    shot_rec.shot_run = shot_run

    if snr is not None:
        shot_rec.apply_noise(snr=snr)
        shot_run = shot_rec.shot_run

    if fs_ms is not None and fs_ms > 0:
        new_nt = int(np.round(ms / fs_ms)) + 1
        t_orig = np.linspace(0, ms, shot_run.shape[-1])
        t_new = np.linspace(0, ms, new_nt)
        from scipy.interpolate import interp1d
        f_interp = interp1d(t_orig, shot_run, axis=-1, kind="linear", fill_value="extrapolate")
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
        if snr is not None:
            f.create_dataset("snr", data=snr)
        f.create_dataset("dx", data=dx)
        f.create_dataset("dz", data=dz)
        f.create_dataset("ms", data=ms)
        f.create_dataset("f0", data=f0)

    t_elapsed = time.time() - t_start
    print(f"[shotgen-gpu CLI] Wavefield simulation complete! SEGY and HDF5 written to: {output_dir}")
    if verbose:
        print(f"[shotgen-gpu CLI] GPU runtime: {t_elapsed:.3f} s")

    if cli_enabled:
        shot_rec.show_shot(cli=True, source_idx=0)



def _run_rtm_cli(cfg: dict, verbose: bool = False):
    t_start = time.time()
    dataset_dir = cfg.get("dataset_dir")
    nbl = cfg.get("nbl", 40)
    space_order = cfg.get("space_order", 4)
    device = cfg.get("device", "auto")

    rtm = ReverseTimeMigration(
        dataset_dir=dataset_dir,
        nbl=nbl,
        space_order=space_order,
        device=device,
        verbose=verbose,
    )
    image = rtm.run(save_wavefield=False)

    output_dir = Path(cfg.get("output_dir", "data/gpu_rtm_output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "rtm_image.npy"
    np.save(out_file, image)
    t_elapsed = time.time() - t_start
    print(f"[shotgen-gpu CLI] RTM complete! Image written to: {out_file}")
    if verbose:
        print(f"[shotgen-gpu CLI] GPU runtime: {t_elapsed:.3f} s")


if __name__ == "__main__":
    main()

