import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import yaml

from shotgen.sampleshot import (
    ShotRecord,
    load_marmousi,
    load_bpsalt,
    load_overthrust,
    load_sigsbee,
    _find_assets_dir
)
from shotgen.utils import generate_simulation_dir_name


def _load_velocity_model_core(cfg: dict):
    vel_spec = cfg.get("velocity_model") or cfg.get("vel_file")

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
            nx = int(cfg.get("nx", meta.get("nx", nx_val)))
            nz = int(cfg.get("nz", meta.get("nz", nz_val)))
            return vel, nx, nz, dx_orig, dz_orig

        filepath = Path(vel_spec)
        if not filepath.exists():
            assets_file = _find_assets_dir() / vel_spec
            if assets_file.exists():
                filepath = assets_file

        if filepath.exists():
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

    v_base = float(cfg.get("v_base", 2500.0))
    vel = v_base * np.ones((default_nx, default_nz), dtype=np.float32)
    vel[:, default_nz // 2:] = v_base + 500.0
    return vel, default_nx, default_nz, default_dx, default_dz


def main():
    parser = argparse.ArgumentParser(description="shotgen-core: CPU/PyTorch wavefield simulation CLI")
    parser.add_argument("--config", "-c", type=str, help="Path to JSON or YAML configuration file")
    parser.add_argument("--cli", action="store_true", help="Display the first shot record on the terminal using chafa after simulation")
    parser.add_argument("--n-sources", type=int, help="Number of sources")
    parser.add_argument("--n-receivers", type=int, help="Number of receivers")
    parser.add_argument("--f0", type=float, help="Central frequency (Hz)")
    parser.add_argument("--ms", "--ntime", type=float, help="Simulation duration (ms)")
    parser.add_argument("--engine", type=str, help="Simulation engine ('pylops' or 'devito')")
    parser.add_argument("--device", type=str, help="Target device ('cpu', 'cuda', etc.)")
    parser.add_argument("--fs-ms", "--fs_ms", dest="fs_ms", type=float, help="Sampling rate in milliseconds (float)")
    parser.add_argument("--snr", type=float, help="Signal-to-noise ratio for noise addition (float)")
    parser.add_argument("--smooth", type=float, help="Smoothing factor for background velocity model")
    parser.add_argument("--fd-order", "--fd_order", dest="fd_order", type=int, help="FD space order")
    parser.add_argument("--n-damping", "--n_damping", dest="n_damping", type=int, help="Damping boundary size")
    parser.add_argument("--gather", type=str, help="Gather type ('cmp' or 'common shot')")
    parser.add_argument("--group-offset", "--group_offset", dest="group_offset", type=float, help="Group offset")
    parser.add_argument("--shot-offset", "--shot_offset", dest="shot_offset", type=float, help="Shot offset")

    args = parser.parse_args()

    cfg = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found at {config_path}", file=sys.stderr)
            sys.exit(1)
        with open(config_path, "r") as f:
            if config_path.suffix.lower() in [".yaml", ".yml"]:
                cfg = yaml.safe_load(f) or {}
            else:
                cfg = json.load(f)

    if args.n_sources is not None:
        cfg["n_sources"] = args.n_sources
    if args.n_receivers is not None:
        cfg["n_receivers"] = args.n_receivers
    if args.f0 is not None:
        cfg["f0"] = args.f0
    if args.ms is not None:
        cfg["ms"] = args.ms
    if args.engine:
        cfg["engine"] = args.engine
    if args.device:
        cfg["device"] = args.device
    if args.fs_ms is not None:
        cfg["fs_ms"] = args.fs_ms
    if args.snr is not None:
        cfg["snr"] = args.snr
    if args.smooth is not None:
        cfg["smooth"] = args.smooth
    if args.fd_order is not None:
        cfg["fd_order"] = args.fd_order
    if args.n_damping is not None:
        cfg["n_damping"] = args.n_damping
    if args.gather:
        cfg["gather"] = args.gather
    if args.group_offset is not None:
        cfg["group_offset"] = args.group_offset
    if args.shot_offset is not None:
        cfg["shot_offset"] = args.shot_offset

    cli_enabled = args.cli or cfg.get("cli", False)

    def _parse_val(val, default, val_type=float):
        if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
            return default
        return val_type(val)

    vel, nx, nz, dx, dz = _load_velocity_model_core(cfg)
    n_sources = _parse_val(cfg.get("n_sources"), 2, int)
    n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
    f0 = _parse_val(cfg.get("f0"), 25.0, float)
    ms = _parse_val(cfg.get("ms", cfg.get("ntime")), 300.0, float)
    engine = str(cfg.get("engine", "pylops"))
    device = str(cfg.get("device", "cpu"))
    fs_ms = _parse_val(cfg.get("fs_ms"), None, float)
    snr = _parse_val(cfg.get("snr"), None, float)
    smooth = _parse_val(cfg.get("smooth"), 5.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 100, int)
    gather = cfg.get("gather", "cmp")
    group_offset = _parse_val(cfg.get("group_offset"), 1.0, float)
    shot_offset = _parse_val(cfg.get("shot_offset"), 1.0, float)
    meters_per_cell = _parse_val(cfg.get("meters_per_cell"), 1.0, float)
    origin = tuple(cfg.get("origin", [0.0, 0.0]))
    src_origin = tuple(cfg.get("src_origin", [0.0, 0.0]))
    rec_origin = tuple(cfg.get("rec_origin", [0.0, 0.0]))

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

    shot = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=n_sources,
        n_receivers=n_receivers,
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
        engine=engine,
        float_type=float_type,
        device=device,
        fs_ms=fs_ms,
    )
    shot.set_model(vel)
    if cli_enabled:
        shot.show_model(cli=True)

    shot.run(ms=ms)

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
    shot.save_shot(str(output_dir))
    print(f"[shotgen-core CLI] Wavefield simulation complete! Saved to: {output_dir}")

    if cli_enabled:
        shot.show_shot(cli=True, source_idx=0)


if __name__ == "__main__":
    main()
