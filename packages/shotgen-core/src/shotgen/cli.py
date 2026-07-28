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
from shotgen.models import resample_velocity_model
from shotgen.io import SegyIO


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
    parser.add_argument("--engine", type=str, default="pylops", help="Simulation engine ('pylops' or 'devito')")
    parser.add_argument("--device", type=str, default="cpu", help="Target device ('cpu', 'cuda', etc.)")

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

    cli_enabled = args.cli or cfg.get("cli", False)

    vel, nx, nz, dx, dz = _load_velocity_model_core(cfg)
    n_sources = int(cfg.get("n_sources", 2))
    n_receivers = int(cfg.get("n_receivers", 10))
    f0 = float(cfg.get("f0", 25.0))
    ms = float(cfg.get("ms", 300.0))
    engine = str(cfg.get("engine", "pylops"))
    device = str(cfg.get("device", "cpu"))

    shot = ShotRecord(
        nx=nx,
        nz=nz,
        dx=dx,
        dz=dz,
        n_sources=n_sources,
        n_receivers=n_receivers,
        f0=f0,
        engine=engine,
        device=device,
    )
    shot.set_model(vel)
    shot.run(ms=ms)

    output_dir = Path(cfg.get("output_dir", "data/core_simulation_output"))
    shot.save_shot(str(output_dir))
    print(f"[shotgen-core CLI] Wavefield simulation complete! Saved to: {output_dir}")

    if cli_enabled:
        shot.show_shot(cli=True, source_idx=0)


if __name__ == "__main__":
    main()
