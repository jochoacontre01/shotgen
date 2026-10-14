#!/usr/bin/env python
"""
wavefield_animation.py — Wavefield simulation and video generator.

Simulates the forward acoustic wavefield propagation for a specific source
from t0 to tn using Devito, and generates an MP4 animation and optional
NumPy array container of the wavefield snapshots.

Usage:
    python examples/wavefield_animation.py --config examples/configs/quick_config_bpsalt.json --source 0 --step 6.0
"""

import os
import sys
import gc
import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

try:
    import cmocean
    CMAP_WAVEFIELD = cmocean.cm.balance
except ImportError:
    CMAP_WAVEFIELD = plt.cm.seismic

# Configure Devito device and suppress verbose banner
from shotgen_gpu.config import configure_devito_device, detect_device
from devito import Function, TimeFunction, ConditionalDimension, Eq, Operator
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic.operators import iso_stencil

from shotgen.pylogger import get_logger, setup_file_logging
from shotgen.utils import generate_simulation_dir_name
from shotgen.sampleshot import ShotRecord
from shotgen.cli import _load_velocity_model_core

logger = get_logger(__name__)


def _load_velocity_model(cfg: dict):
    """
    Parses and loads the velocity model specified in configuration.
    Supports benchmark models, file paths (.segy/.npy), and asset keywords.
    Resamples model to target dx and dz if required.
    """
    try:
        vel, nx, nz, dx, dz = _load_velocity_model_core(cfg)
    except FileNotFoundError as e:
        raise ValueError(f"Unable to resolve velocity model specification: '{cfg.get('velocity_model')}'") from e
    return vel.astype(np.float32), nx, nz, dx, dz


def _parse_val(val, default, val_type=float):
    if val is None or (isinstance(val, str) and val.strip().lower() in ("none", "null")):
        return default
    return val_type(val)


def build_simulation_geometry(cfg: dict):
    """
    Builds the velocity model, crops according to max_size, and calculates
    complete source and receiver acquisition geometry using ShotRecord.
    """
    vel, nx, nz, dx, dz = _load_velocity_model(cfg)

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

    n_sources = _parse_val(cfg.get("n_sources"), 2, int)
    n_receivers = _parse_val(cfg.get("n_receivers"), 10, int)
    f0 = _parse_val(cfg.get("f0"), 25.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 40, int)
    smooth = _parse_val(cfg.get("smooth"), 5.0, float)
    group_offset = _parse_val(cfg.get("group_offset"), 1.0, float)
    shot_offset = _parse_val(cfg.get("shot_offset"), 1.0, float)
    gather = cfg.get("gather", "common shot")
    meters_per_cell = _parse_val(cfg.get("meters_per_cell"), 1.0, float)

    # Use ShotRecord geometry calculation
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
        engine="devito",
        device="auto",
        verbose=False,
    )
    shot_rec.set_model(vel)

    if "sources" in cfg and cfg["sources"] is not None:
        sources = np.array(cfg["sources"], dtype=np.float32)
    else:
        sources = shot_rec.sources

    if "receivers" in cfg and cfg["receivers"] is not None:
        receivers = np.array(cfg["receivers"], dtype=np.float32)
    else:
        receivers = shot_rec.recs

    # Update cfg with effective dimensions for naming convention
    cfg.update({
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "n_sources": len(sources),
        "n_receivers": receivers.shape[1] if receivers.ndim == 3 else len(receivers),
    })

    return vel, sources, receivers, shot_rec


def simulate_wavefield(
    vel: np.ndarray,
    source_pos: np.ndarray,
    rec_positions: np.ndarray,
    origin: tuple,
    dx: float,
    dz: float,
    ms: float,
    f0: float,
    step_ms: float,
    fd_order: int = 4,
    n_damping: int = 40,
    use_smooth: bool = False,
    smooth_sigma: float = 5.0,
    device: str = "auto",
):
    """
    Executes Devito acoustic forward modeling for a single source, saving wavefield
    snapshots at regular time increments of step_ms.
    """
    logger.info(f"Configuring compute device for simulation: '{device}'")
    device_mode = configure_devito_device(device, verbose=False)
    logger.info(f"Active Devito device mode: {device_mode}")

    if use_smooth:
        logger.info(f"Applying Gaussian smoothing to velocity model (sigma={smooth_sigma}).")
        sim_vel = gaussian_filter(vel, sigma=smooth_sigma)
    else:
        sim_vel = vel

    # Devito seismic models use km/s
    vp_km_s = (sim_vel * 1e-3).astype(np.float32)
    shape = sim_vel.shape
    spacing = (float(dx), float(dz))

    logger.info(
        f"Initializing Devito Model: shape={shape}, spacing={spacing}, "
        f"space_order={fd_order}, n_damping={n_damping}"
    )
    model = Model(
        vp=vp_km_s,
        origin=origin,
        spacing=spacing,
        shape=shape,
        space_order=fd_order,
        nbl=n_damping,
        bcs="damp",
        dtype=np.float32,
    )

    src_coords = np.array([[float(source_pos[0]), float(source_pos[1])]], dtype=np.float32)
    if rec_positions.ndim == 3:
        rec_coords = rec_positions[0].astype(np.float32)
    else:
        rec_coords = rec_positions.astype(np.float32)

    f0_khz = float(f0 * 1e-3)
    geometry = AcquisitionGeometry(
        model,
        rec_coords,
        src_coords,
        t0=0.0,
        tn=float(ms),
        src_type="Ricker",
        f0=f0_khz,
    )

    dt = float(geometry.dt)
    nt = geometry.nt
    logger.info(f"Simulation time grid: t0=0.0 ms, tn={ms:.1f} ms, dt={dt:.4f} ms, nt={nt} steps")

    # Determine time subsampling factor based on step_ms
    factor = max(1, int(np.round(step_ms / dt)))
    effective_step_ms = factor * dt
    num_saves = (nt + factor - 1) // factor
    logger.info(
        f"Snapshot subsampling: step_requested={step_ms:.2f} ms -> "
        f"save_every={factor} steps (effective step = {effective_step_ms:.4f} ms), "
        f"total snapshots = {num_saves}"
    )

    # Subsampled time dimension using ConditionalDimension
    ci = ConditionalDimension(name="ci", parent=model.grid.time_dim, factor=factor)
    dim_x, dim_z = model.grid.dimensions

    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=fd_order)
    usave = Function(
        name="usave",
        grid=model.grid,
        dimensions=(ci, dim_x, dim_z),
        shape=(num_saves, *model.grid.shape),
    )

    src = geometry.src
    rec = geometry.rec
    s = model.grid.stepping_dim.spacing
    m = model.m

    eqn = iso_stencil(u, model, "OT2")
    src_term = src.inject(field=u.forward, expr=src * s**2 / m)
    rec_term = rec.interpolate(expr=u)
    save_term = [Eq(usave, u)]

    logger.info("Compiling and executing Devito Forward Wavefield operator...")
    op = Operator(
        eqn + src_term + rec_term + save_term,
        subs=model.spacing_map,
        name="ForwardWavefieldAnimation",
    )
    kwargs = model.physical_params()
    op.apply(dt=dt, **kwargs)
    logger.info("Wavefield forward propagation complete.")

    raw_snapshots = usave.data
    if n_damping > 0:
        logger.info(f"Cropping {n_damping} boundary damping layers from snapshots...")
        cropped_snapshots = np.array(raw_snapshots[:, n_damping:-n_damping, n_damping:-n_damping], copy=True)
    else:
        cropped_snapshots = np.array(raw_snapshots, copy=True)

    saved_times = np.array([min(ms, i * factor * dt) for i in range(num_saves)], dtype=np.float32)

    return cropped_snapshots, saved_times, effective_step_ms, dt


def create_animation(
    wavefield: np.ndarray,
    times: np.ndarray,
    vel: np.ndarray,
    source_pos: np.ndarray,
    rec_positions: np.ndarray,
    origin: tuple,
    dx: float,
    dz: float,
    output_path: Path,
    source_idx: int,
    fps: int = 30,
):
    """
    Renders and exports the wavefield propagation as an MP4 video file.
    """
    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg is not installed or not in PATH. Please install FFmpeg to export MP4 videos."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames, nx, nz = wavefield.shape
    extent = (
        origin[0],
        origin[0] + nx * dx,
        origin[1] + nz * dz,
        origin[1],
    )

    logger.info(f"Preparing animation rendering: {num_frames} frames, extent={extent}, fps={fps}")

    fig, ax = plt.subplots(figsize=(11, 6), dpi=100)

    # Background velocity model
    ax.imshow(
        vel.T,
        extent=extent,
        cmap="gray",
        aspect="auto",
        alpha=0.45,
        origin="upper",
    )

    # Robust amplitude normalization
    vmax_ref = np.percentile(np.abs(wavefield), 99.8)
    if vmax_ref < 1e-6:
        vmax_ref = 1.0
    vmax_f = max(vmax_ref * 0.7, 1e-4)

    norm = mcolors.TwoSlopeNorm(vmin=-vmax_f, vcenter=0.0, vmax=vmax_f)
    frame_0 = wavefield[0]
    im_wf = ax.imshow(
        frame_0.T,
        extent=extent,
        cmap=CMAP_WAVEFIELD,
        norm=norm,
        aspect="auto",
        alpha=0.85,
        origin="upper",
    )

    # Receiver positions
    if rec_positions is not None:
        if rec_positions.ndim == 3:
            recs = rec_positions[source_idx]
        else:
            recs = rec_positions
        ax.scatter(
            recs[:, 0],
            recs[:, 1],
            c="gold",
            marker="v",
            s=25,
            edgecolors="black",
            linewidths=0.5,
            label="Receivers",
            zorder=8,
        )

    # Source position
    ax.scatter(
        [source_pos[0]],
        [source_pos[1]],
        c="red",
        marker="*",
        s=180,
        edgecolors="white",
        linewidths=1.2,
        label=f"Source {source_idx}",
        zorder=10,
    )

    title_obj = ax.set_title(
        f"Forward Wavefield Simulation — Source {source_idx}  |  Time: {times[0]:.1f} ms",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Distance / Offset [m]", fontsize=11)
    ax.set_ylabel("Depth [m]", fontsize=11)

    # Position legend entirely outside the axis to keep the plot area completely unblocked
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
        fontsize=10,
    )

    fig.subplots_adjust(left=0.09, right=0.82, top=0.92, bottom=0.10)

    def update(frame_idx):
        frame = wavefield[frame_idx]
        im_wf.set_data(frame.T)
        title_obj.set_text(
            f"Forward Wavefield Simulation — Source {source_idx}  |  Time: {times[frame_idx]:.1f} ms"
        )
        return [im_wf, title_obj]

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=False,
    )

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )

    logger.info(f"Encoding animation to {output_path}...")
    pbar = tqdm(total=num_frames, desc="Encoding MP4 frames", unit="frame")

    def progress_callback(curr_frame, total_frames):
        pbar.update(1)

    try:
        anim.save(str(output_path), writer=writer, progress_callback=progress_callback)
        pbar.close()
        logger.info(f"Video animation successfully saved to: {output_path}")
    finally:
        plt.close(fig)
        fig.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(
        description="Simulate forward wavefield and generate animation for a specific source."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to JSON or YAML simulation config file.",
    )
    parser.add_argument(
        "--source",
        "-s",
        type=int,
        default=0,
        help="0-indexed source index to simulate (default: 0).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=None,
        help="Wavefield snapshot interval in milliseconds (default: fs_ms from config or 6.0 ms).",
    )
    parser.add_argument(
        "--ms",
        "-t",
        type=float,
        default=None,
        help="Override total simulation duration in ms (default: from config).",
    )
    parser.add_argument(
        "--velocity",
        choices=["true", "smooth"],
        default="true",
        help="Velocity model used for simulation: 'true' (default) or 'smooth'.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Playback frame rate for the generated MP4 video (default: 30).",
    )
    parser.add_argument(
        "--save-numpy",
        action="store_true",
        help="Save raw wavefield snapshots as a compressed .npz file under wavefield_arrays/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/simulations",
        help="Base directory for output datasets (default: data/simulations).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "gpu", "cpu"],
        default="auto",
        help="Compute device for Devito simulation (default: auto).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output logging.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)

    # Build geometry and resolve effective parameters
    vel, sources, receivers, shot_rec = build_simulation_geometry(cfg)
    n_sources = len(sources)

    if args.source < 0 or args.source >= n_sources:
        logger.error(
            f"Invalid source index {args.source}. Available sources: 0 to {n_sources - 1} "
            f"(total {n_sources} sources)."
        )
        sys.exit(1)

    source_idx = args.source
    source_pos = sources[source_idx]
    logger.info(f"Selected source {source_idx}: position = {source_pos}")

    # Determine simulation parameters
    ms = args.ms if args.ms is not None else _parse_val(cfg.get("ms", cfg.get("ntime")), 300.0, float)
    step_ms = args.step if args.step is not None else _parse_val(cfg.get("fs_ms"), 6.0, float)
    f0 = _parse_val(cfg.get("f0"), 25.0, float)
    fd_order = _parse_val(cfg.get("fd_order"), 4, int)
    n_damping = _parse_val(cfg.get("n_damping"), 40, int)
    smooth_sigma = _parse_val(cfg.get("smooth"), 5.0, float)
    origin = tuple(cfg.get("origin", [0.0, 0.0]))
    dx = float(cfg["dx"])
    dz = float(cfg["dz"])
    use_smooth = (args.velocity == "smooth")

    # Update cfg with effective parameters for dataset folder naming
    cfg.update({
        "ms": ms,
        "f0": f0,
        "n_sources": n_sources,
        "n_receivers": receivers.shape[1] if receivers.ndim == 3 else len(receivers),
    })

    # Dataset directory name matching data/ convention
    dataset_name = generate_simulation_dir_name(cfg, base_dir=None)
    dataset_dir = Path(args.output_dir) / dataset_name
    try:
        dataset_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        fallback_dir = Path("/tmp") / args.output_dir / dataset_name
        logger.warning(
            f"Target output directory '{dataset_dir}' could not be created ({e}). "
            f"Falling back to '{fallback_dir}'."
        )
        dataset_dir = fallback_dir
        dataset_dir.mkdir(parents=True, exist_ok=True)

    # Setup centralized run logging
    log_file = setup_file_logging(dataset_dir, "wavefield_animation", logger=logger)
    logger.info(f"Run execution log initialized at: {log_file}")
    logger.info(f"Target simulation dataset directory: {dataset_dir}")

    # Run forward wavefield simulation
    wavefield_snaps, saved_times, effective_step, dt = simulate_wavefield(
        vel=vel,
        source_pos=source_pos,
        rec_positions=receivers[source_idx] if receivers.ndim == 3 else receivers,
        origin=origin,
        dx=dx,
        dz=dz,
        ms=ms,
        f0=f0,
        step_ms=step_ms,
        fd_order=fd_order,
        n_damping=n_damping,
        use_smooth=use_smooth,
        smooth_sigma=smooth_sigma,
        device=args.device,
    )

    # Output paths
    mp4_filename = f"source_{source_idx}.mp4"
    mp4_path = dataset_dir / mp4_filename

    # Save raw NumPy snapshots if requested
    if args.save_numpy:
        array_dir = dataset_dir / "wavefield_arrays"
        array_dir.mkdir(parents=True, exist_ok=True)
        npz_filename = f"source_{source_idx}.npz"
        npz_path = array_dir / npz_filename
        logger.info(f"Saving wavefield snapshot array to {npz_path}...")
        np.savez_compressed(
            npz_path,
            wavefield=wavefield_snaps,
            time=saved_times,
            dt_snapshot=effective_step,
            dt_sim=dt,
            source_idx=source_idx,
            source_pos=source_pos,
            dx=dx,
            dz=dz,
            origin=origin,
            velocity_type="smooth" if use_smooth else "true",
        )
        logger.info(f"Saved NumPy array container: {npz_path}")

    # Generate animation
    create_animation(
        wavefield=wavefield_snaps,
        times=saved_times,
        vel=vel,
        source_pos=source_pos,
        rec_positions=receivers,
        origin=origin,
        dx=dx,
        dz=dz,
        output_path=mp4_path,
        source_idx=source_idx,
        fps=args.fps,
    )

    logger.info(f"Wavefield animation workflow completed successfully!")
    logger.info(f"Final MP4 video: {mp4_path}")


if __name__ == "__main__":
    main()
