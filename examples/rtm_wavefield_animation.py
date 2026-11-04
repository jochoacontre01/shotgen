#!/usr/bin/env python
"""
rtm_wavefield_animation.py — Dissect Reverse Time Migration step-by-step.

Simulates and animates forward wavefield, backward residual wavefield,
instantaneous correlation product, and cumulative migrated image for one or
multiple sources using Devito with the random scattering boundary condition
combined with discrete absorbing formulation, and exports an MP4 animation.

For multiple sources (-s or -n), the animation shows the superposition of all
source wavefields at the same time, with each source contribution superimposed
on top of the other, and the cumulative RTM image is the total stacked migration.

Usage:
    python examples/rtm_wavefield_animation.py data/my_dataset -s 0 --step 20.0 --fps 15
    python examples/rtm_wavefield_animation.py data/my_dataset -s 0 2 4 --step 20.0 --fps 15
    python examples/rtm_wavefield_animation.py data/my_dataset -s 0-3 --step 20.0 --fps 15
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from tqdm import tqdm

try:
    import cmocean
    CMAP_WAVEFIELD = cmocean.cm.balance
    CMAP_IMAGE = cmocean.cm.balance_r
except ImportError:
    CMAP_WAVEFIELD = plt.cm.seismic
    CMAP_IMAGE = plt.cm.seismic

from devito import ConditionalDimension, Eq, Function, Operator, TimeFunction, solve
from examples.seismic import PointSource
from examples.seismic.acoustic.operators import iso_stencil
from shotgen.pylogger import get_logger, setup_file_logging
from shotgen.utils import setup_matplotlib_style
from shotgen_gpu.engine.rtm import ReverseTimeMigration

logger = get_logger("rtm_wavefield_animation")


def parse_source_indices(sources_arg, max_sources=None):
    """Parse comma/space/hyphen separated source indices into a sorted list of ints."""
    if not sources_arg:
        return None
    indices = set()
    items = []
    for item in sources_arg:
        items.extend(item.replace(",", " ").split())

    for token in items:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if start <= end else -1
            indices.update(range(start, end + step, step))
        else:
            indices.add(int(token))

    sorted_indices = sorted(indices)
    if max_sources is not None:
        sorted_indices = [i for i in sorted_indices if 0 <= i < max_sources]
    return sorted_indices


def apply_random_boundary_condition(rtm: ReverseTimeMigration, seed: int = 42):
    """
    Applies the random scattering boundary condition matching shotgen_gpu.engine.rtm.
    Smoothly blends uniform random velocities across the outer boundary zones (non_zero_damp)
    to scatter coherent boundary reflections into diffuse incoherent noise while
    maintaining the absorbing damping layer to dissipate energy and prevent outer-wall reflection.
    """
    logger.info("Applying random scattering boundary condition from engine.rtm...")
    non_zero_damp = (rtm.model0.damp.data > 0)
    damp_max = np.max(rtm.model0.damp.data)
    damp_min = np.min(rtm.model0.damp.data)
    damp_normalized = (
        (rtm.model0.damp.data - damp_min) / (damp_max - damp_min)
        if damp_max > damp_min
        else np.zeros_like(rtm.model0.damp.data)
    )

    rng = np.random.default_rng(seed=seed)
    rand_damp = rng.uniform(
        np.min(rtm.model0.vp.data),
        np.max(rtm.model0.vp.data),
        size=rtm.model0.vp.data.shape,
    )

    w = damp_normalized[non_zero_damp]
    rtm.model0.vp.data[non_zero_damp] = (
        (1.0 - w) * rtm.model0.vp.data[non_zero_damp] + w * rand_damp[non_zero_damp]
    )
    rtm.model.vp.data[non_zero_damp] = rtm.model0.vp.data[non_zero_damp]
    logger.info(
        "Random scattering velocity field assigned across %d outer boundary cells.",
        int(np.count_nonzero(non_zero_damp)),
    )


def run_rtm_animation_simulation(
    dataset_dir: Path,
    source_indices: list[int],
    step_ms: float = 20.0,
    nbl: int = 128,
    space_order: int = 4,
    time_order: int = 2,
    f0: float | None = None,
    device: str = "auto",
):
    """
    Executes RTM forward propagation and backward adjoint propagation for multiple shots,
    accumulating the wavefield superpositions, instantaneous correlation products, and
    migrated stacked images in-place to maintain bounded memory usage.
    """
    logger.info("Initializing native ReverseTimeMigration engine from %s...", dataset_dir)
    rtm = ReverseTimeMigration(
        dataset_dir=str(dataset_dir),
        nbl=nbl,
        space_order=space_order,
        time_order=time_order,
        f0=f0,
        device=device,
        verbose=True,
    )

    valid_indices = [idx for idx in source_indices if 0 <= idx < rtm.n_sources]
    if not valid_indices:
        raise ValueError(f"No valid source indices in {source_indices} (max available: {rtm.n_sources}).")

    # Apply random boundary implementation from engine.rtm
    apply_random_boundary_condition(rtm)

    dt = float(rtm.geometry.dt)
    nt = rtm.geometry.nt
    freq_hz = rtm.f0 * 1000.0 if rtm.f0 < 1.0 else rtm.f0
    logger.info(
        "RTM Geometry: nt=%d, dt=%.4f ms, tn=%.2f ms, f0=%.2f Hz (%.5f kHz)",
        nt, dt, rtm.tn, freq_hz, rtm.f0
    )

    # Subsampling configuration
    factor = max(1, int(np.round(step_ms / dt)))
    effective_step = factor * dt
    num_saves = (nt + factor - 1) // factor
    logger.info(
        "Subsampling: factor=%d (%.3f ms step), total snapshots=%d across %d sources",
        factor, effective_step, num_saves, len(valid_indices)
    )

    ci = ConditionalDimension(name="ci", parent=rtm.model0.grid.time_dim, factor=factor)
    dim_x, dim_z = rtm.model0.grid.dimensions
    dt_sym = rtm.model0.grid.stepping_dim.spacing

    # Devito Forward Operator
    u = TimeFunction(
        name="u",
        grid=rtm.model0.grid,
        time_order=time_order,
        space_order=space_order,
        save=None,
    )
    u_save = Function(
        name="u_save",
        grid=rtm.model0.grid,
        dimensions=(ci, dim_x, dim_z),
        shape=(num_saves, *rtm.model0.grid.shape),
    )
    source = rtm.geometry.src
    eqn_u = iso_stencil(u, rtm.model0, "OT2")
    src_term = source.inject(field=u.forward, expr=source * dt_sym**2 / rtm.model0.m)
    save_u = Eq(u_save, u)

    op_forward = Operator(
        eqn_u + src_term + [save_u],
        subs=rtm.model0.spacing_map,
        name="RTMForwardAnimationOp",
    )

    # Devito Backward Operator
    v = TimeFunction(
        name="v",
        grid=rtm.model0.grid,
        time_order=time_order,
        space_order=space_order,
        save=None,
    )
    v_save = Function(
        name="v_save",
        grid=rtm.model0.grid,
        dimensions=(ci, dim_x, dim_z),
        shape=(num_saves, *rtm.model0.grid.shape),
    )
    residual_source = PointSource(
        name="residual",
        grid=rtm.model0.grid,
        time_range=rtm.geometry.time_axis,
        coordinates=rtm.geometry.rec_positions,
    )
    eqn_v = rtm.model0.m * v.dt2 - v.laplace
    stencil_v = Eq(v.backward, solve(eqn_v, v.backward))
    res_term = residual_source.inject(
        field=v.backward,
        expr=residual_source * dt_sym**2 / rtm.model0.m,
    )
    save_v = Eq(v_save, v)

    op_backward = Operator(
        [stencil_v] + res_term + [save_v],
        subs=rtm.model0.spacing_map,
        name="RTMBackwardAnimationOp",
    )

    # Pre-allocate accumulation arrays in physical domain (excluding nbl boundary)
    nx_phys, nz_phys = rtm.vp.shape
    total_u = np.zeros((num_saves, nx_phys, nz_phys), dtype=np.float32)
    total_v = np.zeros((num_saves, nx_phys, nz_phys), dtype=np.float32)
    total_prod = np.zeros((num_saves, nx_phys, nz_phys), dtype=np.float32)

    all_sources = []
    all_receivers = []

    for s_idx in tqdm(valid_indices, desc="Simulating Sources", unit="source"):
        current_src = rtm.sources[s_idx, :]
        current_recs = rtm.receivers[s_idx] if rtm.receivers.ndim == 3 else rtm.receivers

        all_sources.append(current_src)
        all_receivers.append(current_recs)

        rtm.geometry.src_positions[0, :] = current_src
        source.coordinates.data[0, :] = current_src
        residual_source.coordinates.data[:, :] = current_recs

        # Reset operator state for next source
        u.data[:] = 0.0
        u_save.data[:] = 0.0
        v.data[:] = 0.0
        v_save.data[:] = 0.0

        # Execute forward pass
        op_forward(time_m=0, time_M=nt - 2, dt=dt)

        # Execute backward pass
        op_backward(time_m=0, time_M=nt - 2, dt=dt, residual=rtm.shots[s_idx])

        # Accumulate cropped physical domain snapshots in-place
        raw_u = np.asarray(u_save.data)
        raw_v = np.asarray(v_save.data)
        for k in range(num_saves):
            u_crop = raw_u[k, nbl:-nbl, nbl:-nbl]
            v_crop = raw_v[k, nbl:-nbl, nbl:-nbl]
            total_u[k] += u_crop
            total_v[k] += v_crop
            total_prod[k] += (u_crop * v_crop)

    times = np.array([min(s_idx * factor, nt - 1) * dt for s_idx in range(num_saves)])
    dx, dz = rtm.spacing

    return {
        "u": total_u,
        "v": total_v,
        "prod": total_prod,
        "vp": rtm.vp,
        "times": times,
        "dx": dx,
        "dz": dz,
        "origin": rtm.origin,
        "sources": np.array(all_sources),
        "receivers": np.vstack(all_receivers),
        "source_indices": valid_indices,
    }


def create_rtm_dissection_animation(
    data_dict: dict,
    output_path: Path,
    fps: int = 15,
    reverse: bool = False,
):
    """
    Renders 2x2 grid animation of RTM multi-source superposition:
    Top-Left: Superimposed Forward Wavefields u(x, z, t)
    Top-Right: Superimposed Backward Residual Wavefields v(x, z, t)
    Bottom-Left: Superimposed Instantaneous Correlation Product ∑ u_i(x, z, t) · v_i(x, z, t)
    Bottom-Right: Cumulative Stacked Migrated Image accumulating step-by-step
    """
    setup_matplotlib_style()

    u = data_dict["u"]
    v = data_dict["v"]
    prod = data_dict["prod"]
    vp = data_dict["vp"]
    times = data_dict["times"]
    dx, dz = data_dict["dx"], data_dict["dz"]
    origin = data_dict["origin"]
    sources_pos = data_dict["sources"]
    rec_pos = data_dict["receivers"]
    source_indices = data_dict["source_indices"]

    num_frames, nx, nz = u.shape
    extent = [
        origin[0],
        origin[0] + nx * dx,
        origin[1] + nz * dz,
        origin[1],
    ]

    logger.info("Preparing wavefields and precomputing cumulative image array...")
    if reverse:
        logger.info("Animation mode: Reverse-time order (t = tn -> 0).")
        u_seq = u[::-1]
        v_seq = v[::-1]
        prod_seq = prod[::-1]
        times_seq = times[::-1]
    else:
        logger.info("Animation mode: Forward-time order (t = 0 -> tn).")
        u_seq = u
        v_seq = v
        prod_seq = prod
        times_seq = times

    cum_image = np.cumsum(prod_seq, axis=0)

    # Standard global amplitude normalization matching wavefield_animation.py
    def compute_global_norm(arr, min_clip=1e-6):
        vmax_ref = float(np.percentile(np.abs(arr), 99.8))
        vmax_ref = max(vmax_ref, min_clip)
        vmax_f = max(vmax_ref * 0.7, min_clip)
        return mcolors.TwoSlopeNorm(vmin=-vmax_f, vcenter=0.0, vmax=vmax_f)

    norm_u = compute_global_norm(u_seq, min_clip=1e-4)
    norm_v = compute_global_norm(v_seq, min_clip=1e-11)
    norm_prod = compute_global_norm(prod_seq, min_clip=1e-13)
    norm_cum = compute_global_norm(cum_image[-1], min_clip=1e-13)

    logger.info("Global normalization limits:")
    logger.info("  u_vmax: %.3e", norm_u.vmax)
    logger.info("  v_vmax: %.3e", norm_v.vmax)
    logger.info("  prod_vmax: %.3e", norm_prod.vmax)
    logger.info("  cum_vmax: %.3e", norm_cum.vmax)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    (ax_u, ax_v), (ax_prod, ax_img) = axes

    # Background velocity overlays
    for ax in [ax_u, ax_v, ax_prod, ax_img]:
        ax.imshow(vp.T, extent=extent, cmap="gray", aspect="auto", alpha=0.35, origin="upper")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

    # Sources and receivers markers
    src_label = "Source" if len(sources_pos) == 1 else f"Sources ({len(sources_pos)})"
    ax_u.scatter(
        sources_pos[:, 0],
        sources_pos[:, 1],
        c="red",
        marker="*",
        s=120 if len(sources_pos) > 1 else 160,
        edgecolors="white",
        label=src_label,
        zorder=10,
    )
    ax_u.legend(loc="upper right", fontsize=8)

    ax_v.scatter(
        rec_pos[:, 0],
        rec_pos[:, 1],
        c="gold",
        marker="v",
        s=15,
        edgecolors="black",
        linewidths=0.3,
        label="Receivers",
        zorder=10,
    )
    ax_v.legend(loc="upper right", fontsize=8)

    # Initial image displays
    im_u = ax_u.imshow(u_seq[0].T, extent=extent, cmap=CMAP_WAVEFIELD, norm=norm_u, aspect="auto", alpha=0.85)
    ax_u.set_title("Forward Wavefield: ∑ u_i(x, z, t)")
    fig.colorbar(im_u, ax=ax_u, shrink=0.7, label="u amp")

    im_v = ax_v.imshow(v_seq[0].T, extent=extent, cmap=CMAP_WAVEFIELD, norm=norm_v, aspect="auto", alpha=0.85)
    ax_v.set_title("Backward Residual Wavefield: ∑ v_i(x, z, t)")
    fig.colorbar(im_v, ax=ax_v, shrink=0.7, label="v amp")

    im_prod = ax_prod.imshow(prod_seq[0].T, extent=extent, cmap=CMAP_IMAGE, norm=norm_prod, aspect="auto")
    ax_prod.set_title("Instantaneous Correlation: ∑ u_i · v_i")
    fig.colorbar(im_prod, ax=ax_prod, shrink=0.7, label="u·v")

    im_img = ax_img.imshow(cum_image[0].T, extent=extent, cmap=CMAP_IMAGE, norm=norm_cum, aspect="auto")
    ax_img.set_title("Cumulative Stacked RTM Image: ∑∑ u_i · v_i")
    fig.colorbar(im_img, ax=ax_img, shrink=0.7, label="Image amp")

    time_label = "Reverse Time" if reverse else "Time"
    if len(source_indices) <= 4:
        src_desc = ",".join(str(i) for i in source_indices)
    else:
        src_desc = f"{source_indices[0]}..{source_indices[-1]} ({len(source_indices)} shots)"
    fig.suptitle(f"RTM Step-by-Step Dissection — Source(s): [{src_desc}] | {time_label}: {times_seq[0]:.1f} ms", fontsize=12)
    fig.tight_layout()

    def update(frame_idx):
        im_u.set_data(u_seq[frame_idx].T)
        im_v.set_data(v_seq[frame_idx].T)
        im_prod.set_data(prod_seq[frame_idx].T)
        im_img.set_data(cum_image[frame_idx].T)

        fig.suptitle(
            f"RTM Step-by-Step Dissection — Source(s): [{src_desc}] | {time_label}: {times_seq[frame_idx]:.1f} ms",
            fontsize=12,
        )
        return [im_u, im_v, im_prod, im_img]

    anim = animation.FuncAnimation(fig, update, frames=num_frames, blit=False)

    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
    )

    logger.info("Encoding animation video to %s...", output_path)
    pbar = tqdm(total=num_frames, desc="Encoding MP4 frames", unit="frame")

    def progress_callback(curr_frame, total_frames):
        pbar.update(1)

    try:
        anim.save(str(output_path), writer=writer, progress_callback=progress_callback)
        pbar.close()
        logger.info("RTM Dissection video successfully saved to: %s", output_path)
    finally:
        plt.close(fig)
        fig.clear()
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Dissect RTM step-by-step and produce 2x2 MP4 animation.")
    parser.add_argument("file", type=str, help="Dataset directory or SEGY file path")
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        default=None,
        help="Source index, list of indices, or range (e.g., -s 0, -s 0 2 4, or -s 0-3)",
    )
    parser.add_argument(
        "-n",
        "--num-sources",
        type=int,
        default=None,
        help="Migrate the first N sources (e.g., -n 3)",
    )
    parser.add_argument("--step", type=float, default=20.0, help="Subsampling step in ms (default: 20.0 ms)")
    parser.add_argument("--fps", type=int, default=15, help="Animation frames per second (default: 15)")
    parser.add_argument("--nbl", type=int, default=128, help="Number of boundary layers for absorption and random scattering (default: 128)")
    parser.add_argument("--space-order", type=int, default=4, help="Devito spatial discretization order (default: 4)")
    parser.add_argument("--f0", type=float, default=None, help="Peak frequency in Hz (e.g. 25.0)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--reverse", action="store_true", help="Play animation in reverse-time order (tn -> 0)")
    parser.add_argument("-c", "--cli", action="store_true", help="CLI mode (optional terminal display)")

    args = parser.parse_args()

    shotpath = Path(__file__).resolve().parents[1] / args.file
    if not shotpath.exists():
        data_shotpath = Path(__file__).resolve().parents[1] / "data" / args.file
        if data_shotpath.exists():
            shotpath = data_shotpath
        else:
            raise FileNotFoundError(f"Could not find dataset: {args.file}")

    setup_file_logging(dataset_dir=shotpath, script_name="rtm_wavefield_animation", logger=logger)

    # Resolve source indices
    if args.sources is not None:
        selected_sources = parse_source_indices(args.sources)
    elif args.num_sources is not None:
        selected_sources = list(range(max(1, args.num_sources)))
    else:
        selected_sources = [0]

    logger.info("Target source indices for animation: %s", selected_sources)

    sim_data = run_rtm_animation_simulation(
        dataset_dir=shotpath,
        source_indices=selected_sources,
        step_ms=args.step,
        nbl=args.nbl,
        space_order=args.space_order,
        f0=args.f0,
        device=args.device,
    )

    valid_indices = sim_data["source_indices"]
    if len(valid_indices) == 1:
        out_name = f"rtm_animation_src{valid_indices[0]:03d}.mp4"
    elif len(valid_indices) <= 4:
        out_name = f"rtm_animation_srcs_{'_'.join(str(i) for i in valid_indices)}.mp4"
    else:
        out_name = f"rtm_animation_srcs_{valid_indices[0]}-{valid_indices[-1]}_{len(valid_indices)}shots.mp4"

    output_mp4 = shotpath / out_name
    try:
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        with open(output_mp4, "ab"):
            pass
    except OSError:
        output_mp4 = Path("/tmp") / f"{shotpath.name}_{out_name}"
        logger.warning("Dataset directory is read-only. Saving animation to fallback path: %s", output_mp4)

    create_rtm_dissection_animation(
        data_dict=sim_data,
        output_path=output_mp4,
        fps=args.fps,
        reverse=args.reverse,
    )


if __name__ == "__main__":
    main()
