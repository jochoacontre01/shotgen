import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.spatial import cKDTree
from shotgen.io import convert_npy_to_segy
from shotgen.utils import setup_matplotlib_style

setup_matplotlib_style()

# plt.style.use(["science","no-latex"])

file = "assets/CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908.segy"

with segyio.open(file, ignore_geometry=True) as f:
    data = f.trace.raw[:]
    dz = segyio.tools.dt(f) / 1000.0
    # SourceX/SourceY are stored with 0.01 m precision (UTM Zone 22N, EPSG:32622)
    trace_x = f.attributes(segyio.TraceField.SourceX)[:] / 100.0
    trace_y = f.attributes(segyio.TraceField.SourceY)[:] / 100.0

dx = 15
print("Initial shape: ", data.shape)
print("dz (m): ", dz)
print("dx (m):", dx)

masked = np.where(data==5000, data, np.nan)
new_z = int(35_000 / dz)
new_array = np.zeros((data.shape[0], new_z))
print("New shape:", new_array.shape)

# Assign 5000 to the traces up to 35 km depth
first_idx = np.argmax(~np.isnan(masked), axis=1)
for i, idx in enumerate(first_idx):
    new_array[i, idx:] = 5000

# Assign 8000 to the traces from first boundary + 12 km depth
moho_boundary = first_idx + int(12_000/dz)
for i, idx in enumerate(moho_boundary):
    new_array[i, idx:] = 8000

# Linear gradient from 5000 to 7000
for i, idx in enumerate(first_idx):
    new_val = np.interp(np.arange(idx, moho_boundary[i]), (idx, moho_boundary[i]), (new_array[i, idx], 7000))
    new_array[i, idx:moho_boundary[i]] = new_val

# Linear gradient from 8000 to 8500
for i, idx in enumerate(moho_boundary):
    new_val = np.interp(np.arange(idx, new_array.shape[1]), (idx, new_array.shape[1]), (new_array[i, idx], 8500))
    new_array[i, idx:] = new_val

# Re-assign top traces
for i, idx in enumerate(first_idx):
    new_array[i, :idx] = data[i, :idx]

extent = (
    0,
    new_array.shape[0] * dx,
    new_array.shape[1] * dz,
    0,
    
)

# np.save("assets/CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908_moho.npy", new_array)
# Load interpretation file (UTM Zone 22N coordinates in cols 2, 3; depth in col 4).
def load_interpretation(file, trace_x, trace_y, interpolate=True, max_gap=250, max_edge_gap=50):
    df = pd.read_csv(file, skiprows=1, header=None, sep=r"\s+")
    utm_x = df.iloc[:, 2].to_numpy()
    utm_y = df.iloc[:, 3].to_numpy()
    z = df.iloc[:, 4].to_numpy(dtype=float)

    # Map each interpretation point to its nearest SEG-Y trace index, then to plot x = trace_idx * dx.
    tree = cKDTree(np.column_stack([trace_x, trace_y]))
    _, trace_idx = tree.query(np.column_stack([utm_x, utm_y]))

    if interpolate:
        # Sort and deduplicate trace indices (required for PCHIP).
        trace_idx, unique_pos = np.unique(trace_idx, return_index=True)
        z = z[unique_pos]

        n_traces = len(trace_x)
        full_idx = np.arange(n_traces)
        z_interp = np.full(n_traces, np.nan, dtype=float)

        # Split into segments separated by gaps wider than max_gap and interpolate each with PCHIP.
        splits = np.where(np.diff(trace_idx) > max_gap)[0] + 1
        for seg_idx, seg_z in zip(np.split(trace_idx, splits), np.split(z, splits)):
            if len(seg_idx) == 1:
                z_interp[seg_idx[0]] = seg_z[0]
            else:
                pchip = PchipInterpolator(seg_idx, seg_z)
                grid = np.arange(seg_idx[0], seg_idx[-1] + 1)
                z_interp[grid] = pchip(grid)

        # Extend flat only across small edge margins (<= max_edge_gap); leave large unpicked ends as NaN.
        if 0 < trace_idx[0] <= max_edge_gap:
            z_interp[: trace_idx[0]] = z[0]
        if 0 < (n_traces - 1 - trace_idx[-1]) <= max_edge_gap:
            z_interp[trace_idx[-1] + 1 :] = z[-1]

        return full_idx * dx, z_interp

    x = trace_idx * dx

    # Break the line with NaNs where there are unpicked gaps between consecutive points.
    gap_indices = np.where(np.diff(trace_idx) > 1)[0] + 1
    x_plot = np.insert(x.astype(float), gap_indices, np.nan)
    z_plot = np.insert(z, gap_indices, np.nan)

    return x_plot, z_plot


def _compute_termination_weights(z_curve, taper_traces=60):
    """Compute smooth cosine taper weights in [0, 1] near internal NaN terminations."""
    n = len(z_curve)
    valid = ~np.isnan(z_curve)
    weights = valid.astype(float)
    if taper_traces <= 0 or np.all(valid) or not np.any(valid):
        return weights

    padded = np.concatenate([[False], valid, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0]

    for s, e in zip(starts, ends):
        seg_len = e - s
        if s > 0:
            t_len = min(taper_traces, seg_len)
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(1, t_len + 1) / (t_len + 1)))
            weights[s : s + t_len] = np.minimum(weights[s : s + t_len], ramp)
        if e < n:
            t_len = min(taper_traces, seg_len)
            ramp = 0.5 * (1.0 - np.cos(np.pi * np.arange(t_len, 0, -1) / (t_len + 1)))
            weights[e - t_len : e] = np.minimum(weights[e - t_len : e], ramp)

    return weights


def discretize_velocity_model(
    bg_vel,
    horizon_depths,
    dz,
    stat="median",
    lateral_smooth_sigma=15.0,
    taper_traces=60,
    smooth_sigma=0.0,
):
    """
    Discretize a smooth background velocity model into sharp horizon-bounded layers.

    Parameters
    ----------
    bg_vel : np.ndarray
        Background velocity model of shape (nx, nz).
    horizon_depths : list[np.ndarray]
        List of 1D arrays of length nx giving horizon depth in meters (NaN where absent).
    dz : float
        Vertical grid spacing in meters.
    stat : {"median", "mean"}
        Statistic used to summarize background velocity within each layer at each trace.
    lateral_smooth_sigma : float
        Lateral Gaussian smoothing sigma (in traces) applied to the background model before
        layer aggregation, keeping vertical horizon jumps crisp while smoothing lateral jitter.
    taper_traces : int
        Cosine taper width (in traces) at partial-horizon terminations to prevent artificial
        vertical velocity steps.
    smooth_sigma : float
        Optional final 2D Gaussian smoothing sigma (in grid cells). Default 0.0 (crisp 1-cell steps).

    Returns
    -------
    np.ndarray
        Discretized velocity model of shape (nx, nz), dtype float32.
    """
    nx, nz = bg_vel.shape
    stat_fn = np.median if stat == "median" else np.mean

    if lateral_smooth_sigma > 0:
        bg_work = gaussian_filter1d(bg_vel.astype(float), sigma=lateral_smooth_sigma, axis=0)
    else:
        bg_work = bg_vel.astype(float)

    h_depths = np.asarray(horizon_depths, dtype=float)
    h_weights = np.array([_compute_termination_weights(h, taper_traces=taper_traces) for h in h_depths])

    disc_vel = np.empty((nx, nz), dtype=np.float32)

    for ix in range(nx):
        col_bg = bg_work[ix]
        valid_mask = ~np.isnan(h_depths[:, ix])
        iz_vals = np.clip(np.round(h_depths[valid_mask, ix] / dz).astype(int), 1, nz - 1)
        w_vals = h_weights[valid_mask, ix]

        order = np.argsort(iz_vals)
        iz_sorted = iz_vals[order]
        w_sorted = w_vals[order]

        unique_iz = []
        unique_w = []
        for iz, w in zip(iz_sorted, w_sorted):
            if unique_iz and iz == unique_iz[-1]:
                unique_w[-1] = max(unique_w[-1], w)
            else:
                unique_iz.append(iz)
                unique_w.append(w)

        bounds = [0] + unique_iz + [nz]
        col_out = np.empty(nz, dtype=float)

        for a, b in zip(bounds[:-1], bounds[1:]):
            col_out[a:b] = stat_fn(col_bg[a:b])

        for k, (iz, w) in enumerate(zip(unique_iz, unique_w)):
            if w < 1.0:
                a = bounds[k]
                b = bounds[k + 2]
                v_parent = stat_fn(col_bg[a:b])
                col_out[a:iz] = (1.0 - w) * v_parent + w * col_out[a:iz]
                col_out[iz:b] = (1.0 - w) * v_parent + w * col_out[iz:b]

        disc_vel[ix] = col_out

    if smooth_sigma > 0:
        disc_vel = gaussian_filter(disc_vel, sigma=smooth_sigma).astype(np.float32)

    return disc_vel


base_dir = "assets/fpb_horizons"
horizons = {}
for horizon in sorted(os.listdir(base_dir)):
    path = os.path.join(base_dir, horizon)
    if not os.path.isfile(path):
        continue
    x_plot, z_plot = load_interpretation(path, trace_x, trace_y)
    horizons[horizon] = (x_plot, z_plot)

# Include synthetic 5000 m/s basement and Moho boundaries alongside interpreted horizons
all_boundaries = [z_plot for _, z_plot in horizons.values()] + [
    first_idx * dz,
    moho_boundary * dz,
]

disc_vel = discretize_velocity_model(
    new_array,
    all_boundaries,
    dz=dz,
    stat="median",
    lateral_smooth_sigma=15.0,
    taper_traces=100,
    smooth_sigma=0.0,
)

# Below Horizon8: replace discretized median with linear gradient from last velocity above Horizon8
# to 7000 m/s at the Moho, and restore the original 8000 -> 8500 m/s gradient below the Moho.
_, h8_z = horizons["Horizon8"]
nz = disc_vel.shape[1]
for ix in range(disc_vel.shape[0]):
    moho_iz = int(moho_boundary[ix])
    if not np.isnan(h8_z[ix]):
        h8_iz = int(np.clip(np.round(h8_z[ix] / dz), 1, moho_iz))
        v_above = float(disc_vel[ix, h8_iz - 1])
        if h8_iz < moho_iz:
            disc_vel[ix, h8_iz:moho_iz] = np.interp(
                np.arange(h8_iz, moho_iz), (h8_iz, moho_iz), (v_above, 7000.0)
            )
    if moho_iz < nz:
        disc_vel[ix, moho_iz:] = np.interp(
            np.arange(moho_iz, nz), (moho_iz, nz), (8000.0, 8500.0)
        )

print(
    f"Discretized shape: {disc_vel.shape}, dtype: {disc_vel.dtype}, "
    f"vel range: {disc_vel.min():.1f}..{disc_vel.max():.1f} m/s"
)

# Save discretized velocity model (.npy and project-compliant .segy)
out_stem = "assets/CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908_moho_discretized"
np.save(f"{out_stem}.npy", disc_vel)
convert_npy_to_segy(
    npy_path_or_array=disc_vel,
    output_filepath=f"{out_stem}.segy",
    dx=dx,
    dz=dz,
    order="xz",
    verbose=True,
)

# Plot side-by-side comparison: background vs discretized velocity model
fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
vmin, vmax = float(new_array.min()), float(new_array.max())

im0 = axes[0].imshow(new_array.T, aspect="auto", extent=extent, cmap="nipy_spectral", vmin=vmin, vmax=vmax)
for name, (x_plot, z_plot) in horizons.items():
    axes[0].plot(x_plot, z_plot, linewidth=0.9, label=name)
axes[0].set_title("Background Velocity + Horizons")
axes[0].set_xlim(extent[0], extent[1])
axes[0].set_ylim(extent[2], extent[3])
axes[0].set_xlabel("Distance (m)")
axes[0].set_ylabel("Depth (m)")

im1 = axes[1].imshow(disc_vel.T, aspect="auto", extent=extent, cmap="nipy_spectral", vmin=vmin, vmax=vmax)
axes[1].set_title("Discretized Velocity Model")
axes[1].set_xlim(extent[0], extent[1])
axes[1].set_ylim(extent[2], extent[3])
axes[1].set_xlabel("Distance (m)")

# fig.colorbar(im1, ax=axes, label="Velocity (m/s)", shrink=0.85)
# axes[0].set_ylim(15000, 0)
# axes[1].set_ylim(15000, 0)
plt.tight_layout()
plt.show()
