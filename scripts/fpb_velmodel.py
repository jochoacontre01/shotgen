import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.sparse import diags
from scipy.sparse.linalg import LinearOperator, cg
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


def _fit_compaction_trend(
    bg_vel,
    dz,
    water_vel_max=1550.0,
    basement_vel_min=4990.0,
    poly_deg=2,
):
    """
    Fit a smooth 1D sedimentary compaction trend as a function of depth below seafloor.

    Returns
    -------
    trend_2d : np.ndarray
        2D array of shape (nx, nz) giving the smooth background compaction velocity
        below the seafloor, and 0.0 in the water column (where no compaction trend applies).
    seafloor_iz : np.ndarray
        1D integer array of length nx giving the detected seafloor depth index per trace.
    """
    _, nz = bg_vel.shape
    z_grid = np.arange(nz, dtype=float) * dz

    # Detect seafloor index per trace (first sample exceeding water velocity threshold)
    above_water = bg_vel > water_vel_max
    has_sed = np.any(above_water, axis=1)
    seafloor_iz = np.where(has_sed, np.argmax(above_water, axis=1), 0)
    z_sf = seafloor_iz.astype(float) * dz

    # Depth below seafloor (m), 0 in water column
    z_bsf = np.maximum(0.0, z_grid[None, :] - z_sf[:, None])

    # Fit polynomial strictly on the sedimentary section (below water, above constant basement)
    sed_mask = (bg_vel > water_vel_max) & (bg_vel < basement_vel_min)
    if np.count_nonzero(sed_mask) < poly_deg + 1:
        return np.zeros_like(bg_vel, dtype=float), seafloor_iz

    z_fit = z_bsf[sed_mask]
    v_fit = bg_vel[sed_mask]
    z_max_sed = float(np.percentile(z_fit, 99.5))

    coeffs = np.polyfit(z_fit, v_fit, deg=poly_deg)
    poly = np.poly1d(coeffs)

    # Evaluate smoothly across the sedimentary section, clamping z_bsf at the per-trace
    # basement depth so the steeper sedimentary compaction gradient is not projected into
    # the crystalline crust below basement.
    below_base = bg_vel >= basement_vel_min
    has_base = np.any(below_base, axis=1)
    base_iz = np.where(has_base, np.argmax(below_base, axis=1), nz - 1)
    z_base_bsf = np.minimum(np.maximum(0.0, (base_iz - seafloor_iz).astype(float) * dz), z_max_sed)
    z_bsf_clamped = np.minimum(z_bsf, z_base_bsf[:, None])
    trend_2d = poly(z_bsf_clamped)

    # Assign constant reference water velocity in the water column so residuals remain
    # consistently near zero everywhere (even if a horizon slightly straddles the seafloor).
    water_mask = z_grid[None, :] < z_sf[:, None]
    if np.any(water_mask):
        water_ref_vel = float(np.median(bg_vel[water_mask]))
        trend_2d[water_mask] = water_ref_vel

    return trend_2d, seafloor_iz


def discretize_velocity_model(
    bg_vel,
    horizon_depths,
    dz,
    stat="median",
    lateral_smooth_sigma=15.0,
    taper_traces=60,
    smooth_sigma=0.0,
    detrend_compaction=True,
    use_slowness=True,
    compaction_deg=2,
    water_vel_max=1550.0,
    basement_vel_min=4990.0,
):
    """
    Discretize a smooth background velocity model into sharp horizon-bounded layers.

    When ``detrend_compaction=True`` (Idea 1), a smooth global 1D sedimentary compaction
    trend ``v_trend(z_bsf)`` (as a function of depth below seafloor) is removed prior to
    per-layer aggregation and added back afterward. This decouples layer thickness variations
    from lateral velocity changes while preserving realistic vertical compaction gradients
    and crisp horizon impedance contrasts.

    Parameters
    ----------
    bg_vel : np.ndarray
        Background velocity model of shape (nx, nz).
    horizon_depths : list[np.ndarray]
        List of 1D arrays of length nx giving horizon depth in meters (NaN where absent).
    dz : float
        Vertical grid spacing in meters.
    stat : {"median", "mean"}
        Statistic used to summarize background velocity (or residual) within each layer.
    lateral_smooth_sigma : float
        Lateral Gaussian smoothing sigma (in traces) applied to the background model before
        layer aggregation, keeping vertical horizon jumps crisp while smoothing lateral jitter.
    taper_traces : int
        Cosine taper width (in traces) at partial-horizon terminations to prevent artificial
        vertical velocity steps.
    smooth_sigma : float
        Optional final 2D Gaussian smoothing sigma (in grid cells). Default 0.0 (crisp 1-cell steps).
    detrend_compaction : bool
        If True, remove a global depth-below-seafloor compaction trend before layer aggregation
        and add it back afterward.
    use_slowness : bool
        If True, perform layer aggregation in slowness (1 / v) to preserve vertical traveltimes.
    compaction_deg : int
        Polynomial degree for the 1D sedimentary compaction trend (default 2).
    water_vel_max : float
        Velocity threshold (m/s) separating water column from sediments.
    basement_vel_min : float
        Upper velocity threshold (m/s) excluding constant basement from the sedimentary trend fit.

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

    if detrend_compaction:
        trend_2d, _ = _fit_compaction_trend(
            bg_work,
            dz=dz,
            water_vel_max=water_vel_max,
            basement_vel_min=basement_vel_min,
            poly_deg=compaction_deg,
        )
        has_trend = trend_2d > 0.0
    else:
        trend_2d = np.zeros_like(bg_work)
        has_trend = np.zeros((nx, nz), dtype=bool)

    if use_slowness:
        work_field = np.where(has_trend, (1.0 / bg_work) - (1.0 / np.where(has_trend, trend_2d, 1.0)), 1.0 / bg_work)
    else:
        work_field = bg_work - trend_2d

    h_depths = np.asarray(horizon_depths, dtype=float)
    h_weights = np.array([_compute_termination_weights(h, taper_traces=taper_traces) for h in h_depths])

    disc_vel = np.empty((nx, nz), dtype=np.float32)

    for ix in range(nx):
        col_work = work_field[ix]
        col_trend = trend_2d[ix]
        col_has_trend = has_trend[ix]

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
        col_agg = np.empty(nz, dtype=float)

        for a, b in itertools.pairwise(bounds):
            col_agg[a:b] = stat_fn(col_work[a:b])

        for k, (iz, w) in enumerate(zip(unique_iz, unique_w)):
            if w < 1.0:
                a = bounds[k]
                b = bounds[k + 2]
                v_parent = stat_fn(col_work[a:b])
                col_agg[a:iz] = (1.0 - w) * v_parent + w * col_agg[a:iz]
                col_agg[iz:b] = (1.0 - w) * v_parent + w * col_agg[iz:b]

        if use_slowness:
            col_slowness = np.where(
                col_has_trend,
                (1.0 / np.where(col_has_trend, col_trend, 1.0)) + col_agg,
                col_agg,
            )
            col_out = 1.0 / np.maximum(col_slowness, 1e-6)
        else:
            col_out = col_trend + col_agg

        disc_vel[ix] = col_out

    if smooth_sigma > 0:
        disc_vel = gaussian_filter(disc_vel, sigma=smooth_sigma).astype(np.float32)

    return disc_vel


def discretize_velocity_model_sparse(
    bg_vel,
    horizon_depths,
    dz,
    lambda_x=500.0,
    lambda_z=1000.0,
    taper_traces=60,
    detrend_compaction=True,
    use_slowness=True,
    compaction_deg=2,
    water_vel_max=1550.0,
    basement_vel_min=4990.0,
    max_solve_depth=None,
    cg_rtol=1e-5,
    cg_maxiter=1000,
    smooth_sigma=0.0,
):
    """
    Discretize a smooth background velocity model via a horizon-blocked 2D sparse solver (Idea 4).

    Solves the symmetric positive-definite regularized least-squares system:

        min_u  ||u - u_bg||_2^2 + lambda_x ||W_x D_x u||_2^2 + lambda_z ||W_z D_z u||_2^2

    where ``u`` is the de-trended slowness (or velocity) residual, ``W_z`` drops to 0 across
    interpreted horizons (allowing sharp vertical impedance jumps), and ``W_x`` drops to 0 on
    horizontal edges crossed by dipping/faulted horizons (preventing cross-horizon smearing).
    At partial-horizon terminations, weights taper smoothly back to 1 so the 2D solver blends
    around horizon tips without vertical curtain artifacts.

    Parameters
    ----------
    bg_vel : np.ndarray
        Background velocity model of shape (nx, nz).
    horizon_depths : list[np.ndarray]
        List of 1D arrays of length nx giving horizon depth in meters (NaN where absent).
    dz : float
        Vertical grid spacing in meters.
    lambda_x : float
        Lateral smoothness weight within layers. Larger values enforce stronger lateral continuity.
    lambda_z : float
        Vertical uniformity weight within layers. Larger values push each layer closer to a
        constant residual on top of the compaction trend.
    taper_traces : int
        Cosine taper width (in traces) at partial-horizon terminations.
    detrend_compaction : bool
        If True, remove a global depth-below-seafloor compaction trend before solving and add
        it back afterward.
    use_slowness : bool
        If True, solve in slowness (1 / v) to conserve vertical traveltimes within bounded layers.
    compaction_deg : int
        Polynomial degree for the 1D sedimentary compaction trend.
    water_vel_max : float
        Velocity threshold (m/s) separating water column from sediments.
    basement_vel_min : float
        Upper velocity threshold (m/s) excluding constant basement from the sedimentary trend fit.
    max_solve_depth : float or None
        Optional maximum depth (m) for the 2D sparse solve (cells below copy ``bg_vel``).
        Useful when deep crust/mantle below the deepest horizon is overwritten afterward.
    cg_rtol : float
        Relative convergence tolerance for Conjugate Gradient.
    cg_maxiter : int
        Maximum Conjugate Gradient iterations.
    smooth_sigma : float
        Optional final 2D Gaussian smoothing sigma (in grid cells). Default 0.0.

    Returns
    -------
    np.ndarray
        Discretized velocity model of shape (nx, nz), dtype float32.
    """
    nx, nz = bg_vel.shape
    nz_s = nz if max_solve_depth is None else min(nz, max(2, int(np.ceil(max_solve_depth / dz))))

    bg_work = bg_vel[:, :nz_s].astype(np.float64)

    if detrend_compaction:
        trend_2d, _ = _fit_compaction_trend(
            bg_work,
            dz=dz,
            water_vel_max=water_vel_max,
            basement_vel_min=basement_vel_min,
            poly_deg=compaction_deg,
        )
        has_trend = trend_2d > 0.0
    else:
        trend_2d = np.zeros_like(bg_work)
        has_trend = np.zeros((nx, nz_s), dtype=bool)

    if use_slowness:
        work_field = np.where(
            has_trend,
            (1.0 / bg_work) - (1.0 / np.where(has_trend, trend_2d, 1.0)),
            1.0 / bg_work,
        )
    else:
        work_field = bg_work - trend_2d

    # Build 2D horizon-blocked edge weights in [0, 1]
    wz = np.ones((nx, nz_s - 1), dtype=np.float64)
    wx = np.ones((nx - 1, nz_s), dtype=np.float64)

    for h in horizon_depths:
        h_arr = np.asarray(h, dtype=float)
        w_h = _compute_termination_weights(h_arr, taper_traces=taper_traces)

        # Block vertical edges crossed by the horizon
        valid = ~np.isnan(h_arr)
        if np.any(valid):
            ix_v = np.nonzero(valid)[0]
            iz_v = np.round(h_arr[valid] / dz).astype(int)
            in_bounds = (iz_v >= 1) & (iz_v < nz_s)
            if np.any(in_bounds):
                np.minimum.at(
                    wz,
                    (ix_v[in_bounds], iz_v[in_bounds] - 1),
                    1.0 - w_h[ix_v[in_bounds]],
                )

        # Block horizontal edges crossed by dipping/faulted horizon segments
        both = valid[:-1] & valid[1:]
        if np.any(both):
            ix_b = np.nonzero(both)[0]
            iz0 = np.round(h_arr[ix_b] / dz).astype(int)
            iz1 = np.round(h_arr[ix_b + 1] / dz).astype(int)
            dipping = iz0 != iz1
            if np.any(dipping):
                w_pair = np.minimum(w_h[ix_b[dipping]], w_h[ix_b[dipping] + 1])
                for ix, a, b, wp in zip(ix_b[dipping], iz0[dipping], iz1[dipping], w_pair):
                    lo = max(0, min(a, b))
                    hi = min(nz_s, max(a, b))
                    if lo < hi:
                        wx[ix, lo:hi] = np.minimum(wx[ix, lo:hi], 1.0 - wp)

    # Assemble SPD system: (I + lambda_x * L_x + lambda_z * L_z) u = work_field
    diag = np.ones((nx, nz_s), dtype=np.float64)
    diag[:, :-1] += lambda_z * wz
    diag[:, 1:] += lambda_z * wz
    diag[:-1, :] += lambda_x * wx
    diag[1:, :] += lambda_x * wx

    wz_pad = np.pad(wz, ((0, 0), (0, 1)), constant_values=0.0)
    off_z = (-lambda_z * wz_pad.ravel())[:-1]
    off_x = -lambda_x * wx.ravel()
    diag_flat = diag.ravel()

    n_total = nx * nz_s
    mat_a = diags(
        [off_x, off_z, diag_flat, off_z, off_x],
        offsets=[-nz_s, -1, 0, 1, nz_s],
        shape=(n_total, n_total),
        format="csr",
    )

    inv_diag = 1.0 / diag_flat
    precond = LinearOperator((n_total, n_total), matvec=lambda v: inv_diag * v, dtype=np.float64)

    rhs = work_field.ravel()
    sol_flat, _ = cg(mat_a, rhs, x0=rhs, rtol=cg_rtol, maxiter=cg_maxiter, M=precond)
    sol_agg = sol_flat.reshape((nx, nz_s))

    if use_slowness:
        sol_slowness = np.where(
            has_trend,
            (1.0 / np.where(has_trend, trend_2d, 1.0)) + sol_agg,
            sol_agg,
        )
        sol_vel = 1.0 / np.maximum(sol_slowness, 1e-6)
    else:
        sol_vel = trend_2d + sol_agg

    disc_vel = bg_vel.astype(np.float32, copy=True)
    disc_vel[:, :nz_s] = sol_vel.astype(np.float32)

    if smooth_sigma > 0:
        disc_vel = gaussian_filter(disc_vel, sigma=smooth_sigma).astype(np.float32)

    return disc_vel


base_dir = "assets/fpb_horizons"
if not any(os.path.isfile(os.path.join(base_dir, f)) for f in os.listdir(base_dir)):
    nested_dir = os.path.join(base_dir, "fpb_horizons")
    if os.path.isdir(nested_dir):
        base_dir = nested_dir

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

# Solve only down to just below the deepest interpreted horizon (deep crust/Moho is overwritten below)
max_horizon_depth = max(float(np.nanmax(z_plot)) for _, z_plot in horizons.values()) + 500.0

disc_vel = discretize_velocity_model_sparse(
    new_array,
    all_boundaries,
    dz=dz,
    lambda_x=500.0,
    lambda_z=5000.0,
    taper_traces=60,
    detrend_compaction=True,
    use_slowness=True,
    compaction_deg=2,
    max_solve_depth=max_horizon_depth,
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
