import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import segyio
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
import os
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
plt.figure(figsize=(10,10))
velim = plt.imshow(new_array.T, aspect="auto", extent=extent, cmap="nipy_spectral")

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

base_dir = "assets/fpb_horizons"
for horizon in sorted(os.listdir(base_dir)):
    path = os.path.join(base_dir, horizon)
    if not os.path.isfile(path):
        continue
    x_plot, z_plot = load_interpretation(path, trace_x, trace_y)
    plt.plot(x_plot, z_plot, linewidth=1.0, label=horizon)

plt.xlim(extent[0], extent[1])
plt.ylim(extent[2], extent[3])
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.colorbar(velim, label="Velociy (m/s)")
plt.tight_layout()
plt.legend()
plt.show()
