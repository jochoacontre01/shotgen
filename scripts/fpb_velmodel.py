import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy.signal import convolve2d

file = "assets/CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908.segy"

with segyio.open(file, ignore_geometry=True) as f:
    data = f.trace.raw[:]
    dz = segyio.tools.dt(f) / 1000.

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

np.save("assets/CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908_moho.npy", new_array)
plt.imshow(new_array.T, aspect="auto", extent=extent, cmap="RdBu_r")
plt.show()

# linear interoplation between 5000 and bottom at 8000 

