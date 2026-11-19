import matplotlib.pyplot as plt
import numpy as np
import segyio

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

first_idx = np.argmax(~np.isnan(masked), axis=1)

for i, idx in enumerate(first_idx):
    new_array[i, idx:] = 5000
new_array[:, -1] = 8000

plt.imshow(new_array.T, aspect="auto")
plt.show()

# linear interoplation between 5000 and bottom at 8000 



# plt.imshow(masked.T)
# plt.show()
