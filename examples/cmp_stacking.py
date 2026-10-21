from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from shotgen.sampleshot import LoadShotRecord
from shotgen.utils import setup_matplotlib_style
from pathlib import Path


setup_matplotlib_style()
shotpath = Path(__file__).resolve().parents[1] / "data/shot1_forCMP_dataset"

record = LoadShotRecord(shotpath)

data = record.shots
init_velocity_model = record.smooth_velocity
src_loc = record.sources
rec_loc = record.receivers
wavelet = record.wavelet
f0 = record.f0
time = record.time

cmp_near_offset = data[:, 0, :]

extent = (0, data.shape[1], np.max(time), 0)
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
im = ax.imshow(cmp_near_offset.T, cmap="grey", aspect="auto", extent=extent)
ax.set_xlabel("Receiver position")
ax.set_ylabel("Two-way travel time (s)")
ax.set_title("Near offset CMP stack")
plt.tight_layout()
plt.show()