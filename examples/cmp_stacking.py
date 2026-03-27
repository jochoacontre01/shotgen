from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from shotgen.sampleshot import LoadShot
from pathlib import Path


shotpath = Path(__file__).resolve().parents[1] / "data/shot1_forCMP.h5"

record = LoadShot(shotpath)

data = record.shots
init_velocity_model = record.smooth_velocity
src_loc = record.sources
rec_loc = record.receivers
wavelet = record.wavelet
f0 = record.f0
time = record.time

cmp_near_offset = data[:,0,:]

extent = (0, data.shape[1], np.max(time), 0)
plt.imshow(cmp_near_offset.T, cmap="grey", aspect="auto", extent=extent)
plt.xlabel("Receiver position")
plt.ylabel("Two-way travel time (s)")
plt.title("Near offset CMP stack")
plt.show()