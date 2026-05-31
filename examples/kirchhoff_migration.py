from pathlib import Path
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import KirchhoffMigration
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import numpy as np

shotpath = Path(__file__).resolve().parents[1] / "data/commonshot-shot_750nx_275nz_48rec_6src_100hz_10goffset_45soffset_sigsbee.h5"
data = LoadShotRecord(shotpath)
shots = data.shots
velocity = data.smooth_velocity
v_real = data.velocity_model # Load real velocity for plotting comparison
sources = data.sources
receivers = data.receivers
time = data.time

dx_spacing = 1.0 
dz_spacing = 1.0 

migrator = KirchhoffMigration(
    vp=velocity,
    sources=sources,
    receivers=receivers,
    shots=shots,
    time=time,
    spacing=(dx_spacing, dz_spacing)
)

image = migrator.run()

vmin = np.quantile(laplace(image), 0.10)
vmax = np.quantile(laplace(image), 0.85)
plt.imshow(laplace(image.T), cmap="gray", extent=[0, velocity.shape[1]*dx_spacing, velocity.shape[0]*dz_spacing, 0], aspect="auto", vmin=vmin, vmax=vmax)
plt.colorbar(label="Amplitude")
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.title("Kirchhoff Migration Image")
plt.show()
