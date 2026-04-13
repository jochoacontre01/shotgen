
from shotgen.migration import ReverseTimeMigration, ReverseTimeMigrationGPU
from shotgen.sampleshot import GeoModel, LoadShot
from pypalettes import create_cmap
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import laplace

mig_cmap = create_cmap(
    colors=[
        "#E33008",
        "#EAA51D",
        "#F1F1F1",
        "#4B4B4B",
        "#000000",
    ],
    cmap_type="continuous"
)

# ? With devito propagation

nx = 200
nz = 150
nbl = 200
t0 = 0
tn = 250
smooth_sigma = 5
vp = GeoModel(nx, nz).diapir()

rtm = ReverseTimeMigration(
    vp=vp,
    n_receivers=nx,
    n_sources=3,
    origin=(0,0),
    spacing=(1,1),
    nbl=nbl,
    t0=t0,
    tn=tn,
    f0=150.,
    smooth_sigma=smooth_sigma,
    space_order=8
)

migrated = rtm.run(save_wavefield=True, save_each=20)

vs = rtm.vs
us = rtm.us
vs = np.sum(vs, axis=0)
us = np.sum(us, axis=0)
# vmin = np.quantile(vs, 0.10)
# vmax = np.quantile(vs, 0.90)
vmin = np.amin(us)
vmax = np.amax(us)

corrvmin = np.quantile(us*vs, 0.05)
corrvmax = np.quantile(us*vs, 0.95)
model = rtm.model
domain_size = 1.e-3 * np.array(model.domain_size)
extent = [model.origin[0], model.origin[0] + domain_size[0],
            model.origin[1] + domain_size[1], model.origin[1]]

plt.figure(figsize=(15,8))
try:
    while True:
        correlation = 0
        for u, v in zip(us[::-1], vs[::-1]):
            correlation += laplace(u*v)
            plt.cla()
            plt.imshow(u[nbl:-nbl,nbl:-nbl].T, norm=TwoSlopeNorm(0,vmin,vmax), cmap="gray", aspect="equal", extent=extent, alpha=0.5)
            plt.imshow(v[nbl:-nbl,nbl:-nbl].T, norm=TwoSlopeNorm(0,vmin,vmax), cmap="gray", aspect="equal", extent=extent, alpha=0.5)
            plt.imshow(correlation[nbl:-nbl,nbl:-nbl].T, norm=TwoSlopeNorm(0,corrvmin,corrvmax), cmap="gray", aspect="equal", extent=extent, alpha=0.5)
            # plt.imshow(vp.T, cmap="turbo", extent=extent, alpha=0.5)
            plt.xlabel("X position (km)")
            plt.ylabel("Depth (km)")
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
except KeyboardInterrupt:
    plt.show()


# ? With deepwave propagation

# pathfile = Path(__file__).resolve().parents[1] / "data/shot1.h5"
# shotobject = LoadShot(pathfile)
# data = shotobject.shots
# init_velocity_model = shotobject.smooth_velocity
# src_loc = shotobject.sources
# rec_loc = shotobject.receivers
# wavelet = shotobject.wavelet
# f0 = shotobject.f0
# time = shotobject.time

# rtm = ReverseTimeMigrationGPU(
#     shot_record=data,
#     velocity_model=init_velocity_model,
#     sources=src_loc,
#     receivers=rec_loc,
#     wavelet=wavelet,
#     f0=f0,
#     time=time
# )

# migrated = rtm.run(save)
