
from shotgen.migration import ReverseTimeMigration, ReverseTimeMigrationGPU
from shotgen.sampleshot import GeoModel, LoadShot
from pypalettes import create_cmap
from pathlib import Path

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

# nx = 200
# nz = 150
# nbl = 200
# t0 = 0
# tn = 250
# smooth_sigma = 5
# vp = GeoModel(nx, nz).circle_model(10)

# rtm = ReverseTimeMigration(
#     vp=vp,
#     n_receivers=nx,
#     n_sources=5,
#     origin=(0,0),
#     spacing=(1,1),
#     nbl=nbl,
#     t0=t0,
#     tn=tn,
#     f0=150.,
#     smooth_sigma=smooth_sigma,
#     space_order=8
# )

# migrated = rtm.run(save_wavefield=False, save_each=20)

# ? With deepwave propagation

pathfile = Path(__file__).resolve().parents[1] / "data/shot1.h5"
shotobject = LoadShot(pathfile)
data = shotobject.shots
init_velocity_model = shotobject.smooth_velocity
src_loc = shotobject.sources
rec_loc = shotobject.receivers
wavelet = shotobject.wavelet
f0 = shotobject.f0
time = shotobject.time

rtm = ReverseTimeMigrationGPU(
    shot_record=data,
    velocity_model=init_velocity_model,
    sources=src_loc,
    receivers=rec_loc,
    wavelet=wavelet,
    f0=f0,
    time=time
)

migrated = rtm.run()
