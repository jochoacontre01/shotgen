from shotgen.sampleshot import ShotRecord, load_marmousi, load_sigsbee
from shotgen.models import GeoModel
from time import perf_counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-cli", action="store_true", help="setup runtime for non-gui interface")
args = parser.parse_args()

vp = load_marmousi()
dx = 5

# vp = GeoModel(750, 50).foothills(
vp = vp[3000:9000:dx, 250:2000:dx] #vp[:7000:5, :2000:5]


nx = vp.shape[0]
nz = vp.shape[1]

# nx = 150
# nz = 50
shot = ShotRecord(
    nx=nx,
    nz=nz,
    dx=dx,
    dz=dx,
    n_sources=4,
    n_receivers=32,
    f0=30, # high frequency yields numerical instability
    # src_origin=(0,10),
    # rec_origin=(0,2),
    src_origin=(0.0,2.0),
    rec_origin=(0.0,2.0),
    group_offset=50.0, # offset between receivers
    shot_offset=300.0, # offset between shots
    gather="common shot",
    smooth=5,
    snr=10,
    fd_order=8,
    n_damping=200,
    engine="pylops"
)
# model = GeoModel(nx, nz).layer_model()

shot.set_model(vp)
# n_receivers = 100
# theta = np.linspace(0, 2 * np.pi, n_receivers, endpoint=False)
# radius = nz / 3
# center_x = nx / 2
# center_y = nz / 2
# x_pos = center_x + radius * np.cos(theta)
# y_pos = center_y + radius * np.sin(theta)

# shot.set_receiver_position(x_pos, y_pos)

shot.show_model(cmap="turbo", cli=args.cli)

start = perf_counter()
data = shot.run(1000) # 550
end = perf_counter()

print(f"Simulation ended after {end-start};.6f seconds")
shot.show_shot(cmap="grey", cli=args.cli)

shot.save_shot(f"data/{shot.gather.replace(" ","")}-shot_{nx}nx_{nz}nz_{shot.n_receivers}rec_{shot.n_sources}src_{shot.f0}hz_{shot.group_offset:.0f}goffset_{shot.shot_offset:.0f}soffset_snr{shot.snr}")
