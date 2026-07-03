from shotgen.sampleshot import ShotRecord, load_marmousi, load_sigsbee
from shotgen.models import GeoModel
from time import perf_counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cli", action="store_true", help="Setup runtime for non-gui interface")
parser.add_argument("-y", "--yes", action="store_true", help="Proceed with modeling after showing model geometry.")
parser.add_argument("-s", "--save", action="store_true", help="Save the model to disk or discard after modeling.")
args = parser.parse_args()

vp = load_marmousi()
dx = 10

# vp = GeoModel(750, 50).foothills(
vp = vp[5000:12000:dx, ::dx] #vp[:7000:5, :2000:5]
print(vp.shape)


nx = vp.shape[0]
nz = vp.shape[1]

# nx = 150
# nz = 50
shot = ShotRecord(
    nx=nx,
    nz=nz,
    dx=dx,
    dz=dx,
    n_sources=5,
    n_receivers=24,
    f0=30, # high frequency yields numerical instability
    # src_origin=(0,10),
    # rec_origin=(0,2),
    src_origin=(500.0,2.0),
    rec_origin=(500.0,2.0),
    origin=(5000.0, 0.0),
    group_offset=120.0, # offset between receivers
    shot_offset=400.0, # offset between shots
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

if args.yes:
    start = perf_counter()
    data = shot.run(3000) # 550
    end = perf_counter()

    print(f"Simulation ended after {end-start};.6f seconds")
    shot.show_shot(cmap="grey", cli=args.cli)

if args.save:
    shot.save_shot(f"data/{shot.gather.replace(" ","")}-shot_{nx}nx_{nz}nz_{shot.n_receivers}rec_{shot.n_sources}src_{shot.f0}hz_{shot.group_offset:.0f}goffset_{shot.shot_offset:.0f}soffset_{shot.snr}snr")
