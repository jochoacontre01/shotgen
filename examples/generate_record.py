from shotgen.sampleshot import ShotRecord, GeoModel, load_marmousi, load_sigsbee
from time import perf_counter

vp = load_sigsbee()
    
vp = vp[1000:2500:2, 250:800:2] #vp[:7000:5, :2000:5]


nx = vp.shape[0]
nz = vp.shape[1]

# nx = 150
# nz = 50
shot = ShotRecord(
    nx=nx,
    nz=nz,
    dx=1,
    dz=1,
    n_sources=6,
    n_receivers=48,
    f0=100,
    # src_origin=(0,10),
    # rec_origin=(0,2),
    src_origin=(0.0,2.0),
    rec_origin=(0.0,2.0),
    group_offset=10.0,
    shot_offset=45.0,
    gather="common shot",
    smooth=5,
    noise_scale=0,
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

shot.show_model(cmap="turbo")

start = perf_counter()
data = shot.run(260) # 550
end = perf_counter()

print(f"Simulation ended after {end-start};.6f seconds")
shot.show_shot(cmap="grey")

shot.save_shot(f"{shot.gather.replace(" ","")}-shot_{nx}nx_{nz}nz_{shot.n_receivers}rec_{shot.n_sources}src_{shot.f0}hz_{shot.group_offset:.0f}goffset_{shot.shot_offset:.0f}soffset_sigsbee.h5")