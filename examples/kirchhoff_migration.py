from pathlib import Path
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import KirchhoffMigration
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
import numpy as np
import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("-cli", action="store_true", help="setup runtime for non-gui interface")
args = parser.parse_args()

shotpath = Path(__file__).resolve().parents[1] / "data/commonshot-shot_750nx_50nz_32rec_6src_250hz_10goffset_45soffset_sigsbee_dataset"

dx_spacing = 1.0 
dz_spacing = 1.0 

migrator = KirchhoffMigration(
    dataset_dir=shotpath,
    spacing=(dx_spacing, dz_spacing)
)

image = migrator.run()

vmin = np.quantile(laplace(image), 0.10)
vmax = np.quantile(laplace(image), 0.85)
plt.imshow(laplace(image.T), cmap="gray", extent=[0, migrator.vp.shape[1]*dx_spacing, migrator.vp.shape[0]*dz_spacing, 0], aspect="auto", vmin=vmin, vmax=vmax)
plt.colorbar(label="Amplitude")
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.title("Kirchhoff Migration Image")
if args.cli:
    plt.savefig("img.png")
    subprocess.run("timg img.png".split())
    time.sleep(0.5)
    subprocess.run("rm img.png".split())
else:    
    plt.show()
