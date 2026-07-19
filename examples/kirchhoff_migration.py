from pathlib import Path
import re
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import KirchhoffMigration
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cmocean
from scipy.ndimage import laplace
import numpy as np
import argparse
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Dataset file to process")
parser.add_argument("-c","--cli", action="store_true", help="setup runtime for non-gui interface")
args = parser.parse_args()

if args.file is not None:
    shotpath = Path(__file__).resolve().parents[1] / args.file
else:
    raise ValueError("No data file was passed to the arguments, pass it with 'kirchhoff_migration.py path/to/file")

name = shotpath.name
dx_match = re.search(r'_(\d+(?:\.\d+)?)dx_', name)
dz_match = re.search(r'_(\d+(?:\.\d+)?)dz_', name)

if dx_match:
    dx_spacing = float(dx_match.group(1))
else:
    dxdz_match = re.search(r'_(\d+(?:\.\d+)?)dxdz_', name)
    dx_spacing = float(dxdz_match.group(1)) if dxdz_match else 10.0

if dz_match:
    dz_spacing = float(dz_match.group(1))
else:
    dxdz_match = re.search(r'_(\d+(?:\.\d+)?)dxdz_', name)
    dz_spacing = float(dxdz_match.group(1)) if dxdz_match else 10.0

migrator = KirchhoffMigration(
    dataset_dir=shotpath,
    spacing=(dx_spacing, dz_spacing)
)

image = migrator.run()
np.save(f"data/migrated/{shotpath.name}_Kirchhoff_migrated.npy", image)


model = migrator.model
plt.figure(figsize=(10, 5))
extent = [
    model.origin[0], 
    model.origin[0] + model.shape[0] * dx_spacing,
    model.origin[1] + model.shape[1] * dz_spacing, 
    model.origin[1]
]
# vmin = np.quantile(laplace(image), 0.10)
# vmax = np.quantile(laplace(image), 0.85)
vmin = np.min(laplace(image))
vmax = np.max(laplace(image))
# vmin = min(vmin, -1e-10)
# vmax = max(vmax, 1e-10)
norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
im = plt.imshow(laplace(image.T), cmap=cmocean.cm.balance_r, extent=extent, norm=norm)
plt.colorbar(im, label="Amplitude", shrink=0.75)
plt.xlabel("Distance (m)")
plt.ylabel("Depth (m)")
plt.title("Kirchhoff Migration Image")
plt.gca().set_aspect("auto")
plt.tight_layout()
if args.cli:
    plt.savefig("img.png")
    subprocess.run("timg img.png".split())
    time.sleep(0.5)
    subprocess.run("rm img.png".split())
else:    
    plt.show()
