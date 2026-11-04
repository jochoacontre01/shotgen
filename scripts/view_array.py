import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from shotgen.utils import setup_matplotlib_style

setup_matplotlib_style()

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Numpy array to view")
parser.add_argument("--cli", action="store_true")
parser.add_argument("--savefig", action="store_true")
args = parser.parse_args()

filepath = args.file
if filepath is not None:
    datapath = Path(__file__).resolve().parents[1] / filepath
else:
    raise ValueError(
        "No data was passed to the script, pass it with view_array.py path/to/file.npy"
    )
data = np.load(datapath)

fig = plt.figure(figsize=(10, 5), dpi=150)
plt.imshow(data.T, aspect="auto", cmap="gray", origin="upper")
plt.tight_layout()
plt.colorbar()
plt.savefig("img.png", dpi=150)
if args.cli:
    plt.close(fig)
    subprocess.run(["chafa", "-w", "9", "img.png"], check=False)
else:
    plt.show()

if not args.savefig:
    subprocess.run(["rm", "img.png"], check=False)
