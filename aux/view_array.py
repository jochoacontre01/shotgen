import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
from pathlib import Path

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

plt.figure(figsize=(10,5))
plt.imshow(data.T, aspect="auto", cmap="gray")
plt.tight_layout()
plt.colorbar()
if args.cli:
    plt.savefig("img.png")
    subprocess.run("chafa -w 9 img.png".split())
    if not args.savefig:
        subprocess.run("rm img.png".split())
else:
    plt.show()
