import argparse
import os
import subprocess
import sys
import time

import matplotlib

if "--cli" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-base", help="Array to take as base")
parser.add_argument("--cli", action="store_true")

args = parser.parse_args()

if args.base is not None:
    data = np.load(args.base)
else:
    data = np.ones((1000,500))*1500

center = 250
amp = 50
lambdas = 2
x = np.linspace(0, 2*np.pi*lambdas, data.shape[1])
wave = center + amp*np.sin(x)

indices = np.arange(data.shape[0])[:, np.newaxis]
data[indices>wave] = 2500

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
im = ax.imshow(data, cmap="grey", aspect="auto")
plt.tight_layout()

if args.cli:
    plt.savefig("img.png")
    plt.close(fig)
    subprocess.run(["chafa", "img.png"], check=False)
    time.sleep(0.5)
    os.remove("img.png")
else:
    plt.show()


