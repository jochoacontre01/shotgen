import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("-cli", action="store_true")
args = parser.parse_args()

data = np.load("examples/rtm.npy")

plt.figure(figsize=(8,5))
plt.imshow(data.T, aspect="auto", cmap="gray")
plt.colorbar()
if args.cli:
    plt.savefig("img.png")
    subprocess.run("chafa -w 9 img.png".split())
    subprocess.run("rm img.png".split())
else:
    plt.show()
