import numpy as np
import matplotlib.pyplot as plt
import argparse
import subprocess
from shotgen.sampleshot import load_sigsbee

def main():
    parser = argparse.ArgumentParser(description="Read and plot velocity model data.")
    parser.add_argument("-c", action="store_true", help="Display the image in the terminal using timg")
    args = parser.parse_args()

    # Load data
    # Dimensions discovered: nx = 5395, nz = 1911
    nx, nz = 5395, 1911
    filename = "assets/vel_z6.25m_x12.5m_exact.bin"
    data = np.fromfile(filename, dtype=np.float32).reshape(nx, nz)

    # Plotting
    dx = 12.5
    dz = 6.25

    data, metadata = load_sigsbee()
    data = data.astype(np.float32)
    nx, nz = data.shape
    dx, dz = metadata["dx"], metadata["dz"]
    extent = (0, nx * dx, nz * dz, 0)

    plt.figure(figsize=(10, 5))
    im = plt.imshow(data.T, extent=extent, cmap="turbo", aspect="auto")
    plt.colorbar(im, label="Velocity (m/s)")
    plt.xlabel("X (m)")
    plt.ylabel("Depth Z (m)")
    plt.title("Velocity Model (Exact)")
    plt.tight_layout()

    if args.c:
        plt.savefig("image.png")
        subprocess.run(["timg", "image.png"])
        subprocess.run(["rm", "image.png"])
    else:
        plt.show()

if __name__ == "__main__":
    main()
