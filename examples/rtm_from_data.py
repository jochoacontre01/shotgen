import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import ReverseTimeMigration
import argparse
import subprocess
import time

def main(cli=False, file=None):
    # 1. Locate and load the Sigsbee shot record file
    if file is not None:
        shotpath = Path(__file__).resolve().parents[1] / file
    else:
        raise ValueError("No data file was passed to the arguments, pass it with 'rtm_from_data.py -f path/to/file")
    print(f"Loading shot record from: {shotpath}")
    
    nbl = 200

    # 3. Initialize native ReverseTimeMigration directly using the SEGY dataset directory
    rtm = ReverseTimeMigration(
        dataset_dir=shotpath,
        nbl=nbl,
        smooth_sigma=5.0,
        space_order=4,
    )
    
    dx_spacing, dz_spacing = rtm.spacing

    # 5. Execute run method
    print("Running Reverse Time Migration...")
    migrated_image = rtm.run(save_wavefield=False)
    print(migrated_image)

    # 6. Post-process and Plot
    # Apply Laplace filter to the migrated image to remove low-frequency acquisition footprints.
    lap_image = laplace(migrated_image)
    
    # Exclude boundary (nbl) for plotting
    plotted_image = lap_image[nbl:-nbl, nbl:-nbl]
   
    # np.save("rtm.npy", plotted_image)
    # Compute display extent in meters
    model = rtm.model
    print(model.origin, model.shape, rtm.spacing, nbl)
    extent = [
        model.origin[0] * dx_spacing, 
        model.origin[0] + model.shape[0] * dx_spacing,
        model.origin[1] + model.shape[1] * dz_spacing, 
        model.origin[1] * dz_spacing
    ]
    vmin = np.quantile(laplace(plotted_image), 0.10)
    vmax = np.quantile(laplace(plotted_image), 0.85)
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        plotted_image.T,
        cmap="gray",
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax
    )
    plt.colorbar(im, label="Amplitude", shrink=0.75)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title("RTM Migration Image")
    plt.tight_layout()
    
    # plt.savefig(output_plot_path, bbox_inches="tight")
    # print(f"Plot saved to: {output_plot_path}")
    if cli:
        plt.savefig("img.png")
        subprocess.run("timg img.png".split())
        time.sleep(0.5)
        subprocess.run("rm img.png".split())
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Dataset file to process")
    parser.add_argument("-c", "--cli", action="store_true")
    args = parser.parse_args()
    main(args.cli, args.file)
