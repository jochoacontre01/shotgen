import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import numpy as np
import matplotlib.pyplot as plt
import cmocean
from pathlib import Path
from scipy.ndimage import laplace
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import ReverseTimeMigration
from shotgen.pylogger import get_logger, setup_file_logging
import argparse
import subprocess
import time

logger = get_logger("rtm_from_data")

def main(cli=False, file=None):
    # 1. Locate and load the Sigsbee shot record file
    if file is not None:
        shotpath = Path(__file__).resolve().parents[1] / file
        if not shotpath.exists():
            data_shotpath = Path(__file__).resolve().parents[1] / "data" / file
            if data_shotpath.exists():
                shotpath = data_shotpath
    else:
        raise ValueError("No data file was passed to the arguments, pass it with 'rtm_from_data.py path/to/file")
    
    setup_file_logging(dataset_dir=shotpath, script_name="rtm_from_data", logger=logger)
    logger.info("Loading shot record from: %s", shotpath)
    
    nbl = 200

    # 3. Initialize native ReverseTimeMigration directly using the SEGY dataset directory
    rtm = ReverseTimeMigration(
        dataset_dir=shotpath,
        nbl=nbl,
        smooth_sigma=5.0,
        # space_order=4
    )
    
    dx_spacing, dz_spacing = rtm.spacing

    # 5. Execute run method
    logger.info("Running Reverse Time Migration...")
    migrated_image = rtm.run(save_wavefield=False, verbose=True, cli=cli)

    # 6. Post-process and Plotting
    # Apply Laplace filter to the migrated image to remove low-frequency acquisition footprints.
    lap_image = laplace(migrated_image)
    
    # Exclude boundary (nbl) for plotting
    plotted_image = lap_image[nbl:-nbl, nbl:-nbl]
    output_npy = shotpath / 'rtm_migrated.npy'
    try:
        np.save(output_npy, migrated_image)
        logger.info("Saved migrated image to: %s", output_npy)
    except OSError:
        fallback_npy = Path("/tmp") / f"{shotpath.name}_rtm_migrated.npy"
        np.save(fallback_npy, migrated_image)
        logger.warning(
            "Dataset directory is read-only; saved migrated image to fallback path: %s",
            fallback_npy,
        )
   
    # np.save("rtm.npy", plotted_image)
    # Compute display extent in meters
    model = rtm.model
    extent = [
        model.origin[0] * dx_spacing, 
        model.origin[0] + model.shape[0] * dx_spacing,
        model.origin[1] + model.shape[1] * dz_spacing, 
        model.origin[1] * dz_spacing
    ]
    from matplotlib.colors import TwoSlopeNorm
    # vmin = np.quantile(laplace(plotted_image), 0.10)
    # vmax = np.quantile(laplace(plotted_image), 0.85)
    # vmin = min(vmin, -1e-10)
    # vmax = max(vmax, 1e-10)
    vmin = np.min(plotted_image)
    vmax = np.max(plotted_image)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    logger.info("Plotting results...")
    logger.info(
        "Image stats - Shape: %s, Max: %s, Min: %s, Mean: %s, Std: %s",
        plotted_image.shape,
        np.amax(plotted_image),
        np.amin(plotted_image),
        np.mean(plotted_image),
        np.std(plotted_image),
    )
    plt.figure(figsize=(10, 5))
    im = plt.imshow(
        plotted_image.T,
        cmap=cmocean.cm.balance,
        extent=extent,
        aspect="auto",
        norm=norm
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
        subprocess.run("chafa -w 9 img.png".split())
        time.sleep(0.5)
        # subprocess.run("rm img.png".split())
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Dataset file to process")
    parser.add_argument("-c", "--cli", action="store_true")
    args = parser.parse_args()
    main(args.cli, args.file)
