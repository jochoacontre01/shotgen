import os

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import argparse
import subprocess
import time
from pathlib import Path

import cmocean
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace
from shotgen.migration import ReverseTimeMigration
from shotgen.pylogger import get_logger, log_array_stats, setup_file_logging
from shotgen.utils import setup_matplotlib_style

logger = get_logger("rtm_from_data")

def main(cli=False, file=None, t_start=None, t_end=None, t_taper=50.0, n_sources=None):
    setup_matplotlib_style()
    # 1. Locate and load the Sigsbee shot record file
    if file is not None:
        shotpath = Path(__file__).resolve().parents[1] / file
        if not shotpath.exists():
            data_shotpath = Path(__file__).resolve().parents[1] / "data" / file
            if data_shotpath.exists():
                shotpath = data_shotpath
    else:
        raise ValueError("No data file was passed to the arguments, pass it with 'rtm_from_data.py path/to/file")
    
    log_path = setup_file_logging(dataset_dir=shotpath, script_name="rtm_from_data", logger=logger)
    timestamp = log_path.name[:15]
    logger.info("Loading shot record from: %s", shotpath)
    if t_start is not None or t_end is not None:
        logger.info("Migration time window: t_start=%s ms, t_end=%s ms, t_taper=%s ms", t_start, t_end, t_taper)
    
    nbl = 512

    # 3. Initialize native ReverseTimeMigration directly using the SEGY dataset directory
    rtm = ReverseTimeMigration(
        dataset_dir=shotpath,
        nbl=nbl,
        smooth_sigma=5.0,
        t_start=t_start,
        t_end=t_end,
        t_taper=t_taper,
        # space_order=4,
    )
    
    dx_spacing, dz_spacing = rtm.spacing

    # 5. Execute run method
    logger.info("Running Reverse Time Migration...")
    source_indices = list(range(n_sources)) if n_sources is not None else None
    migrated_image = rtm.run(save_wavefield=False, verbose=True, cli=cli, preview=True, source_indices=source_indices)

    # 6. Post-process and Plotting
    # Apply Laplace filter to the migrated image to remove low-frequency acquisition footprints.
    lap_image = laplace(migrated_image)
    
    # Exclude boundary (nbl) for plotting
    plotted_image = lap_image[nbl:-nbl, nbl:-nbl]
    output_npy = shotpath / f"{timestamp}_rtm_migrated.npy"
    try:
        np.save(output_npy, plotted_image)
        logger.info("Saved migrated image to: %s", output_npy)
    except OSError:
        fallback_npy = Path("/tmp") / f"{timestamp}_{shotpath.name}_rtm_migrated.npy"
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
    norm = TwoSlopeNorm(vcenter=0.0, vmin=0.1*vmin, vmax=0.1*vmax)
    logger.info("Plotting results...")
    log_array_stats(logger, "plotted_image", plotted_image)
    fig = plt.figure(figsize=(10, 5), dpi=150)
    im = plt.imshow(
        plotted_image.T,
        cmap=cmocean.cm.balance_r,
        extent=extent,
        aspect="auto",
        norm=norm
    )
    plt.colorbar(im, label="Amplitude", shrink=0.75)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title("RTM Migration Image")
    plt.gca().set_aspect("auto")
    plt.tight_layout()
    
    # plt.savefig(output_plot_path, bbox_inches="tight")
    # print(f"Plot saved to: {output_plot_path}")
    if cli:
        plt.savefig("img.png", dpi=150)
        plt.close(fig)
        subprocess.run(["chafa", "-w", "9", "img.png"], check=False)
        time.sleep(0.5)
        # subprocess.run("rm img.png".split())
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Dataset file to process")
    parser.add_argument("-c", "--cli", action="store_true")
    parser.add_argument("-n", "--n-sources", type=int, default=None, help="Number of sources to migrate")
    parser.add_argument("--t-start", type=float, default=None, help="Start time of migration window in ms (suppresses early-time direct arrivals)")
    parser.add_argument("--t-end", type=float, default=None, help="End time of migration window in ms")
    parser.add_argument("--t-taper", type=float, default=50.0, help="Cosine taper duration in ms at window edges (default: 50.0 ms)")
    args = parser.parse_args()
    main(args.cli, args.file, t_start=args.t_start, t_end=args.t_end, t_taper=args.t_taper, n_sources=args.n_sources)
