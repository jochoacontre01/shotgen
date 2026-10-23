import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace
from matplotlib.colors import TwoSlopeNorm

from shotgen.migration.kirchhoff import KirchhoffMigration
from shotgen.pylogger import get_logger, setup_file_logging
from shotgen.utils import setup_matplotlib_style

logger = get_logger("kirchhoff")


def main():
    setup_matplotlib_style()
    parser = argparse.ArgumentParser(description="kirchhoff.py: Kirchhoff pre-stack depth migration example")
    parser.add_argument("--dataset_dir", "-d", type=str, default="data/example_simulation_output", help="Path to generated simulation dataset directory")
    parser.add_argument("--no_plot", action="store_true", help="Skip display of plot window")
    parser.add_argument("--cli", action="store_true", help="Save plot preview in non-GUI terminal mode")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_path}' not found. Please run shotrecord.py first.", file=sys.stderr)
        sys.exit(1)

    log_path = setup_file_logging(dataset_dir=dataset_path, script_name="kirchhoff", logger=logger)
    timestamp = log_path.name[:15]

    logger.info("Loading dataset from: %s", dataset_path)
    migrator = KirchhoffMigration(dataset_dir=str(dataset_path))

    logger.info("Running Kirchhoff PSDM on grid shape: %s...", migrator.vp.shape)
    image = migrator.run()

    out_file = dataset_path / f"{timestamp}_kirchhoff_migrated.npy"
    np.save(out_file, image)
    logger.info("Kirchhoff migration complete! Image saved to: %s", out_file)

    if not args.no_plot:
        plt.figure(figsize=(10, 5), dpi=150)
        extent = [
            migrator.origin[0],
            migrator.origin[0] + migrator.vp.shape[0] * migrator.spacing[0],
            migrator.origin[1] + migrator.vp.shape[1] * migrator.spacing[1],
            migrator.origin[1]
        ]
        lap_img = laplace(image)
        vmin = np.quantile(lap_img, 0.05)
        vmax = np.quantile(lap_img, 0.95)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if vmin < 0 < vmax else None

        im = plt.imshow(lap_img.T, cmap="grey", extent=extent, norm=norm, aspect="auto")
        plt.colorbar(im, label="Amplitude")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.title("Kirchhoff PSDM Migrated Image")
        plt.tight_layout()

        if args.cli:
            plt.savefig("kirchhoff_preview.png", dpi=150)
            print("[kirchhoff.py] Saved preview image to kirchhoff_preview.png")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
