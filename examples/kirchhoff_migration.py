from pathlib import Path
import re
import os
import time
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import cmocean
from scipy.ndimage import laplace

from shotgen.migration import KirchhoffMigration
from shotgen.pylogger import get_logger, setup_file_logging

logger = get_logger("kirchhoff_migration")


def main():
    parser = argparse.ArgumentParser(description="Kirchhoff Pre-Stack Depth Migration (PSDM)")
    parser.add_argument("file", type=str, help="Dataset file/folder to process")
    parser.add_argument("-c", "--cli", action="store_true", help="setup runtime for non-gui interface")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel worker jobs (default: -1, all CPUs)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size of shots to process in parallel (default: 8)")
    args = parser.parse_args()

    if args.file is not None:
        shotpath = Path(args.file)
        if not shotpath.exists():
            shotpath = Path(__file__).resolve().parents[1] / args.file
        if not shotpath.exists():
            data_shotpath = Path(__file__).resolve().parents[1] / "data" / args.file
            if data_shotpath.exists():
                shotpath = data_shotpath
        if not shotpath.exists():
            raise FileNotFoundError(f"Dataset path not found: {args.file} (checked root and data/)")
    else:
        raise ValueError("No data file was passed to the arguments, pass it with 'kirchhoff_migration.py path/to/file'")

    setup_file_logging(dataset_dir=shotpath, script_name="kirchhoff_migration", logger=logger)

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

    logger.info(
        "Running Kirchhoff PSDM on dataset: %s (dx=%.2fm, dz=%.2fm, jobs=%s, batch_size=%d)",
        shotpath,
        dx_spacing,
        dz_spacing,
        args.jobs,
        args.batch_size,
    )

    migrator = KirchhoffMigration(
        dataset_dir=str(shotpath),
        spacing=(dx_spacing, dz_spacing),
        smooth_sigma=15.0,
        cli=args.cli,
        n_jobs=args.jobs,
        batch_size=args.batch_size,
    )

    image = migrator.run()
    output_npy = shotpath / "kirchhoff_migrated.npy"
    try:
        np.save(output_npy, image)
        logger.info("Saved migrated image to: %s", output_npy)
    except OSError:
        fallback_npy = Path("/tmp") / f"{name}_kirchhoff_migrated.npy"
        np.save(fallback_npy, image)
        logger.warning(
            "Dataset directory is read-only; saved migrated image to fallback path: %s",
            fallback_npy,
        )

    model = migrator.model
    plt.figure(figsize=(10, 5))
    extent = [
        model.origin[0], 
        model.origin[0] + model.shape[0] * dx_spacing,
        model.origin[1] + model.shape[1] * dz_spacing, 
        model.origin[1]
    ]
    vmin = np.min(laplace(image))
    vmax = np.max(laplace(image))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = plt.imshow(laplace(image.T), cmap=cmocean.cm.balance_r, extent=extent, norm=norm)
    plt.colorbar(im, label="Amplitude", shrink=0.75)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title("Kirchhoff Migration Image")
    plt.gca().set_aspect("auto")
    plt.tight_layout()
    if args.cli:
        temp_img = "/tmp/kirchhoff_preview.png"
        plt.savefig(temp_img)
        plt.close()
        subprocess.run(f"chafa -w 9 {temp_img}".split())
        time.sleep(0.5)
        try:
            os.remove(temp_img)
        except Exception:
            pass
    else:    
        plt.show()


if __name__ == "__main__":
    main()
