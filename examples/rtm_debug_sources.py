import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
import argparse
import subprocess
import time
from pathlib import Path
import cmocean
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import laplace

from shotgen.migration import ReverseTimeMigration
from shotgen.pylogger import get_logger, setup_file_logging
from shotgen.utils import setup_matplotlib_style

logger = get_logger("rtm_debug_sources")


def parse_source_indices(sources_arg, max_sources=None):
    """Parse comma/space/hyphen separated source indices into a sorted list of ints."""
    if not sources_arg:
        return None
    indices = set()
    # Normalize multiple arguments or comma/space-separated strings
    items = []
    for item in sources_arg:
        items.extend(item.replace(",", " ").split())

    for token in items:
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start, end = int(start_s), int(end_s)
            step = 1 if start <= end else -1
            indices.update(range(start, end + step, step))
        else:
            indices.add(int(token))

    sorted_indices = sorted(indices)
    if max_sources is not None:
        sorted_indices = [i for i in sorted_indices if 0 <= i < max_sources]
    return sorted_indices


def plot_and_save_image(image, extent, title, output_png=None, cli=False):
    """Plot migrated/laplace-filtered image slice and optionally display in CLI."""
    vmin = np.min(image)
    vmax = np.max(image)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=0.1 * vmin if vmin < 0 else -1e-6, vmax=0.1 * vmax if vmax > 0 else 1e-6)

    fig = plt.figure(figsize=(10, 5), dpi=150)
    im = plt.imshow(
        image.T,
        cmap=cmocean.cm.balance_r,
        extent=extent,
        aspect="auto",
        norm=norm,
    )
    plt.colorbar(im, label="Amplitude", shrink=0.75)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title(title)
    plt.gca().set_aspect("auto")
    plt.tight_layout()

    if output_png is not None:
        plt.savefig(output_png, bbox_inches="tight")
        logger.info("Saved plot to: %s", output_png)

    if cli:
        tmp_cli_img = Path("img.png")
        plt.savefig(tmp_cli_img, dpi=150)
        plt.close(fig)
        subprocess.run(["chafa", "-w", "9", str(tmp_cli_img)], check=False)
        time.sleep(0.5)
    else:
        plt.close(fig)


def main(file, cli=False, num_sources=None, sources=None, nbl=512):
    setup_matplotlib_style()

    shotpath = Path(__file__).resolve().parents[1] / file
    if not shotpath.exists():
        data_shotpath = Path(__file__).resolve().parents[1] / "data" / file
        if data_shotpath.exists():
            shotpath = data_shotpath
        else:
            raise FileNotFoundError(f"Could not locate dataset path: {file}")

    setup_file_logging(dataset_dir=shotpath, script_name="rtm_debug_sources", logger=logger)
    logger.info("Loading shot record from: %s", shotpath)

    # Output directory for individual source contributions
    out_dir = shotpath / "rtm_sources"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        out_dir = Path("/tmp") / f"{shotpath.name}_rtm_sources"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.warning("Dataset directory is read-only. Using fallback directory: %s", out_dir)

    rtm = ReverseTimeMigration(
        dataset_dir=shotpath,
        nbl=nbl,
        smooth_sigma=5.0,
    )

    dx_spacing, dz_spacing = rtm.spacing
    total_sources = rtm.n_sources
    logger.info("Dataset has %d total sources.", total_sources)

    # Determine which sources to run
    selected_indices = None
    if sources is not None:
        selected_indices = parse_source_indices(sources, max_sources=total_sources)
    elif num_sources is not None:
        selected_indices = list(range(min(num_sources, total_sources)))
    else:
        selected_indices = list(range(total_sources))

    logger.info("Selected %d source(s) to migrate: %s", len(selected_indices), selected_indices)

    # Run RTM with return_individual=True
    migrated_stack, individual_images = rtm.run(
        save_wavefield=False,
        verbose=True,
        cli=cli,
        source_indices=selected_indices,
        return_individual=True,
    )

    model = rtm.model
    extent = [
        model.origin[0] * dx_spacing,
        model.origin[0] + model.shape[0] * dx_spacing,
        model.origin[1] + model.shape[1] * dz_spacing,
        model.origin[1] * dz_spacing,
    ]

    # Process and save each individual source contribution
    for src_idx, src_img in individual_images.items():
        src_lap = laplace(src_img)[nbl:-nbl, nbl:-nbl]
        np.save(out_dir / f"source_{src_idx:03d}.npy", src_lap)
        logger.info(
            "Source %03d - Min: %.3e, Max: %.3e, Mean: %.3e, Std: %.3e",
            src_idx,
            np.amin(src_lap),
            np.amax(src_lap),
            np.mean(src_lap),
            np.std(src_lap),
        )
        plot_and_save_image(
            src_lap,
            extent,
            title=f"RTM Contribution - Source {src_idx}",
            output_png=out_dir / f"source_{src_idx:03d}.png",
            cli=cli,
        )

    # Process and save cumulative stacked image
    stack_lap = laplace(migrated_stack)[nbl:-nbl, nbl:-nbl]
    np.save(out_dir / "rtm_stack.npy", stack_lap)
    logger.info("Saved final stacked image to: %s", out_dir / "rtm_stack.npy")
    plot_and_save_image(
        stack_lap,
        extent,
        title=f"RTM Stacked Image ({len(selected_indices)} sources)",
        output_png=out_dir / "rtm_stack.png",
        cli=cli,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug RTM individual source contributions.")
    parser.add_argument("file", type=str, help="Dataset file/folder to process")
    parser.add_argument("-c", "--cli", action="store_true", help="Terminal chafa preview")
    parser.add_argument(
        "-n",
        "--num-sources",
        type=int,
        default=None,
        help="Migrate the first N sources (e.g., -n 3)",
    )
    parser.add_argument(
        "-s",
        "--sources",
        nargs="+",
        default=None,
        help="Explicit source indices or ranges (e.g., -s 0 2 4 or -s 0-3)",
    )
    parser.add_argument(
        "--nbl",
        type=int,
        default=512,
        help="Number of boundary layers (default: 512)",
    )

    args = parser.parse_args()
    main(
        file=args.file,
        cli=args.cli,
        num_sources=args.num_sources,
        sources=args.sources,
        nbl=args.nbl,
    )
