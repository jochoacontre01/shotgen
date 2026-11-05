#!/usr/bin/env python3
"""
Convert a 2D NumPy array (.npy) into a SEG-Y velocity model compliant with shotgen and shotgen-gpu.

Example:
    python scripts/npy_to_segy.py assets/fpb.npy --dx 50 --dz 50
    python scripts/npy_to_segy.py -i assets/fpb.npy -o assets/fpb.segy --dx 50 --dz 50 --order auto
"""

import argparse
import sys
from pathlib import Path

from shotgen.io import convert_npy_to_segy
from shotgen.pylogger import get_logger

logger = get_logger("shotgen.scripts.npy_to_segy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a 2D NumPy velocity model array (.npy) into a project-compliant SEG-Y file."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to the input .npy file (e.g., assets/fpb.npy).",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_flag",
        default=None,
        help="Alternative flag for input .npy file path.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path to output .segy file. Defaults to same name and directory as input with .segy extension.",
    )
    parser.add_argument(
        "--dx",
        type=float,
        required=True,
        help="Horizontal grid spacing in meters (dx > 0).",
    )
    parser.add_argument(
        "--dz",
        type=float,
        required=True,
        help="Vertical / depth grid spacing in meters (dz > 0).",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "zx", "xz"],
        default="auto",
        help=(
            "Orientation of the input array: "
            "'auto' (default: transposes (nz, nx) to (nx, nz) if nz < nx), "
            "'zx' (input is depth x lateral, transposes to (nx, nz)), "
            "'xz' (input is already lateral x depth)."
        ),
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress informational console output.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_file = args.input_flag or args.input
    if not input_file:
        logger.error("Input file must be provided either as positional argument or with -i / --input.")
        sys.exit(1)

    input_path = Path(input_file).resolve()
    if not input_path.is_file():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = input_path.with_suffix(".segy")

    verbose = not args.quiet
    try:
        model, dx, dz = convert_npy_to_segy(
            npy_path_or_array=input_path,
            output_filepath=output_path,
            dx=args.dx,
            dz=args.dz,
            order=args.order,
            verbose=verbose,
        )
        if verbose:
            logger.info("Conversion complete!")
            logger.info("Output SEG-Y: %s", output_path)
            logger.info("Grid dimensions: nx=%d, nz=%d, dx=%.1f m, dz=%.1f m", model.shape[0], model.shape[1], dx, dz)
            logger.info("Velocity range: min=%.1f m/s, max=%.1f m/s", float(model.min()), float(model.max()))
    except (OSError, ValueError, RuntimeError, TypeError) as exc:
        logger.error("Failed to convert %s to SEG-Y: %s", input_path, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
