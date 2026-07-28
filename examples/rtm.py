import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace
from matplotlib.colors import TwoSlopeNorm

from shotgen_gpu.config import configure_devito_device
from shotgen_gpu.engine.rtm import ReverseTimeMigration


def main():
    parser = argparse.ArgumentParser(description="rtm.py: Devito / OpenACC Reverse Time Migration example")
    parser.add_argument("--dataset_dir", "-d", type=str, default="data/example_simulation_output", help="Path to generated simulation dataset directory")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cpu', 'cuda')")
    parser.add_argument("--no_plot", action="store_true", help="Skip display of plot window")
    parser.add_argument("--cli", action="store_true", help="Save plot preview in non-GUI terminal mode")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory '{dataset_path}' not found. Please run shotrecord.py first.", file=sys.stderr)
        sys.exit(1)

    configured_device = configure_devito_device(args.device)
    print(f"[rtm.py] Running RTM on device: {configured_device.upper()} | Dataset: {dataset_path}")

    rtm = ReverseTimeMigration(
        dataset_dir=str(dataset_path),
        nbl=40,
        space_order=4,
        device=configured_device,
    )

    image = rtm.run(save_wavefield=False)

    out_file = dataset_path / "rtm_migrated.npy"
    np.save(out_file, image)
    print(f"[rtm.py] Reverse Time Migration complete! Image saved to: {out_file}")

    if not args.no_plot:
        plt.figure(figsize=(10, 5))
        extent = [
            rtm.origin[0],
            rtm.origin[0] + rtm.vp.shape[0] * rtm.spacing[0],
            rtm.origin[1] + rtm.vp.shape[1] * rtm.spacing[1],
            rtm.origin[1]
        ]
        lap_img = laplace(image)
        nbl = rtm.nbl
        crop_img = lap_img[nbl:-nbl, nbl:-nbl] if (image.shape[0] > 2*nbl and image.shape[1] > 2*nbl) else lap_img

        vmin = np.quantile(crop_img, 0.05)
        vmax = np.quantile(crop_img, 0.95)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if vmin < 0 < vmax else None

        im = plt.imshow(crop_img.T, cmap="gray", extent=extent, norm=norm, aspect="auto")
        plt.colorbar(im, label="Amplitude")
        plt.xlabel("Distance (m)")
        plt.ylabel("Depth (m)")
        plt.title(f"Reverse Time Migration ({configured_device.upper()})")
        plt.tight_layout()

        if args.cli:
            plt.savefig("rtm_preview.png", dpi=100)
            print("[rtm.py] Saved preview image to rtm_preview.png")
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    main()
