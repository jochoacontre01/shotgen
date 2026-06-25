import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import ReverseTimeMigration

def main():
    # 1. Locate and load the Sigsbee shot record file
    shotpath = Path(__file__).resolve().parents[1] / "data/commonshot-shot_750nx_50nz_32rec_6src_250hz_10goffset_45soffset_sigsbee_dataset"
    print(f"Loading shot record from: {shotpath}")
    
    # 2. Spatial spacing (dx = 1.0, dz = 1.0) and absorbing boundary size
    dx_spacing = 1.0
    dz_spacing = 1.0
    spacing = (dx_spacing, dz_spacing)
    nbl = 40
    
    # 3. Initialize native ReverseTimeMigration directly using the SEGY dataset directory
    rtm = ReverseTimeMigration(
        dataset_dir=shotpath,
        spacing=spacing,
        nbl=nbl,
        smooth_sigma=5.0,
        space_order=4,
    )

    # 5. Execute run method
    print("Running Reverse Time Migration...")
    migrated_image = rtm.run(save_wavefield=False)
    print(migrated_image)

    # 6. Post-process and Plot
    # Apply Laplace filter to the migrated image to remove low-frequency acquisition footprints.
    lap_image = laplace(migrated_image)
    
    # Exclude boundary (nbl) for plotting
    plotted_image = lap_image[nbl:-nbl, nbl:-nbl]
    
    # Compute display extent in meters
    model = rtm.model
    extent = [
        model.origin[0] + nbl * dx_spacing, 
        model.origin[0] + (model.shape[0] - nbl) * dx_spacing,
        model.origin[1] + (model.shape[1] - nbl) * dz_spacing, 
        model.origin[1] + nbl * dz_spacing
    ]

    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.imshow(
        plotted_image.T,
        cmap="gray",
        extent=extent,
        aspect="auto"
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    plt.title("RTM Migrated Image (Sigsbee)")
    
    output_plot_path = Path(__file__).resolve().parent / "rtm_migrated_sigsbee.png"
    # plt.savefig(output_plot_path, bbox_inches="tight")
    print(f"Plot saved to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
