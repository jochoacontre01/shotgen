import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import ReverseTimeMigration

def main():
    # 1. Locate and load the Sigsbee shot record file
    shotpath = Path(__file__).resolve().parents[1] / "data/commonshot-shot_750nx_275nz_48rec_6src_100hz_10goffset_45soffset_sigsbee.h5"
    print(f"Loading shot record from: {shotpath}")
    data = LoadShotRecord(shotpath)
    
    # 2. Extract acquisition parameters
    shots = data.shots
    velocity = data.velocity_model
    sources = data.sources
    receivers = data.receivers
    time = data.time
    f0 = data.f0

    print(f"Loaded shots shape: {shots.shape}")
    print(f"Loaded velocity model shape: {velocity.shape}")
    print(f"Loaded sources shape: {sources.shape}")
    print(f"Loaded receivers shape: {receivers.shape}")
    print(f"Time vector length: {len(time)}")

    # 3. Spatial spacing (dx = 1.0, dz = 1.0) and absorbing boundary size
    dx_spacing = 1.0
    dz_spacing = 1.0
    spacing = (dx_spacing, dz_spacing)
    nbl = 40
    
    # 4. Initialize native ReverseTimeMigration directly using the acquisition parameters
    rtm = ReverseTimeMigration(
        vp=velocity,
        sources=sources,
        receivers=receivers,
        shots=shots,
        time=time,
        spacing=spacing,
        nbl=nbl,
        f0=f0,
        smooth_sigma=5.0,
        space_order=4,
    )

    # 5. Execute run method
    print("Running Reverse Time Migration...")
    migrated_image = rtm.run(save_wavefield=False)

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
    plt.savefig(output_plot_path, bbox_inches="tight")
    print(f"Plot saved to: {output_plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
