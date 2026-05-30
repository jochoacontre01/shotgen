from shotgen.sampleshot import LoadShot
import matplotlib.pyplot as plt
from matplotlib.path import Path
import numpy as np
import h5py
import os

def filter_2d(data, x, y):
    """
    Computes the 2D FFT of the input array, displays the magnitude spectrum
    interactively for the user to draw a polygon mask, zeros out the frequency
    components within that mask, and returns the inverse 2D FFT filtered array.

    Parameters:
    -----------
    data : numpy.ndarray
        2D input array (image or signal data).
    x : numpy.ndarray
        1D array representing the x-coordinates (spatial domain).
    y : numpy.ndarray
        1D array representing the y-coordinates (spatial domain).

    Returns:
    --------
    filtered_data : numpy.ndarray
        The reconstructed 2D array after filtering.
    """
    
    # 1. Compute 2D FFT and shift zero frequency to center
    F = np.fft.fft2(data)
    F_shifted = np.fft.fftshift(F)
    
    # Compute magnitude spectrum for visualization (log scale)
    # We add a small constant to avoid log(0)
    mag_spectrum = np.log(1 + np.abs(F_shifted))

    # 2. Determine Frequency Coordinates
    # Assuming uniform sampling, we calculate the frequency axes
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    rows, cols = data.shape
    
    freq_x = np.fft.fftshift(np.fft.fftfreq(cols, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(rows, d=dy))
    
    # Create a meshgrid of these frequency coordinates for masking later
    FX, FY = np.meshgrid(freq_x, freq_y)

    # 3. Interactive Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Draw a polygon to mask frequencies.\nSingle-click to add points. Double-click to close and apply.")
    ax.set_xlabel("Frequency X")
    ax.set_ylabel("Frequency Y")
    
    # Plot using extent so axis labels match physical frequency units
    extent = [freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]
    im = ax.imshow(mag_spectrum, extent=extent, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Log Magnitude')

    # State container to store vertices from the interaction
    interaction_state = {
        'vertices': [],
        'finished': False
    }

    # Event handler function
    def on_click(event):
        # Ignore clicks outside axes or if already finished
        if event.inaxes != ax or interaction_state['finished']:
            return

        # Double click to finish
        if event.dblclick:
            interaction_state['finished'] = True
            plt.close(fig)
            return

        # Handle mouse buttons
        if event.button == 1:  # Left click: Add point
            interaction_state['vertices'].append((event.xdata, event.ydata))
        elif event.button == 3:  # Right click: Remove last point
            if interaction_state['vertices']:
                interaction_state['vertices'].pop()
        else:
            return  # Ignore other buttons

        # Visual feedback: Redraw the whole polygon
        # We remove existing lines (previous polygon segments) to prevent clutter
        # while preserving the background image (which is in ax.images)
        for line in list(ax.lines):
            line.remove()
        
        # Update title to reflect controls
        ax.set_title("Left: Add Point | Right: Undo Last | Double-Click: Finish")

        if interaction_state['vertices']:
            # Unzip list of tuples to x and y lists
            x_pts = [v[0] for v in interaction_state['vertices']]
            y_pts = [v[1] for v in interaction_state['vertices']]
            # Plot all vertices as a connected line with markers
            ax.plot(x_pts, y_pts, 'r-+', markersize=10)
        
        fig.canvas.draw()

    # Connect the event handler
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Show window and BLOCK execution until window is closed
    print("Please draw the polygon on the plot window...")
    plt.show()

    # 4. Create Mask and Apply Filter
    verts = interaction_state['vertices']
    
    # If user drew a polygon (needs at least 3 points, but we'll check for >0)
    if len(verts) > 2:
        path = Path(verts)
        
        # Flatten the grid to check points, then reshape back to image shape
        # FX, FY are the frequency coordinates of every pixel in the FFT image
        points = np.vstack((FX.flatten(), FY.flatten())).T
        
        # contains_points returns a boolean mask
        mask_flat = path.contains_points(points)
        mask = mask_flat.reshape(FX.shape)
        
        # Apply mask: Delete (zero out) the masked part
        # If the goal is to REMOVE the selected area (e.g., removing a spike):
        F_shifted[mask] = 0
        
        print(f"Filter applied. Masked {np.sum(mask)} frequency components.")
    else:
        print("No valid polygon drawn. Returning original data.")

    # 5. Inverse FFT
    # Shift zero freq back to corner
    F_inverse_shifted = np.fft.ifftshift(F_shifted)
    # Inverse FFT
    img_filtered = np.fft.ifft2(F_inverse_shifted)
    
    # Return real part (assuming input was real)
    return np.real(img_filtered)

# # ? Generate shot record
# sim = ShotRecord(
#     nx=200,
#     nz=30,
#     n_sources=1,
#     n_receivers=100,
#     f0=75,
#     origin=(100,0)
# )

# sim.layer_model()
# sim.show_model()

# sim.run(ms=0.15E3)
# sim.save_shot("shot1.h5", overwrite=True)


# ? Load shot record
shot = LoadShot(r"/media/jochoa/DATA/Documentos/MUN Earth Sciences Masters Degree/100 - Thesis research project/Codes/osteo-seismic_ultrasound/src/shotgen/data/shot1.h5")

shots = shot.shots
nshots = shot.nshots
time = shot.time
receivers = shot.receivers
sources = shot.sources

# ? Plot single shot
shot.plot()

# ? Plot and perform 2D FFT filter of shot
# shots_filtered = np.zeros_like(shots, dtype=float)
# for i, ishot in enumerate(shots):
#     shot_filtered = filter_2d(ishot, receivers[:,0], time)

#     plt.figure()
#     plt.subplot(121)
#     plt.imshow(ishot.T, extent=(np.min(receivers[:,0]), np.max(receivers[:,0]), np.min(time), np.max(time)), aspect="auto", cmap="gray")
#     plt.subplot(122)
#     plt.imshow(shot_filtered.T, extent=(np.min(receivers[:,0]), np.max(receivers[:,0]), np.min(time), np.max(time)), aspect="auto", cmap="gray")
#     plt.xlabel("receiver")
#     plt.ylabel("time (ms)")
#     plt.show()
    
#     shots_filtered[i] = shot_filtered

# with h5py.File(os.path.join(os.path.dirname(os.path.dirname(__file__)),"data/shot1_f.h5"), mode="w") as f:
#     f.create_dataset("shots", data=shots_filtered)
#     f.create_dataset("receivers", data=receivers, shape=receivers.shape)
#     f.create_dataset("sources", data=sources, shape=sources.shape)
#     f.create_dataset("velocity_model", data=shot.velocity_model, shape=shot.velocity_model.shape)
#     f.create_dataset("smooth_velocity", data=shot.smooth_velocity, shape=shot.smooth_velocity.shape)
#     f.create_dataset("time", data=shot.time, shape=shot.time.shape)
#     print("Saved file")

# ? Plot single trace
# rx = receivers[0]
# shot1 = shot.shots[1]
# plt.plot(shot1[int(len(rx)/2)], time*1E-3)
# plt.gca().invert_yaxis()
# plt.show()
