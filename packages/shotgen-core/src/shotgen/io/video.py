import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cmocean


def generate_video(shot_record, output_path=None, source_idx=0):
    """
    Generate a video of the wavefield propagation.
    """
    if shot_record.us is None or len(shot_record.us) == 0:
        raise ValueError("ShotRecord does not contain saved wavefield data (self.us). Ensure you run with save_wavefield=True.")

    if shot_record.v0 is None:
        raise ValueError("ShotRecord does not contain background velocity model (self.v0).")

    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg is not installed or not in the system PATH. "
            "Please install FFmpeg to export videos."
        )

    if output_path is None:
        output_path = "./videos/wavefield.mp4"

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    us_src = shot_record.us[source_idx]
    nbl = getattr(shot_record, "n_damping", 0)

    if nbl > 0:
        us_src_clean = us_src[:, nbl:-nbl, nbl:-nbl]
    else:
        us_src_clean = us_src

    total_saved_frames = us_src_clean.shape[0]

    if shot_record.tn is None or shot_record.tn == 0:
        raise ValueError("ShotRecord total simulation time (self.tn) is not set or is zero.")
    tn_s = shot_record.tn / 1000.0
    fps = total_saved_frames / tn_s

    fig, ax = plt.subplots(figsize=(10, 6))

    extent = (
        shot_record.origin[0], 
        shot_record.origin[0] + shot_record.nx * shot_record.dx, 
        shot_record.origin[-1] + shot_record.nz * shot_record.dz, 
        shot_record.origin[-1]
    )

    im_bg = ax.imshow(shot_record.v0.T, extent=extent, cmap='gray', alpha=0.3, aspect='auto')

    frame_0 = us_src_clean[0]
    vmin = min(frame_0.min(), -1e-10)
    vmax = max(frame_0.max(), 1e-10)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    im_fg = ax.imshow(frame_0.T, extent=extent, cmap=cmocean.cm.balance, norm=norm, alpha=1.0, aspect='auto')

    ax.set_title(f"Wavefield Propagation (Source {source_idx + 1})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")

    def update(frame_idx):
        frame = us_src_clean[frame_idx]
        vmin_f = min(frame.min(), -1e-10)
        vmax_f = max(frame.max(), 1e-10)
        norm_f = mcolors.TwoSlopeNorm(vmin=vmin_f, vcenter=0.0, vmax=vmax_f)
        im_fg.set_norm(norm_f)
        im_fg.set_data(frame.T)
        return [im_fg]

    anim = animation.FuncAnimation(fig, update, frames=total_saved_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec='libx264')

    try:
        anim.save(output_path, writer=writer)
        print(f"Video successfully saved to {output_path}")
    finally:
        plt.close(fig)
        fig.clear()
        import gc
        gc.collect()
