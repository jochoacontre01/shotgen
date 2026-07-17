import os
import segyio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cmocean

class SegyIO:
    """
    A class to handle creating, writing, and reading SEGY files for seismic shot records.
    """

    @staticmethod
    def write(filepath, shot_record):
        """
        Write a ShotRecord object to a SEGY file.
        
        Parameters
        ----------
        filepath : str
            The path where the SEGY file will be saved.
        shot_record : ShotRecord
            The shot record instance containing data and geometry.
        """
        if shot_record.shot_run is None:
            raise ValueError("ShotRecord does not contain simulated data. Please run it first.")

        n_sources, n_receivers, n_time = shot_record.shot_run.shape
        total_traces = n_sources * n_receivers
        
        # dt is in seconds, we need microseconds for the SEGY binary header
        dt_s = shot_record.time_vector[1] - shot_record.time_vector[0]
        dt_us = int(np.round(dt_s * 1e6))
        
        # Setup SEGY specification
        spec = segyio.spec()
        spec.sorting = 1 # Shot gather
        spec.format = 5 # IEEE floating point
        spec.samples = shot_record.time_vector * 1000 # Time vector in milliseconds
        spec.tracecount = total_traces
        
        # Calculate unique CMP X positions for CDP binning
        cmp_x_all = []
        for src_idx in range(n_sources):
            sx = shot_record.sources[src_idx, 0]
            for rec_idx in range(n_receivers):
                if shot_record.recs.ndim == 3:
                    rx = shot_record.recs[src_idx, rec_idx, 0]
                else:
                    rx = shot_record.recs[rec_idx, 0]
                cmp_x = (sx + rx) / 2
                cmp_x_all.append(cmp_x)
                
        # Use millimetric precision rounding to avoid floating point anomalies when unique sorting
        unique_cmp_x = np.unique(np.round(cmp_x_all, 3))
        cmp_x_to_cdp = {val: idx + 1 for idx, val in enumerate(unique_cmp_x)}
        
        with segyio.create(filepath, spec) as f:
            f.bin[segyio.BinField.Interval] = dt_us
            f.bin[segyio.BinField.Samples] = n_time
            
            trace_idx = 0
            
            for src_idx in range(n_sources):
                sx = shot_record.sources[src_idx, 0]
                sy = shot_record.sources[src_idx, 1]
                
                for rec_idx in range(n_receivers):
                    trace_data = shot_record.shot_run[src_idx, rec_idx, :]
                    
                    if shot_record.recs.ndim == 3:
                        rx = shot_record.recs[src_idx, rec_idx, 0]
                        ry = shot_record.recs[src_idx, rec_idx, 1]
                    else:
                        rx = shot_record.recs[rec_idx, 0]
                        ry = shot_record.recs[rec_idx, 1]
                        
                    cmp_x = (sx + rx) / 2
                    cmp_y = (sy + ry) / 2
                    cdp_num = cmp_x_to_cdp[np.round(cmp_x, 3)]
                    
                    # Write trace data
                    f.trace[trace_idx] = trace_data.astype(np.float32)
                    
                    # Write headers
                    f.header[trace_idx] = {
                        segyio.TraceField.TRACE_SEQUENCE_LINE: trace_idx + 1,
                        segyio.TraceField.TRACE_SEQUENCE_FILE: trace_idx + 1,
                        segyio.TraceField.FieldRecord: src_idx + 1,
                        segyio.TraceField.TraceNumber: rec_idx + 1,
                        segyio.TraceField.SourceX: int(np.round(sx * 1000)),
                        segyio.TraceField.SourceY: int(np.round(sy * 1000)),
                        segyio.TraceField.GroupX: int(np.round(rx * 1000)),
                        segyio.TraceField.GroupY: int(np.round(ry * 1000)),
                        segyio.TraceField.CDP: cdp_num,
                        segyio.TraceField.CDP_X: int(np.round(cmp_x * 1000)),
                        segyio.TraceField.CDP_Y: int(np.round(cmp_y * 1000)),
                        segyio.TraceField.SourceGroupScalar: -1000,
                        segyio.TraceField.ElevationScalar: -1000,
                        segyio.TraceField.offset: int(np.round(np.sqrt((sx - rx)**2 + (sy - ry)**2) * 1000))
                    }
                    
                    trace_idx += 1

    @staticmethod
    def read(filepath):
        """
        Read a SEGY file and return its geometry and traces.
        
        Parameters
        ----------
        filepath : str
            The path to the SEGY file.
            
        Returns
        -------
        dict
            A dictionary containing data, sources, receivers, cdps, dt, and time vector.
        """
        with segyio.open(filepath, "r", strict=False) as f:
            n_traces = f.tracecount
            
            data = segyio.tools.collect(f.trace)
            
            sources = np.zeros((n_traces, 2))
            receivers = np.zeros((n_traces, 2))
            cdps = np.zeros(n_traces, dtype=int)
            offsets = np.zeros(n_traces)
            
            for i in range(n_traces):
                header = f.header[i]
                scalar = header[segyio.TraceField.SourceGroupScalar]
                if scalar < 0:
                    mult = 1.0 / abs(scalar)
                elif scalar > 0:
                    mult = scalar
                else:
                    mult = 1.0
                    
                sources[i, 0] = header[segyio.TraceField.SourceX] * mult
                sources[i, 1] = header[segyio.TraceField.SourceY] * mult
                receivers[i, 0] = header[segyio.TraceField.GroupX] * mult
                receivers[i, 1] = header[segyio.TraceField.GroupY] * mult
                cdps[i] = header[segyio.TraceField.CDP]
                
                # Check for offset header, though we calculate it if missing
                offset_scalar_value = header[segyio.TraceField.offset]
                offsets[i] = offset_scalar_value * mult if offset_scalar_value != 0 else np.sqrt(
                    (sources[i, 0] - receivers[i, 0])**2 + (sources[i, 1] - receivers[i, 1])**2
                )
                
            dt = f.bin[segyio.BinField.Interval] / 1e6 # microseconds to seconds
            time_vector = f.samples / 1000.0 # milliseconds to seconds
            
        return {
            "data": data,
            "sources": sources,
            "receivers": receivers,
            "cdp": cdps,
            "offset": offsets,
            "dt": dt,
            "time": time_vector
        }

    @staticmethod
    def write_model(filepath, model, dx, dz):
        """
        Write a 2D velocity model array to a SEGY file.
        
        Parameters
        ----------
        filepath : str
            The path where the SEGY file will be saved.
        model : ndarray
            The 2D model array (nx, nz).
        dx : float
            Horizontal spacing.
        dz : float
            Vertical spacing.
        """
        nx, nz = model.shape
        spec = segyio.spec()
        spec.sorting = 1
        spec.format = 5 # IEEE floating point
        spec.samples = np.arange(nz) * dz # Pseudo-time axis
        spec.tracecount = nx
        
        with segyio.create(filepath, spec) as f:
            f.bin[segyio.BinField.Interval] = int(dz * 1000)
            f.bin[segyio.BinField.Samples] = nz
            
            for i in range(nx):
                f.trace[i] = model[i, :].astype(np.float32)
                f.header[i] = {
                    segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
                    segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
                    segyio.TraceField.CDP: i + 1,
                    segyio.TraceField.CDP_X: int(np.round(i * dx * 1000)),
                    segyio.TraceField.SourceGroupScalar: -1000,
                }

    @staticmethod
    def read_model(filepath):
        """
        Read a 2D velocity model array and dimensions from a SEGY file.

        Parameters
        ----------
        filepath : str
            The path to the SEGY file.

        Returns
        -------
        tuple
            (data, dx, dz) where data is the 2D model array, and dx/dz are spacings.
        """
        with segyio.open(filepath, "r", strict=False) as f:
            data = segyio.tools.collect(f.trace)
            dz = f.bin[segyio.BinField.Interval] / 1000.0
            nx = len(f.trace)
            if nx > 1:
                dx = (f.header[1][segyio.TraceField.CDP_X] - f.header[0][segyio.TraceField.CDP_X]) / 1000.0
            else:
                dx = 1.0
            
            # fallback to 1.0 if not written correctly (e.g. older files)
            if dx <= 0.0:
                dx = 1.0
            if dz <= 0.0:
                dz = 1.0

        return data, dx, dz


def generate_video(shot_record, output_path=None, source_idx=0):
    """
    Generate a video of the wavefield propagation.
    
    Parameters
    ----------
    shot_record : ShotRecord
        The shot record instance containing simulated data and geometry.
    output_path : str, optional
        The path where the MP4 video will be saved. Default is './videos/wavefield.mp4'.
    source_idx : int, optional
        The index of the source to animate. Default is 0.
    """
    if shot_record.us is None or len(shot_record.us) == 0:
        raise ValueError("ShotRecord does not contain saved wavefield data (self.us). Ensure you run with save_wavefield=True.")
    
    if shot_record.v0 is None:
        raise ValueError("ShotRecord does not contain background velocity model (self.v0).")
        
    # Check if FFMpegWriter is available
    if not animation.FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg is not installed or not in the system PATH. "
            "Please install FFmpeg to export videos. "
            "On Ubuntu/Debian: sudo apt-get install ffmpeg\n"
            "On Fedora/CentOS: sudo dnf install ffmpeg\n"
            "On macOS (with Homebrew): brew install ffmpeg"
        )
    
    # 1. Output Path Setup
    if output_path is None:
        output_path = "./videos/wavefield.mp4"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # 2. Extract and Slice Wavefield
    # self.us shape is (n_sources, n_saved_frames, nx_padded, nz_padded)
    us_src = shot_record.us[source_idx]
    nbl = getattr(shot_record, "n_damping", 0)
    
    if nbl > 0:
        # Slice out the padding boundary layers
        us_src_clean = us_src[:, nbl:-nbl, nbl:-nbl]
    else:
        us_src_clean = us_src
        
    total_saved_frames = us_src_clean.shape[0]
    
    # 3. Framerate Calculation
    # fps = (Total Saved Frames) / (self.tn / 1000)
    if shot_record.tn is None or shot_record.tn == 0:
        raise ValueError("ShotRecord total simulation time (self.tn) is not set or is zero.")
    tn_s = shot_record.tn / 1000.0
    fps = total_saved_frames / tn_s
    
    # 4. Setup Plot & Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extent details: (x_min, x_max, z_max, z_min)
    extent = (
        shot_record.origin[0], 
        shot_record.origin[0] + shot_record.nx * shot_record.dx, 
        shot_record.origin[-1] + shot_record.nz * shot_record.dz, 
        shot_record.origin[-1]
    )
    
    # Background: Access ShotRecord.v0 (alpha=0.3, static colormap).
    im_bg = ax.imshow(shot_record.v0.T, extent=extent, cmap='gray', alpha=0.3, aspect='auto')
    
    # Foreground: Access ShotRecord.us (alpha=1.0, cmocean.cm.balance).
    # First frame init
    frame_0 = us_src_clean[0]
    vmin = min(frame_0.min(), -1e-10)
    vmax = max(frame_0.max(), 1e-10)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
    im_fg = ax.imshow(frame_0.T, extent=extent, cmap=cmocean.cm.balance, norm=norm, alpha=1.0, aspect='auto')
    
    ax.set_title(f"Wavefield Propagation (Source {source_idx + 1})")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    
    # To update the existing plot objects using set_data() / set_array()
    def update(frame_idx):
        frame = us_src_clean[frame_idx]
        vmin_f = min(frame.min(), -1e-10)
        vmax_f = max(frame.max(), 1e-10)
        norm_f = mcolors.TwoSlopeNorm(vmin=vmin_f, vcenter=0.0, vmax=vmax_f)
        im_fg.set_norm(norm_f)
        im_fg.set_data(frame.T)
        return [im_fg]
        
    # FuncAnimation
    anim = animation.FuncAnimation(fig, update, frames=total_saved_frames, blit=False)
    
    # Writer
    writer = animation.FFMpegWriter(fps=fps, codec='libx264')
    
    # Save the animation
    try:
        anim.save(output_path, writer=writer)
        print(f"Video successfully saved to {output_path}")
    finally:
        # Explicitly clear/close figure objects per frame / at the end of the writer process
        plt.close(fig)
        fig.clear()
        import gc
        gc.collect()
