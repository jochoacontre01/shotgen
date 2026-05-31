import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import laplace, gaussian_filter
from devito import Function, TimeFunction, Eq, solve, Operator
from examples.seismic import PointSource
from shotgen.sampleshot import LoadShotRecord
from shotgen.migration import ReverseTimeMigration
from tqdm import tqdm

class CustomReverseTimeMigration(ReverseTimeMigration):
    """
    Subclass of ReverseTimeMigration to handle dynamic receiver coordinates
    varying per shot.
    """
    def __init__(
        self,
        vp: np.ndarray,
        n_sources: int,
        n_receivers: int,
        origin: tuple,
        spacing: tuple,
        nbl: int,
        t0: float,
        tn: float,
        f0: float,
        smooth_sigma: float,
        dt: float,
        space_order: int = 4,
        time_order: int = 2,
    ):
        self.vp = vp / 1000
        self.v0 = gaussian_filter(self.vp, sigma=smooth_sigma)
        self.n_sources = n_sources
        self.n_receivers = n_receivers
        self.origin = origin
        self.spacing = spacing
        self.nbl = nbl
        self.space_order = space_order
        self.time_order = time_order
        self.t0 = t0
        self.tn = tn
        self.f0 = f0 / 1000
        self.dtype = np.float32

        self._create_model()
        
        # Placeholder coordinates; will be updated dynamically per shot.
        self.sources = np.zeros((self.n_sources, 2), dtype=np.float32)
        self.receivers = np.zeros((self.n_receivers, 2), dtype=np.float32)
        
        self._create_geometry()
        self.geometry.resample(dt)
        self._setup_solver()

    def _imaging_operator(self, model, image, save_wavefield=False):
        if save_wavefield:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        else:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order)

        u = TimeFunction(name="u", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        
        eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T
        
        stencil = Eq(v.backward, solve(eqn, v.backward))
        
        dt = model.grid.stepping_dim.spacing
        
        # Store residual PointSource as an attribute so we can update its coordinates per shot
        self.residual_source = PointSource(name="residual", grid=model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        res_term = self.residual_source.inject(field=v.backward, expr=self.residual_source * dt**2 / model.m)
        
        image_update = Eq(image, image - u * v)
    
        return Operator([stencil] + res_term + [image_update], subs=model.spacing_map)

    def migrate_from_data(self, shot_records, sources, receivers, save_wavefield=False, save_each=5):
        us = []
        vs = []
        image = Function(name="image", grid=self.model.grid)
        operator = self._imaging_operator(self.model, image, save_wavefield=save_wavefield)
        nshots = shot_records.shape[0]
        
        for i in tqdm(range(nshots), desc="source", total=nshots):
            # Dynamic update of source coordinates
            self.geometry.src_positions[0, :] = sources[i, :]
            
            # Dynamic update of receiver coordinates for this shot
            current_recs = receivers[i] if receivers.ndim == 3 else receivers
            self.geometry.rec.coordinates.data[:, :] = current_recs
            self.residual_source.coordinates.data[:, :] = current_recs
            
            # Run forward solver with smoothed velocity and save wavefield using geom dt
            _, u0, _ = self.solver.forward(vp=self.model0.vp, save=True, dt=self.geometry.dt)
            
            # Instantiate backward wavefield
            if save_wavefield:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
            else:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order)
            
            # Execute imaging operator
            operator(u=u0, v=v, vp=self.model0.vp, dt=self.geometry.dt, residual=shot_records[i])
            
            if save_wavefield:
                us.append(u0.data.copy()[::save_each])
                vs.append(v.data.copy()[::save_each])
                
        self.us = np.array(us)
        self.vs = np.array(vs)
        self.image = image.data
        return self.image

def main():
    # 1. Locate and load the Sigsbee shot record file
    shotpath = Path(__file__).resolve().parents[1] / "data/commonshot-shot_750nx_275nz_48rec_6src_100hz_10goffset_45soffset_sigsbee.h5"
    print(f"Loading shot record from: {shotpath}")
    data = LoadShotRecord(shotpath)
    
    # 2. Extract necessary arrays
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

    # 3. Spatial spacing (dx = 1.0, dz = 1.0)
    dx_spacing = 1.0
    dz_spacing = 1.0
    spacing = (dx_spacing, dz_spacing)
    nbl = 40
    
    # Devito expects physical coordinates (which are grid indices * spacing).
    # Since spacing is (1.0, 1.0), the indices are identical to physical coordinates in meters.
    sources_physical = sources * dx_spacing
    receivers_physical = receivers * dx_spacing

    # Devito expects time in ms (data.time is in seconds)
    t0 = time[0] * 1000.0
    tn = time[-1] * 1000.0
    dt_data = (time[1] - time[0]) * 1000.0

    # Devito expects shots to have shape (n_sources, nt, n_receivers) and dtype float32
    # The loaded shots array has shape (n_sources, n_receivers, nt)
    # We transpose axes 1 and 2 and cast to float32 to obtain the expected shape and type.
    shots_transposed = np.transpose(shots, (0, 2, 1)).astype(np.float32)
    print(f"Transposed shots shape: {shots_transposed.shape}")

    # 4. Initialize Custom RTM
    rtm = CustomReverseTimeMigration(
        vp=velocity,
        n_sources=sources.shape[0],
        n_receivers=receivers.shape[1],  # Number of receivers per shot (48)
        origin=(0.0, 0.0),
        spacing=spacing,
        nbl=nbl,
        t0=t0,
        tn=tn,
        f0=f0,
        smooth_sigma=5.0,
        dt=dt_data,
        space_order=4,
    )

    # 5. Run Migration
    print("Running Reverse Time Migration...")
    migrated_image = rtm.migrate_from_data(
        shot_records=shots_transposed,
        sources=sources_physical,
        receivers=receivers_physical,
        save_wavefield=False
    )

    # 6. Post-process and Plot
    # We apply Laplace filter to the migrated image to remove low-frequency acquisition footprints.
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
