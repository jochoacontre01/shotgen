from joblib import Parallel, delayed
import skfmm
from matplotlib import pyplot as plt
import numpy as np
from devito import configuration, TimeFunction, Operator, Eq, solve, Function
from examples.seismic import AcquisitionGeometry, PointSource, Model
from examples.seismic.acoustic import AcousticWaveSolver
import torch
from deepwave import scalar_born
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import scienceplots
import matplotlib
configuration["log-level"] = "WARNING"
plt.style.use(['science','no-latex'])
matplotlib.rcParams.update({"font.size":14})

matplotlib.rcParams.update({"font.size":14})

def load_dataset_dir(dataset_dir, require_f0=False, provided_f0=None):
    import os
    import h5py
    from shotgen.io import SegyIO
    
    traces_path = os.path.join(dataset_dir, "traces.segy")
    v0_path = os.path.join(dataset_dir, "smooth_velocity.segy")
    vel_path = os.path.join(dataset_dir, "velocity_model.segy")
    meta_path = os.path.join(dataset_dir, "metadata.h5")
    
    if os.path.exists(v0_path):
        vp = SegyIO.read_model(v0_path)
    elif os.path.exists(vel_path):
        vp = SegyIO.read_model(vel_path)
    else:
        raise FileNotFoundError(f"Neither smooth_velocity.segy nor velocity_model.segy found in {dataset_dir}")
        
    segy_data = SegyIO.read(traces_path)
    receivers = segy_data["receivers"]
    sources = segy_data["sources"]
    time = segy_data["time"]
    
    flat_data = segy_data["data"]
    n_traces, n_time = flat_data.shape
    
    # We must preserve the original order of sources, np.unique might sort them.
    # Instead, we find how many unique sources by counting.
    # Actually, we can just reshape since data is sequentially saved.
    unique_src = np.unique(np.round(sources, 3), axis=0)
    n_sources = unique_src.shape[0]
    n_receivers = n_traces // n_sources
    
    shots = flat_data.reshape((n_sources, n_receivers, n_time))
    
    # Reshape coordinates back to their original multidimensional shapes
    sources = sources.reshape((n_sources, n_receivers, 2))[:, 0, :]
    receivers = receivers.reshape((n_sources, n_receivers, 2))
    
    f0 = provided_f0
    if os.path.exists(meta_path) and f0 is None:
        with h5py.File(meta_path, "r") as f:
            if "f0" in f:
                f0 = f["f0"][()]
                
    if require_f0 and f0 is None:
        raise ValueError(f"f0 is required for RTM but was not found in {meta_path} and not provided as an argument.")
        
    return vp, sources, receivers, shots, time, f0

class ReverseTimeMigration:
    """
    The `ReverseTimeMigration` class builds on top of `Devito`, `PyLops` and `shotgen` to create an RTM image.
    
    It supports two modes:
    1. Simulation Mode: Generates forward data and migrations internally from standard acquisition geometry.
    2. Data Mode: Runs RTM directly using pre-recorded/simulated dataset parameters (e.g. from LoadShotRecord).
    """
    
    def __init__(
        self,
        vp: np.ndarray = None,
        *args,
        dataset_dir: str = None,
        sources: np.ndarray = None,
        receivers: np.ndarray = None,
        shots: np.ndarray = None,
        time: np.ndarray = None,
        spacing: tuple = (1.0, 1.0),
        nbl: int = 40,
        smooth_sigma: float = 5.0,
        f0: float = None,
        space_order: int = 4,
        time_order: int = 2,
        dtype=np.float32,
        **kwargs,
    ):
        # Parse arguments to support both positional and keyword arguments for both modes
        if dataset_dir is not None:
            vp, sources, receivers, shots, time, f0 = load_dataset_dir(dataset_dir, require_f0=True, provided_f0=f0)
            self.from_data = True
        elif sources is not None:
            self.from_data = True
        else:
            self.from_data = False
            if f0 is None:
                f0 = 25.0 # default fallback for simulation mode

        if self.from_data:
            # Keyword/data-driven signature
            origin = kwargs.get("origin", (0.0, 0.0))
            
            n_sources = sources.shape[0]
            n_receivers = receivers.shape[1] if receivers.ndim == 3 else receivers.shape[0]
            
            # Convert time vector (seconds) to ms
            t0 = time[0] * 1000.0
            tn = time[-1] * 1000.0
        else:
            # Simulation signature: parse n_sources, n_receivers, origin, t0, tn from positional args or kwargs.
            # Other parameters like spacing, nbl, smooth_sigma, f0, space_order, time_order, dtype are already
            # bound to local variables via the method signature.
            n_sources = args[0] if len(args) > 0 else kwargs.get("n_sources")
            n_receivers = args[1] if len(args) > 1 else kwargs.get("n_receivers")
            origin = args[2] if len(args) > 2 else kwargs.get("origin")
            
            if len(args) > 3:
                spacing = args[3]
            if len(args) > 4:
                nbl = args[4]
                
            t0 = args[5] if len(args) > 5 else kwargs.get("t0")
            tn = args[6] if len(args) > 6 else kwargs.get("tn")
            
            if len(args) > 7:
                f0 = args[7]
            if len(args) > 8:
                smooth_sigma = args[8]
            if len(args) > 9:
                dtype = args[9]
            if len(args) > 10:
                space_order = args[10]
            if len(args) > 11:
                time_order = args[11]

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
        self.f0 = f0 / 1000  # Convert Hz/kHz to Devito kHz
        self.dtype = dtype
        
        self._create_model()
        
        if self.from_data:
            self.sources = sources
            self.receivers = receivers
            # Transpose to (n_sources, nt, n_receivers) and convert to float32
            self.shots = np.transpose(shots, (0, 2, 1)).astype(np.float32)
            
            self._create_geometry()
            
            # Resample geometry to the dataset's dt (in ms)
            dt_data = (time[1] - time[0]) * 1000.0
            self.geometry.resample(dt_data)
        else:
            source_locations = np.empty((self.n_sources, 2), dtype=np.float32)
            source_locations[:, 0] = np.linspace(0, self.model.domain_size[0], num=self.n_sources)
            source_locations[:, 1] = 0.
            self.sources = source_locations
            
            rec_locations = np.empty((self.n_receivers, 2))
            rec_locations[:, 0] = np.linspace(0, self.model.domain_size[0], num=self.n_receivers)
            rec_locations[:, 1] = 0.
            self.receivers = rec_locations
            
            self._create_geometry()
            
        self._setup_solver()
        
    def _create_model(self):
        self.model = Model(
            vp=self.vp,
            origin=self.origin,
            shape=self.vp.shape,
            spacing=self.spacing,
            space_order=self.space_order,
            nbl=self.nbl,
            bcs="damp",
            dtype=np.float32,
            grid=None
        )
        
        self.model0 = Model(
            vp=self.v0,
            origin=self.origin,
            shape=self.vp.shape,
            spacing=self.spacing,
            space_order=self.space_order,
            nbl=self.nbl,
            bcs="damp",
            dtype=np.float32,
            grid=None
        )
    
    def _create_geometry(self):
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = np.array(self.model.domain_size) * 0.5
        src_coordinates[0, -1] = 0.0
        # If in data mode, use self.receivers if 2D, else use dummy placeholder receivers (will be updated dynamically)
        rec_positions = self.receivers[0] if (self.from_data and self.receivers.ndim == 3) else self.receivers
        self.geometry = AcquisitionGeometry(
            model=self.model,
            rec_positions=rec_positions,
            src_positions=src_coordinates,
            t0=self.t0,
            tn=self.tn,
            f0=self.f0,
            src_type="Ricker"
        )
    
    def _setup_solver(self):
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=self.space_order, time_order=self.time_order)

    def run(self, save_wavefield=False, save_each=5):
        if self.from_data:
            us = []
            vs = []
            image = Function(name="image", grid=self.model.grid)
            operator = self._imaging_operator(self.model, image, save_wavefield=save_wavefield)
            
            for i in tqdm(range(self.n_sources), desc="Source", total=self.n_sources):
                self.geometry.src_positions[0, :] = self.sources[i, :]
                
                # Update receiver coordinates for this shot dynamically
                current_recs = self.receivers[i] if self.receivers.ndim == 3 else self.receivers
                self.geometry.rec.coordinates.data[:, :] = current_recs
                self.residual_source.coordinates.data[:, :] = current_recs
                
                _, u0, _ = self.solver.forward(vp=self.model0.vp, save=True, dt=self.geometry.dt)
                
                if save_wavefield:
                    v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
                else:
                    v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order)
                
                operator(u=u0, v=v, vp=self.model0.vp, dt=self.geometry.dt, residual=self.shots[i])
                
                if save_wavefield:
                    us.append(u0.data.copy()[::save_each])
                    vs.append(v.data.copy()[::save_each])
            
            self.us = np.array(us)
            self.vs = np.array(vs)
            self.image = image.data
            return self.image
        else:
            shots = []
            us = []
            vs = []
            image = Function(name="image", grid=self.model.grid)
            operator = self._imaging_operator(self.model, image, save_wavefield=save_wavefield)
            
            for i in tqdm(range(self.n_sources), desc="Source", total=self.n_sources):
                self.geometry.src_positions[0, :] = self.sources[i, :]
                
                true_d, _, _ = self.solver.forward(vp=self.model.vp)
                smooth_d, u0, _ = self.solver.forward(vp=self.model0.vp, save=True)
                
                if save_wavefield:
                    v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
                else:
                    v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order)
                
                residual = smooth_d.data - true_d.data
                shots.append(residual)
                operator(u=u0, v=v, vp=self.model0.vp, dt=self.model0.critical_dt, residual=residual)
                
                if save_wavefield:
                    us.append(u0.data.copy()[::save_each])
                    vs.append(v.data.copy()[::save_each])
            
            self.shots = np.array(shots)
            self.us = np.array(us)
            self.vs = np.array(vs)
            self.image = image.data
            return self.image
    
    def migrate_from_data(self, shot_records, save_wavefield=False, save_each=5):
        # Kept for backward compatibility
        us = []
        vs = []
        image = Function(name="image", grid=self.model.grid)
        operator = self._imaging_operator(self.model, image)
        nshots = shot_records.shape[0]
        
        # Cast to float32 to avoid warnings
        shot_records = shot_records.astype(np.float32)
        
        for i in tqdm(range(nshots), desc="source", total=nshots):
            self.geometry.src_positions[0, :] = self.sources[i, :]
            
            current_recs = self.receivers[i] if self.receivers.ndim == 3 else self.receivers
            self.geometry.rec.coordinates.data[:, :] = current_recs
            self.residual_source.coordinates.data[:, :] = current_recs
            
            dt = self.geometry.dt if self.from_data else self.model0.critical_dt
            _, u0, _ = self.solver.forward(vp=self.model0.vp, save=True, dt=dt)
            
            if save_wavefield:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
            else:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order)
            
            operator(u=u0, v=v, vp=self.model0.vp, dt=dt, residual=shot_records[i])
            
            if save_wavefield:
                us.append(u0.data.copy()[::save_each])
                vs.append(v.data.copy()[::save_each])
        self.us = np.array(us)
        self.vs = np.array(vs)
        self.image = image.data
        return self.image
            
    
    def _imaging_operator(self, model, image, save_wavefield=False):
        if save_wavefield:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        else:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order)

        u = TimeFunction(name="u", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        
        eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt.T
        
        stencil = Eq(v.backward, solve(eqn, v.backward))
        
        dt = model.grid.stepping_dim.spacing
        
        # Store residual PointSource so coordinates can be updated dynamically
        self.residual_source = PointSource(name="residual", grid=model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        res_term = self.residual_source.inject(field=v.backward, expr=self.residual_source*dt**2/model.m)
        
        image_update = Eq(image, image-u*v)
    
        return Operator([stencil] + res_term + [image_update], subs=model.spacing_map)

    @property
    def src(self):
        return self.geometry.src.wavelet
    
class ReverseTimeMigrationGPU:
    
    def __init__(
        self,
        shot_record=None,
        velocity_model=None,
        dataset_dir=None,
        sources=None,
        receivers=None,
        wavelet=None,
        f0=None,
        time=None,
    ):
        if dataset_dir is not None:
            velocity_model, sources, receivers, shot_record, time, f0 = load_dataset_dir(dataset_dir, require_f0=True, provided_f0=f0)
            # We don't have wavelet exactly from SEGY, so if not loaded elsewhere we need it, but LoadDataset doesn't return wavelet currently unless we read h5
            import h5py
            import os
            meta_path = os.path.join(dataset_dir, "metadata.h5")
            if os.path.exists(meta_path) and wavelet is None:
                with h5py.File(meta_path, "r") as f:
                    if "wavelet" in f:
                        wavelet = f["wavelet"][()]
        self.shot_record = shot_record
        self.velocity_model = velocity_model
        self.sources = sources
        self.receivers = receivers
        self.wavelet = wavelet
        self.f0 = f0
        self.time = time
        self.dt = time[1] - time[0]
        self.dx = 1
        self.nshots = self.shot_record.shape[0]
        self.nreceivers = len(self.receivers)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.acquisition_params = self.set_acquisition_params()
        
    def set_acquisition_params(self):
        nsource_per_shot = 1
        source_locations = torch.from_numpy(self.sources[:,np.newaxis,:]).long().to(self.device)
        receiver_locations = torch.from_numpy(self.receivers).unsqueeze(0).expand(self.nshots, -1, -1).long().to(self.device)

        source_amplitudes = (
            torch.from_numpy(self.wavelet).float().repeat(self.nshots, nsource_per_shot, 1).to(self.device)
        )

        init_velocity_model = torch.from_numpy(self.velocity_model).float().to(self.device)
        
        self.acquisition_params = dict(
            source_locations=source_locations,
            receiver_locations=receiver_locations,
            source_amplitudes=source_amplitudes,
            init_velocity_model=init_velocity_model
        )
        
        return self.acquisition_params
    
    
    def run(self, epochs=3):
        if self.acquisition_params is None:
            raise ValueError("Acquisition params have not been set")
        
        scatter = torch.zeros_like(self.acquisition_params["init_velocity_model"])
        scatter.requires_grad_()
        optimizer = torch.optim.LBFGS([scatter])
        loss_fn = torch.nn.MSELoss()

        observed = torch.from_numpy(self.shot_record).float().to(self.device)
        for epoch in tqdm(range(epochs), desc="Epoch", total=epochs):
            def closure():
                optimizer.zero_grad()
                out = scalar_born(
                    self.acquisition_params["init_velocity_model"], scatter, self.dx, self.dt,
                    source_amplitudes=self.acquisition_params["source_amplitudes"],
                    source_locations=self.acquisition_params["source_locations"],
                    receiver_locations=self.acquisition_params["receiver_locations"],
                    pml_freq=self.f0,
                )
                loss = 1e6 * loss_fn(out[-1], observed)
                loss.backward()
                return loss.item()
            optimizer.step(closure)
        
        return scatter.detach().numpy()


class KirchhoffMigration:
    """
    Kirchhoff pre-stack depth migration (PSDM) for 2D seismic data.

    This class computes a depth-migrated seismic image using Kirchhoff migration
    with traveltimes computed via the Fast Marching Method (FMM).

    The data parameters passed to the `__init__` method (such as velocity, sources, 
    receivers, shots, and time) typically originate from a `ShotRecord` object 
    generated by the `shotgen` module (or loaded via `LoadShotRecord`).

    Parameters
    ----------
    vp : np.ndarray
        The P-wave velocity model of shape (nx, nz), where nx is the number of grid 
        points in the horizontal direction and nz is the number of grid points in 
        the vertical direction.
    sources : np.ndarray
        The grid coordinates of the sources. Shape should be (n_sources, 2) 
        containing coordinate indices (ix, iz).
    receivers : np.ndarray
        The grid coordinates of the receivers. Can be a 2D array of shape 
        (n_receivers, 2) if receivers are fixed, or a 3D array of shape 
        (n_sources, n_receivers, 2) if receivers vary per shot, containing 
        coordinate indices (ix, iz).
    shots : np.ndarray
        The seismic shot record (common-source gathers). Shape should be 
        (n_sources, n_receivers, nt), where nt is the number of time samples.
    time : np.ndarray
        The time axis array of shape (nt,).
    spacing : tuple of float
        The grid spacing (dx, dz) in physical units (e.g. meters).

    Attributes
    ----------
    vp : np.ndarray
        The P-wave velocity model.
    sources : np.ndarray
        The source grid coordinates.
    receivers : np.ndarray
        The receiver grid coordinates.
    shots : np.ndarray
        The seismic shot records.
    time : np.ndarray
        The time axis array.
    spacing : tuple of float
        The spatial grid spacing (dx, dz).
    unique_coords : list of tuple
        A list of unique coordinates (sources and receivers combined) for which 
        traveltime fields are computed.
    traveltime_dict : dict
        A dictionary mapping each coordinate in `unique_coords` to its FMM traveltime 
        field computed over the velocity grid.
    output : np.ndarray
        The migrated image of shape (nx, nz).

    Methods
    -------
    run()
        Run the Kirchhoff migration and return the migrated image.
    """
    
    def __init__(
        self,
        vp:np.ndarray = None,
        sources:np.ndarray = None,
        receivers:np.ndarray = None,
        shots:np.ndarray = None,
        time:np.ndarray = None,
        spacing:tuple = (1.0, 1.0),
        dataset_dir: str = None,
    ):
        if dataset_dir is not None:
            vp, sources, receivers, shots, time, _ = load_dataset_dir(dataset_dir, require_f0=False)

        self.vp = vp
        self.sources = sources
        self.receivers = receivers
        self.shots = shots
        self.time = time
        self.spacing = spacing

        self._gather_unique_coords()
        self._setup_solver()

    def _gather_unique_coords(self):
        all_coords = [tuple(s) for s in self.sources]

        if self.receivers.ndim == 3:
            for i in range(self.receivers.shape[0]):
                for j in range(self.receivers.shape[1]):
                    all_coords.append(tuple(self.receivers[i, j]))
        else:
            for r in self.receivers:
                all_coords.append(tuple(r))

        self.unique_coords = list(set(all_coords))
        
    def _setup_solver(self):
        computed_fields = Parallel(n_jobs=-1)(
            delayed(self.compute_single_traveltime_field)(c, self.vp, self.spacing[0], self.spacing[1])
            for c in self.unique_coords
        )

        # 3. Create a dictionary to instantly look up traveltime fields
        self.traveltime_dict = {coord: field for coord, field in zip(self.unique_coords, computed_fields)}
        self.output = np.zeros_like(self.vp)

    def compute_single_traveltime_field(self, coord, vp, dx, dz):
        """Computes the traveltime field for a single unique coordinate."""
        phi = np.ones_like(vp)
        idx_x, idx_z = int(coord[0]), int(coord[1])
        phi[idx_x, idx_z] = 0
        return skfmm.travel_time(phi, vp, dx=[dx, dz])
        
    def run(self):
        # Build spatial coordinate grids for calculating analytical distances

        nx, nz = self.vp.shape
        x_coords = np.arange(nx) * self.spacing[0]
        z_coords = np.arange(nz) * self.spacing[1]
        X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
        eps = 1e-4 # Small epsilon to prevent division by zero at source/receiver points
        
        for si, source in enumerate(self.sources):
            
            # Source physical coordinates
            sx, sz = source[0] * self.spacing[0], source[1] * self.spacing[1]
            
            # Distance from source to all grid points
            Rs = np.sqrt((X - sx)**2 + (Z - sz)**2) + eps
            
            # Retrieve pre-computed source traveltime field
            traveltime_s = self.traveltime_dict[tuple(source)]
            
            n_receivers = self.receivers.shape[1] if self.receivers.ndim == 3 else self.receivers.shape[0]
            
            for ri in range(n_receivers):
                # Extract receiver coordinates safely whether 2D or 3D array
                rec_coord = self.receivers[si, ri] if self.receivers.ndim == 3 else self.receivers[ri]
                    
                # Total traveltime is just the sum of the two pre-computed fields
                traveltime_r = self.traveltime_dict[tuple(rec_coord)]
                total_traveltime = traveltime_s + traveltime_r
                
                trace = self.shots[si, ri]
                
                # Receiver physical coordinates
                rx = rec_coord[0] * self.spacing[0]
                rz = rec_coord[1] * self.spacing[1]
                
                # Distance from receiver to all grid points
                Rr = np.sqrt((X - rx)**2 + (Z - rz)**2) + eps
                
                # --- 1. Geometric Dispersion (Spreading) ---
                spreading = 1.0 / np.sqrt(Rs * Rr)
                
                # --- 2. Obliquity Factor ---
                obliquity = np.abs(Z - rz) / Rr
                
                # Combined amplitude weight
                weight = spreading * obliquity
                
                # Interpolate amplitudes (forces out-of-bounds to 0.0)
                amplitudes = np.interp(total_traveltime.ravel(), self.time, trace, left=0.0, right=0.0).reshape(total_traveltime.shape)
                
                # Apply the weighting factors and add to output
                self.output += amplitudes * weight

        return self.output


