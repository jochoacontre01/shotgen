import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seisplot
import numpy as np
from scipy.ndimage import gaussian_filter
import pylops
import h5py
import os
from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration
import segyio
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import subprocess
import time

configuration["log-level"] = "WARNING"

        
class ShotRecord:
    """
    A class to generate and manage 2D acoustic wave propagation shot records.

    This class handles the setup of velocity models, source/receiver geometries,
    running simulations using Born perturbation via PyLops, and visualizing results.

    Attributes
    ----------
    nx : int
        Number of samples in the x-direction.
    nz : int
        Number of samples in the z-direction.
    dx : float
        Grid spacing in the x-direction (fixed at 1).
    dz : float
        Grid spacing in the z-direction (fixed at 1).
    n_receivers : int
        Total number of receivers.
    n_sources : int
        Total number of sources.
    recs : ndarray
        2D array of receiver coordinates (x, z).
    sources : ndarray
        2D array of source coordinates (x, z).
    vel : ndarray or None
        The velocity model grid.
    v0 : ndarray or None
        The smoothed background velocity model used for Born modeling.
    shot_run : ndarray or None
        The simulated shot records after running the modeling.
    """
    
    def __init__(
        self,
        nx,
        nz,
        dx,
        dz,
        n_receivers,
        n_sources,
        f0=25.0,
        src_origin=(0,0),
        rec_origin=(0,0),
        origin=(0,0),
        meters_per_cell=1.,
        fd_order=4,
        n_damping=100,
        gather="cmp",
        group_offset=1,
        shot_offset=1,
        smooth=5,
        noise_scale=0.05,
        engine="pylops",
        float_type=np.float32
    ):
        """
        Initialize the ShotRecord with grid dimensions and survey geometry.

        Parameters
        ----------
        nx : int
            Number of grid points in x.
        nz : int
            Number of grid points in z.
        dx : float
            Sample spacing in the x direction.
        dz : float
            Sample spacing in the z direction.
        n_receivers : int
            Number of receivers to place along the surface.
        n_sources : int
            Number of sources to place along the surface.
        f0 : float
            Central frequency of the wavelet in Hz
        origin : tuple
            First point of the first source
        fd_order : int
            Order of the Finite Differences equation
        n_damping : int
            Number of cells to use in the damping border
        gather : str
            Type of shot gather. Can be 'common midpoint', 'common shot'.
        noise_scale : float
            Scale of the random noise to add to the shot data as a percentage of the maximum shot amplitude
        """
        self.engine = engine
        self.float_type = float_type
        
        self.nx = nx 
        self.nz = nz 
        self.dx = dx 
        self.dz = dz
        
        self.meters_per_cell = meters_per_cell
        x = np.arange(0, nx, self.dx, dtype=self.float_type)#*self.dx
        z = np.arange(0, nz, self.dz, dtype=self.float_type)#*self.dz
        
        self.n_receivers = n_receivers
        self.n_sources = n_sources
        self.group_offset = group_offset
        self.shot_offset = shot_offset
        self.gather = gather
        
        self._model_ready = False
        self.vel = None
        self.smooth = smooth
        self.origin = origin
        self.noise_scale = noise_scale
        
        self.src_origin = src_origin
        self.rec_origin = rec_origin
        
        if self.gather == "common shot":
            initial_nx = int(self.nx)
            new_nx = self.src_origin[0] + self.n_sources*self.shot_offset + self.n_receivers*self.group_offset
            self._set_common_shot()
            
            if new_nx > initial_nx:
                self.nx = int(new_nx)
                warnings.warn(
                    f"\nThe initial shape ({int(initial_nx)}, {int(self.nz)}) is too small for the required geometry."
                    f"\nAfter modification, the new shape is ({int(self.nx)}, {int(self.nz)})",
                    category=UserWarning
                )
        elif self.gather == "common midpoint":
            # Receivers
            nr = self.n_receivers
            rx = np.linspace(self.rec_origin[0], x[-1], nr, dtype=self.float_type)
            rz = np.ones(nr, dtype=self.float_type)*self.rec_origin[1]
            self.recs = np.vstack((rx, rz)).T

            # sources
            ns = self.n_sources
            sx = np.linspace(self.src_origin[0], x[-1], ns, dtype=self.float_type)
            sz = np.ones(ns, dtype=self.float_type)*self.src_origin[1]
            sources = np.vstack((sx, sz))
            self.sources = sources.T if sources.ndim >= 2 else sources.reshape((-1,2))
        
        self.x = x
        self.z = z
        
        self.shot_run = None
        self.aop = None
        
        self.f0 = f0
        self.fd_order = fd_order
        self.n_damping = n_damping
        self.src = None
        
        self.X, self.Z = np.meshgrid(np.arange(self.nx, dtype=self.float_type), np.arange(self.nz, dtype=self.float_type), indexing='ij')
    
    def _set_common_shot(self):
        
        rec_span = (self.n_receivers-1) * self.group_offset
        src_span = (self.n_sources-1) * self.shot_offset
        
        sx = np.linspace(0, src_span, self.n_sources, dtype=self.float_type) + self.src_origin[0]
        sz = np.ones(self.n_sources, dtype=self.float_type) * self.src_origin[1]
        self.sources = np.vstack([sx, sz]).T
        
        rx_list = []
        for si in range(self.n_sources):
            src_dx = (si+1) * self.shot_offset
            rec_x = np.linspace(0, rec_span, self.n_receivers, dtype=self.float_type) + src_dx + self.rec_origin[0]
            rec_z = np.ones(self.n_receivers, dtype=self.float_type) * self.rec_origin[1]
            rx_list.append(np.vstack([rec_x, rec_z]).T)
        
        self.recs = np.array(rx_list)
                
    def set_model(self, model):
        if model.shape[0] < self.nx:
            warnings.warn(
                "\nThe input model does not match the internal model size"
                f"\nExpected ({int(self.nx)}, {int(self.nz)}) but got {model.shape}"
                "\nThe input model will be padded"
            )
        
            diff = self.nx - model.shape[0]
            model = np.pad(model, ((0, int(diff)), (0,0)), mode="edge")
        self.vel = model
        self._model_ready = True
        
    def show_model(self, draw_recs=True, cli=False, **kwargs):
        """
        Plot the current velocity model with source and receiver positions.

        Raises
        ------
        ValueError
            If no velocity model has been initialized yet.
        """
        if not self._model_ready:
            raise ValueError("You need to create a model first")
        else:
            if self.gather == "common shot":
                recs_4plot_x = (self.recs.reshape(-1,2)[:,0])
                recs_4plot_z = (self.recs.reshape(-1,2)[:,1])
            elif self.gather == "common midpoint":
                recs_4plot_x = self.recs[:, 0]
                recs_4plot_z = self.recs[:, 1]
                
            plt.figure(figsize=(10, 5))
            im = plt.imshow(self.vel.T, extent=(self.origin[0], self.nx, self.nz, self.origin[-1]), **kwargs)
            if draw_recs:
                plt.scatter(recs_4plot_x, recs_4plot_z, marker="v", s=150, c="b", edgecolors="k")
                plt.scatter(self.sources[:, 0], self.sources[:, 1], marker="*", s=150, c="r", edgecolors="k")
            cb = plt.colorbar(im)
            cb.set_label("[m/s]")
            plt.gca().set_aspect("equal")
            plt.axis("tight")
            plt.xlabel("x [m]"), plt.ylabel("z [m]")
            plt.title("Velocity")
            plt.xlim(self.origin[0], self.nx)
            plt.tight_layout()
            if cli:
                plt.savefig("img.png", dpi=100)
                subprocess.run("timg img.png".split())
                time.sleep(0.5)
                subprocess.run("rm img.png".split())
            else:
                plt.show()
        
    def set_source_position(self, x_pos, y_pos):
        src_pos = np.vstack([x_pos, y_pos])
        self.sources = src_pos.T if src_pos.ndim >= 2 else src_pos.reshape((-1,2))
    
    def set_receiver_position(self, x_pos, y_pos):
        rec_pos = np.vstack([x_pos, y_pos]).T
        self.recs = rec_pos
        
    def _setup_devito(self, ms):
        vel = self.vel / 1000 # to km/s
        self._devito_model = Model(
            vp=vel,
            origin=self.origin,
            shape=vel.shape,
            spacing=(self.meters_per_cell, self.meters_per_cell),
            space_order=self.fd_order,
            nbl=self.n_damping,
            bcs="damp",
            dtype=np.float32,
            grid=None
        )
        
        v0 = gaussian_filter(vel, sigma=self.smooth)
        self._devito_model0 = Model(
            vp=v0,
            origin=self.origin,
            shape=vel.shape,
            spacing=(self.meters_per_cell, self.meters_per_cell),
            space_order=self.fd_order,
            nbl=self.n_damping,
            bcs="damp",
            dtype=np.float32,
            grid=None
        )
        
        
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, :] = np.array(self._devito_model.domain_size, dtype=self.float_type) * 0.5
        src_coordinates[0, -1] = 0.0
        
        f0 = self.f0 / 1000 # to kHz
        self._devito_geometry = AcquisitionGeometry(
            model=self._devito_model,
            rec_positions=self.recs,
            src_positions=src_coordinates,
            t0=0,
            tn=ms,
            f0=f0,
            src_type="Ricker"
        )
        
        self._devito_solver = AcousticWaveSolver(
            model=self._devito_model,
            geometry=self._devito_geometry,
            space_order=self.fd_order,
            time_order=2
        )
        
    def _execute_devito(self, save_wavefield=False, save_each=20):
        shots = []
        us = []
        
        for i in tqdm(range(self.n_sources), desc="Source", total=self.n_sources):

            self._devito_geometry.src_positions[0, :] = self.sources[i, :]
            
            true_d, _, _ = self._devito_solver.forward(vp=self._devito_model.vp)
            smooth_d, u0, _ = self._devito_solver.forward(vp=self._devito_model0.vp, save=True)
                    
            residual = smooth_d.data - true_d.data
            if save_wavefield:
                us.append(u0.data.copy()[::save_each])
            shots.append(residual.T)
        
        self.shot_run = np.array(shots, dtype=self.float_type)
        self.us = np.array(us, dtype=self.float_type)
        self.src = self._devito_geometry.src.wavelet
        # self.dt = self._devito_model0.critical_dt
        
    def _execute_pylops(self, ms):
        dv = self.vel**(-2) - self.v0**(-2)
        
        if self.gather == "common shot":
            # Define a helper function to process a single shot
            def _process_single_shot(si, s):
                Aop = pylops.waveeqprocessing.AcousticWave2D(
                    shape=(self.nx, self.nz),
                    origin=(0,0),
                    spacing=(self.dx, self.dz),
                    vp=self.v0,
                    src_x=np.array([s[0]], dtype=float),
                    src_z=np.array([s[1]], dtype=float),
                    rec_x=self.recs[si][:, 0],
                    rec_z=self.recs[si][:, 1],
                    t0=0.0,
                    tn=ms,
                    src_type="Ricker",
                    space_order=self.fd_order,
                    nbl=self.n_damping,
                    f0=self.f0,
                    dtype=str(np.dtype(self.float_type))
                )
                return (Aop @ dv)[0]

            # Run the simulation in parallel using all available cores (n_jobs=-1)
            run = Parallel(n_jobs=-1)(
                delayed(_process_single_shot)(si, s) 
                for si, s in tqdm(enumerate(self.sources), desc="Source", total=self.n_sources)
            )
            run = np.array(run, dtype=self.float_type)
            
            noise = np.random.normal(0, self.noise_scale*np.median(run), run.shape).astype(self.float_type)
            self.shot_run = run + noise
            
            # Re-instantiate the last operator to populate metadata (self.aop, self.src, self.dt)
            # This is necessary because the parallel workers do not update the main object instance
            s = self.sources[-1]
            si = self.n_sources - 1
            self.aop = pylops.waveeqprocessing.AcousticWave2D(
                shape=(self.nx, self.nz),
                origin=(0,0),
                spacing=(self.dx, self.dz),
                vp=self.v0,
                src_x=np.array([s[0]], dtype=self.float_type),
                src_z=np.array([s[1]], dtype=self.float_type),
                rec_x=self.recs[si][:, 0],
                rec_z=self.recs[si][:, 1],
                t0=0.0,
                tn=ms,
                src_type="Ricker",
                space_order=self.fd_order,
                nbl=self.n_damping,
                f0=self.f0,
                dtype=str(np.dtype(self.float_type))
            )
            self.src = self.aop.geometry.src.data[:, 0]
            self.dt = self.aop.geometry.dt
                
        elif self.gather == "common midpoint":
            Aop = pylops.waveeqprocessing.AcousticWave2D(
                    shape=(self.nx, self.nz),
                    origin=(0,0),
                    spacing=(self.dx, self.dz),
                    vp=self.v0,
                    src_x=self.sources[:, 0],
                    src_z=self.sources[:, 1],
                    rec_x=self.recs[:, 0],
                    rec_z=self.recs[:, 1],
                    t0=0.0,
                    tn=ms,
                    src_type="Ricker",
                    space_order=self.fd_order,
                    nbl=self.n_damping,
                    f0=self.f0,
                    dtype=str(np.dtype(self.float_type))
                )
            self.aop = Aop
            self.dt = self.aop.geometry.dt
            
            run = Aop @ dv
            noise = np.random.normal(0, self.noise_scale*np.median(run), run.shape)
            self.shot_run = run + noise
            self.src = self.aop.geometry.src.data[:, 0]
            
    def run(self, ms=500, gain=None, **devito_kwargs):
        """
        Execute the 2D Acoustic Wave simulation using Born modeling.

        Parameters
        ----------
        ms : float, optional
            Total simulation time in milliseconds. Default is 500ms.
        gain : float, optional
            Gain factor to apply to the data, using t^factor amplification. Default is None.
        devito_kwargs : dict, optional
            Devito simulation keyword arguments if necessary. The available arguments are: `save_wavefield` (bool) and `save_each` (int)

        Returns
        -------
        ndarray
            The modeled shot data (Born perturbation result).

        Raises
        ------
        ValueError
            If no velocity model has been initialized before running.
        """
        if self._model_ready:
            self.v0 = gaussian_filter(self.vel, sigma=self.smooth)
            
            if self.engine.lower() == "pylops":
                self._execute_pylops(ms)
                
            elif self.engine.lower() == "devito":
                self._setup_devito(ms)
                self._execute_devito(**devito_kwargs)
            
            if gain is not None:
                self.shot_run = self.apply_gain(gain)
            
            nelements_time = self.shot_run.shape[-1]
            self.time_vector = np.linspace(0, ms, nelements_time)*(1e-3)
            # self.dt = np.diff(self.time_vector)[0] # dt in s
            return self.shot_run
        else:
            raise ValueError("You need to create a model before running a simulation")
            
    def apply_gain(self, factor=2):
        return self.shot_run * (self.aop.geometry.time_axis.time_values**factor)
            
    def save_shot(self, name, overwrite=True):
        """
        Save the simulation results and metadata to a specified folder.
        
        The files saved in the folder are:
        - traces.segy (Seismic traces and geometry)
        - velocity_model.segy (Velocity model matrix)
        - smooth_velocity.segy (Smooth background velocity model)
        - metadata.h5 (Other scalar values and wavelets)

        Parameters
        ----------
        name : str
            Directory path to save the files into.
        overwrite : bool, optional
            Whether to overwrite the files if the directory already exists. Default is True.
        """
        import os
        import h5py
        from shotgen.io import SegyIO
        
        if os.path.exists(name) and not overwrite:
            print(f"Existing folder with overwrite set to {overwrite} could not be created")
            return
            
        os.makedirs(name, exist_ok=True)
        
        SegyIO.write(os.path.join(name, "traces.segy"), self)
        if self.vel is not None:
            SegyIO.write_model(os.path.join(name, "velocity_model.segy"), self.vel, self.dx, self.dz)
        if hasattr(self, 'v0') and self.v0 is not None:
            SegyIO.write_model(os.path.join(name, "smooth_velocity.segy"), self.v0, self.dx, self.dz)
            
        with h5py.File(os.path.join(name, "metadata.h5"), mode="w") as f:
            if hasattr(self, 'time_vector') and self.time_vector is not None:
                f.create_dataset("time", data=self.time_vector, shape=self.time_vector.shape)
            if hasattr(self, 'src') and self.src is not None:
                f.create_dataset("wavelet", data=self.src)
            if hasattr(self, 'f0'):
                f.create_dataset("f0", data=self.f0)
                
        print(f"Saved simulation files to folder {name}")
    
    def show_shot(self, cmap="seismic"):
        """
        Visualize all generated shot records side-by-side.
        """
        if self.shot_run is not None:
            
            shots_stack = np.hstack([shot.T for shot in self.shot_run])
            vmax = np.max([np.abs(np.amin(self.shot_run)), np.abs(np.amax(self.shot_run))])
            vmin = -vmax
            
            fig = plt.figure(figsize=(10, 6))
            try:
                norm = TwoSlopeNorm(0, vmin, vmax)
            except ValueError:
                norm = None
                
            im = plt.imshow(
                shots_stack, aspect="auto",
                        extent=(
                            0,
                            shots_stack.shape[1],
                            self.time_vector[-1],
                            0,
                        ),
                        norm=norm,
                        cmap=cmap,
                )
        
                
            fig.suptitle("Shot record", y=0.99)
            plt.colorbar(im)
            fig.supxlabel("rec [m]")
            fig.supylabel("t [s]")
            # plt.subplots_adjust(wspace=0)
            plt.show()
            
                

class LoadShotRecord:
    """
    A class to load and visualize shot records from HDF5 files.

    Attributes
    ----------
    receivers : ndarray
        Receiver coordinates loaded from file.
    sources : ndarray
        Source coordinates loaded from file.
    velocity_model : ndarray
        The original velocity model used in the simulation.
    smooth_velocity : ndarray
        The background velocity model used in the simulation.
    time : ndarray
        The time axis values.
    shots : ndarray
        A 3D stack of all shot records (n_shots, n_receivers, n_time).
    nshots : int
        The number of shots found in the file.
    """
    
    def __init__(self, path):
        """
        Initialize LoadShotRecord and automatically load data from the given path.

        Parameters
        ----------
        path : str
            The filesystem path to the HDF5 file.
        """
        self.receivers = None
        self.sources = None
        self.velocity_model = None
        self.smooth_velocity = None
        self.time = None
        self.shots = None
        self.nshots = None
        self.wavelet = None
        
        self._load_shot(path)
        
    def _load_shot(self, path):
        """
        Read datasets from the specified folder and populate class attributes.

        Parameters
        ----------
        path : str
            The filesystem path to the simulation folder.
        """
        import os
        import h5py
        from shotgen.io import SegyIO
        
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a valid directory.")
            
        traces_path = os.path.join(path, "traces.segy")
        vel_path = os.path.join(path, "velocity_model.segy")
        v0_path = os.path.join(path, "smooth_velocity.segy")
        meta_path = os.path.join(path, "metadata.h5")
        
        # Load traces and geometry
        if os.path.exists(traces_path):
            segy_data = SegyIO.read(traces_path)
            self.receivers = segy_data["receivers"]
            self.sources = segy_data["sources"]
            
            # Reshape shots data
            flat_data = segy_data["data"]
            n_traces, n_time = flat_data.shape
            
            # Reconstruct (n_sources, n_receivers, n_time)
            # Find unique sources by rounding to ignore minor floating point diffs
            unique_sources = np.unique(np.round(self.sources, 3), axis=0)
            n_sources = unique_sources.shape[0]
            n_receivers = n_traces // n_sources
            
            if n_sources * n_receivers == n_traces:
                self.shots = flat_data.reshape((n_sources, n_receivers, n_time))
            else:
                self.shots = flat_data # Fallback to flat if it's irregular
                
            self.nshots = n_sources
        
        # Load models
        if os.path.exists(vel_path):
            self.velocity_model = SegyIO.read_model(vel_path)
        if os.path.exists(v0_path):
            self.smooth_velocity = SegyIO.read_model(v0_path)
            
        # Load metadata
        if os.path.exists(meta_path):
            with h5py.File(meta_path, "r") as f:
                if "time" in f:
                    self.time = f["time"][()]
                if "wavelet" in f:
                    self.wavelet = f["wavelet"][()]
                if "f0" in f:
                    self.f0 = f["f0"][()]

    def plot(self, **kwargs):
        """
        Plot a single shot record from the loaded data.

        Parameters
        ----------
        shot_number : int, optional
            The index of the shot to plot. Default is 0.
        """
        shots_stack = np.hstack([shot.T for shot in self.shots]).T
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fig, ax = seisplot.plot(shots_stack, fig=fig, ax=ax, linewidth=0.1, vaxis=self.time, hlabel="rec (m)", vlabel="Two-way travel time (s)", title="Shot gather", colorbar=True, **kwargs)
        plt.show()
        
        return plt.gca()
        
    def plot3d(self, nshot=0):
        shot = self.shots[nshot]
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        
        T, R = np.meshgrid(self.time, self.receivers[:,0])
        ax.plot_surface(T, R, shot, cmap="seismic")
        plt.show()

    def wiggle(self):
        shots_stack = np.hstack([shot.T for shot in self.shots]).T
        shots_stack = (shots_stack - np.amin(shots_stack)) / (np.amax(shots_stack) - np.amin(shots_stack))
        for xi, trace in enumerate(shots_stack):
            xi = xi * 1E-2
            trace_toplot = trace+xi
            plt.plot(trace_toplot, self.time, lw=0.1, c="#000000FF")
            plt.fill_betweenx(self.time, np.mean(trace_toplot), trace_toplot, trace_toplot>xi, color="k")
        plt.gca().invert_yaxis()
        plt.show()
        
def load_marmousi():
    
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/vp_marmousi-ii.segy"

    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:])*1000
    
    return seismic_data

def load_sigsbee(reflection_coeffs=False):
    
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/sigsbee2a_stratigraphy.sgy"
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:])/3.281
    
    if reflection_coeffs:
        filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/sigsbee2a_reflection_coefficients.sgy"
        with segyio.open(filepath, "r", ignore_geometry=True) as f:
            ref_coeffs = np.array(f.trace.raw[:])/3.281
        
        return seismic_data, ref_coeffs
    return seismic_data, None

def load_complex_graben():
    
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/complex_graben.sgy"

    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = f.trace.raw[:][:,::-1]
    
    return seismic_data
