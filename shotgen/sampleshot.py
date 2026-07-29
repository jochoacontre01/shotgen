import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import seisplot
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
import os
import shutil
import ctypes
import sys
import types
import importlib.machinery





# Environment variables for OpenACC GPU execution
os.environ["ACC_DEVICE_TYPE"] = "nvidia"
os.environ["ACC_DEVICE_NUM"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["NVCOMPILER_ACC_TIME"] = "1"
os.environ["OMP_TARGET_OFFLOAD"] = "DISABLED"
os.environ["NVCOMPILER_ACC_NOTIFY"] = "0"

# Tell NVHPC compiler flags to target CUDA OpenACC explicitly
os.environ["DEVITO_OPTIONS"] = "compiler=nvc++"
os.environ["CFLAGS"] = "-O3 -acc -gpu=cc89 -fPIC -shared"  # cc89 targets Ada Lovelace architecture (RTX 4000 Ada)

from examples.seismic import AcquisitionGeometry, Model
from examples.seismic.acoustic import AcousticWaveSolver
from devito import configuration
import devito.arch.compiler as dac
import segyio
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import subprocess
import time
import contextlib
import joblib

configuration["log-level"] = "WARNING"

class PureNvidiaCompiler(dac.NvidiaCompiler):
    """
    Subclasses NvidiaCompiler to strip out OpenMP flags (-mp, -fopenmp) and -gpu=pinned,
    returning clean OpenACC flags to avoid triggering system GNU libgomp linkage in WSL2.
    """
    def __init_finalize__(self, **kwargs):
        self.cflags = [
            "-O3",
            "-acc",
            "-gpu=cc89",
            "-fPIC",
            "-shared"
        ]

# Register PureNvidiaCompiler into Devito's compiler registry
dac.compiler_registry['nvc++'] = PureNvidiaCompiler
dac.compiler_registry['nvc'] = PureNvidiaCompiler
dac.compiler_registry['custom'] = PureNvidiaCompiler

def detect_device():
    """
    Detect whether CUDA GPU hardware and a suitable Devito GPU compiler (nvc++/nvc) are available.

    Returns
    -------
    str
        'cuda' if CUDA GPU acceleration is available AND an OpenACC GPU compiler (nvc++/nvc) is present, otherwise 'cpu'.
    """
    # Devito requires NVIDIA HPC SDK (nvc++ or nvc) to JIT-compile OpenACC GPU code
    has_gpu_compiler = bool(shutil.which("nvc++") or shutil.which("nvc"))
    if not has_gpu_compiler:
        return "cpu"

    # 1. Check system nvidia-smi command
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode == 0:
                return "cuda"
        except Exception:
            pass

    # 2. Check shared CUDA runtime library
    if _check_cuda_lib():
        return "cuda"

    return "cpu"


def _check_cuda_lib():
    """Helper to check if libcuda runtime library can be loaded via ctypes."""
    for libname in ["libcuda.so", "libcuda.dylib", "nvcuda.dll", "libcudart.so"]:
        try:
            ctypes.CDLL(libname)
            return True
        except Exception:
            pass
    return False


def configure_devito_device(device="auto", platform=None, compiler=None, language=None):
    """
    Configures Devito environment variables and runtime settings for CUDA GPU or CPU.

    Parameters
    ----------
    device : str, optional
        Device to configure: 'auto', 'cuda', or 'cpu'.
    platform : str, optional
        Devito platform override (e.g. 'nvidiaX', 'volta', 'ampere', 'intel64').
    compiler : str, optional
        Devito compiler override (e.g. 'nvc++', 'nvc', 'custom', 'gcc').
    language : str, optional
        Devito code generation language override (e.g. 'openacc', 'openmp', 'C').

    Returns
    -------
    str
        The configured device string ('cuda' or 'cpu').
    """
    if device == "auto" or device is None:
        device = detect_device()

    device = device.lower()

    if device in ("cuda", "gpu"):
        # Determine available GPU compiler
        gpu_compiler = compiler
        if not gpu_compiler:
            if shutil.which("nvc++"):
                gpu_compiler = "nvc++"
            elif shutil.which("nvc"):
                gpu_compiler = "nvc"

        if not gpu_compiler:
            warnings.warn(
                "NVIDIA GPU hardware was detected, but Devito GPU compiler (nvc++/nvc) is not available. "
                "Devito requires NVIDIA HPC SDK (nvc++) to JIT-compile GPU stencils. "
                "Falling back to CPU execution.",
                category=UserWarning
            )
            return configure_devito_device("cpu")

        target_platform = platform if platform else "nvidiaX"
        target_compiler = gpu_compiler
        target_language = language if language else "openacc"

        os.environ["DEVITO_ARCH"] = target_compiler
        os.environ["DEVITO_PLATFORM"] = target_platform
        os.environ["DEVITO_COMPILER"] = target_compiler
        os.environ["DEVITO_LANGUAGE"] = target_language
        os.environ["CC"] = target_compiler
        os.environ["CFLAGS"] = "-O3 -acc -gpu=cc89 -fPIC -shared"

        try:
            from devito import configuration
            configuration["platform"] = target_platform
            configuration["compiler"] = target_compiler
            configuration["language"] = target_language
        except Exception as e:
            warnings.warn(f"Failed to set Devito GPU configuration ({e}). Falling back to CPU.", category=UserWarning)
            return configure_devito_device("cpu")
        device = "cuda"
    else:
        target_platform = platform if platform else "intel64"
        target_compiler = compiler if compiler else "custom"
        target_language = language if language else "C"

        os.environ["DEVITO_PLATFORM"] = target_platform
        os.environ["DEVITO_COMPILER"] = target_compiler
        os.environ["DEVITO_LANGUAGE"] = target_language

        try:
            from devito import configuration
            configuration["platform"] = target_platform
            configuration["compiler"] = target_compiler
            configuration["language"] = target_language
        except Exception as e:
            warnings.warn(f"Failed to set Devito CPU configuration: {e}")
        device = "cpu"

    print(target_platform, target_compiler, target_language)
    return device


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()



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
    
    @property
    def origin(self):
        if not hasattr(self, '_origin'):
            self._origin = (0.0, 0.0)
        return self._origin

    @origin.setter
    def origin(self, value):
        if not hasattr(self, '_origin'):
            self._origin = value

    def validate_spatial_bounds(self):
        max_x = self.nx * self.dx
        max_z = self.nz * self.dz
        if hasattr(self, 'sources') and self.sources is not None:
            local_sources = self.sources - np.array(self.origin)
            if np.any(local_sources[..., 0] < 0.0) or np.any(local_sources[..., 0] > max_x):
                raise ValueError(f"Source coordinate X is outside model bounds [0, {max_x}] relative to the origin.")
            if np.any(local_sources[..., 1] < 0.0) or np.any(local_sources[..., 1] > max_z):
                raise ValueError(f"Source coordinate Z is outside model bounds [0, {max_z}] relative to the origin.")
        if hasattr(self, 'recs') and self.recs is not None:
            local_recs = self.recs - np.array(self.origin)
            if np.any(local_recs[..., 0] < 0.0) or np.any(local_recs[..., 0] > max_x):
                raise ValueError(f"Receiver coordinate X is outside model bounds [0, {max_x}] relative to the origin.")
            if np.any(local_recs[..., 1] < 0.0) or np.any(local_recs[..., 1] > max_z):
                raise ValueError(f"Receiver coordinate Z is outside model bounds [0, {max_z}] relative to the origin.")

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
        snr=None,
        engine="pylops",
        float_type=np.float32,
        device="auto",
        fs_ms=None
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
        snr : float, optional
            Signal-to-noise ratio. Noise is added per-trace such that RMS(trace)/snr = std(noise). If None, no noise is added.
        device : str, optional
            Computation target device: 'auto' (detect CPU or CUDA GPU), 'cpu', or 'cuda'. Default is 'auto'.
        """
        self.engine = engine
        self.float_type = float_type
        self.fs_ms = fs_ms
        
        # Device detection and setup
        if device == "auto" or device is None:
            self.device = detect_device()
        elif device.lower() in ("cuda", "gpu"):
            detected = detect_device()
            if detected != "cuda":
                warnings.warn(
                    "device='cuda' was explicitly requested, but CUDA hardware or compiler was not detected. Devito may fail or fall back.",
                    category=UserWarning
                )
            self.device = "cuda"
        elif device.lower() == "cpu":
            self.device = "cpu"
        else:
            raise ValueError(f"Invalid device '{device}'. Expected 'auto', 'cpu', or 'cuda'.")

        self.device = configure_devito_device(self.device)
        
        self.nx = nx 
        self.nz = nz 
        self.dx = dx 
        self.dz = dz
        
        self.meters_per_cell = meters_per_cell
        x = np.arange(nx, dtype=self.float_type) * self.dx
        z = np.arange(nz, dtype=self.float_type) * self.dz
        
        self.n_receivers = n_receivers
        self.n_sources = n_sources
        self.group_offset = group_offset
        self.shot_offset = shot_offset
        if gather in ("cmp", "common midpoint"):
            self.gather = "common midpoint"
        elif gather in ("cs", "common shot"):
            self.gather = "common shot"
        else:
            self.gather = gather
        
        self._model_ready = False
        self.vel = None
        self.smooth = smooth
        self.v0 = None
        self.us = None
        self.tn = None
        self.origin = origin
        self.snr = snr
        
        self.src_origin = src_origin
        self.rec_origin = rec_origin
        
        if self.gather == "common shot":
            initial_nx = int(self.nx)
            new_nx_physical = self.src_origin[0] + self.n_sources*self.shot_offset + self.n_receivers*self.group_offset
            self._set_common_shot()
            
            if new_nx_physical > initial_nx * self.dx:
                self.nx = int(np.ceil(new_nx_physical / self.dx))
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
            
            # Shift by self.origin to get absolute coordinates
            self.sources[..., 0] += self.origin[0]
            self.sources[..., 1] += self.origin[1]
            self.recs[..., 0] += self.origin[0]
            self.recs[..., 1] += self.origin[1]
        
        self.x = x
        self.z = z
        
        self.shot_run = None
        self.aop = None
        
        self.f0 = f0
        self.fd_order = fd_order
        self.n_damping = n_damping
        self.src = None
        
        self.X, self.Z = np.meshgrid(np.arange(self.nx, dtype=self.float_type), np.arange(self.nz, dtype=self.float_type), indexing='ij')
        self.validate_spatial_bounds()
    
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
        
        # Shift by self.origin to get absolute coordinates
        self.sources[..., 0] += self.origin[0]
        self.sources[..., 1] += self.origin[1]
        self.recs[..., 0] += self.origin[0]
        self.recs[..., 1] += self.origin[1]
                
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
        
    def show_model(self, draw_recs=True, cli=False, hq=False, **kwargs):
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
            extent = (self.origin[0], self.origin[0] + self.nx * self.dx, self.origin[-1] + self.nz * self.dz, self.origin[-1])
            im = plt.imshow(self.vel.T, extent=extent, **kwargs)
            if draw_recs:
                plt.scatter(recs_4plot_x, recs_4plot_z, marker="v", s=150, c="b", edgecolors="k")
                plt.scatter(self.sources[:, 0], self.sources[:, 1], marker="*", s=150, c="r", edgecolors="k")
            cb = plt.colorbar(im)
            cb.set_label("[m/s]")
            plt.gca().set_aspect("equal")
            plt.axis("tight")
            plt.xlabel("x [m]"), plt.ylabel("z [m]")
            plt.title("Velocity")
            plt.xlim(self.origin[0], self.origin[0] + self.nx * self.dx)
            plt.tight_layout()
            if hq:
                import pathlib
                name = pathlib.Path(__file__).resolve().parents[1] / "examples/velmodel.png"
                plt.savefig(name)
                subprocess.run(["bash", "-ic", f"open-on-termux '{name}'"])
                # subprocess.run("rm img.png".split())
            if cli:
                plt.savefig("img.png", dpi=100)
                subprocess.run("chafa -w 9 img.png".split())
                time.sleep(0.5)
                subprocess.run("rm img.png".split())
            else:
                plt.show()
        
    def set_source_position(self, x_pos, y_pos):
        src_pos = np.vstack([x_pos, y_pos])
        self.sources = src_pos.T if src_pos.ndim >= 2 else src_pos.reshape((-1,2))
        self.validate_spatial_bounds()
    
    def set_receiver_position(self, x_pos, y_pos):
        rec_pos = np.vstack([x_pos, y_pos]).T
        self.recs = rec_pos
        self.validate_spatial_bounds()
        
    def _setup_devito(self, ms):
        vel = self.vel / 1000 # to km/s
        self._devito_model = Model(
            vp=vel,
            origin=self.origin,
            shape=vel.shape,
            spacing=(self.dx, self.dz),
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
            spacing=(self.dx, self.dz),
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
            self._devito_geometry.src.coordinates.data[0, :] = self.sources[i, :]
            
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
        import pylops
        dv = self.vel**(-2) - self.v0**(-2)
        
        def _get_rec_coords(si):
            if self.gather == "common shot":
                return self.recs[si][:, 0], self.recs[si][:, 1]
            else:
                return self.recs[:, 0], self.recs[:, 1]

        # Define a helper function to process a single shot
        def _process_single_shot(si, s):
            rec_x, rec_z = _get_rec_coords(si)
            Aop = pylops.waveeqprocessing.AcousticWave2D(
                shape=(self.nx, self.nz),
                origin=self.origin,
                spacing=(self.dx, self.dz),
                vp=self.v0,
                src_x=np.array([s[0]], dtype=float),
                src_z=np.array([s[1]], dtype=float),
                rec_x=rec_x,
                rec_z=rec_z,
                t0=0.0,
                tn=ms,
                src_type="Ricker",
                space_order=self.fd_order,
                nbl=self.n_damping,
                f0=self.f0,
                dtype=str(np.dtype(self.float_type))
            )
            return (Aop @ dv)[0]

        # Execute single-process loop directly for CUDA GPU to avoid IPC deadlock in joblib/loky subprocesses,
        # while using joblib process parallelism for CPU multi-threading.
        if self.device == "cuda":
            run = []
            for si, s in enumerate(tqdm(self.sources, desc="Source")):
                run.append(_process_single_shot(si, s))
        else:
            with tqdm_joblib(tqdm(desc="Source", total=self.n_sources)):
                run = Parallel(n_jobs=-1)(
                    delayed(_process_single_shot)(si, s) 
                    for si, s in enumerate(self.sources)
                )
        run = np.array(run, dtype=self.float_type)
        self.shot_run = run
        
        # Re-instantiate the last operator to populate metadata (self.aop, self.src, self.dt)
        # This is necessary because the parallel workers do not update the main object instance
        s = self.sources[-1]
        si = self.n_sources - 1
        rec_x, rec_z = _get_rec_coords(si)
        self.aop = pylops.waveeqprocessing.AcousticWave2D(
            shape=(self.nx, self.nz),
            origin=self.origin,
            spacing=(self.dx, self.dz),
            vp=self.v0,
            src_x=np.array([s[0]], dtype=self.float_type),
            src_z=np.array([s[1]], dtype=self.float_type),
            rec_x=rec_x,
            rec_z=rec_z,
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
            self.tn = ms
            self.v0 = gaussian_filter(self.vel, sigma=self.smooth)
            
            # Ensure Devito environment variables & runtime configuration match target device
            self.device = configure_devito_device(self.device)
            print(f"[ShotRecord] Running wave simulation on device: {self.device.upper()} (Engine: {self.engine})")

            try:
                if self.engine.lower() == "pylops":
                    self._execute_pylops(ms)
                    
                elif self.engine.lower() == "devito":
                    self._setup_devito(ms)
                    self._execute_devito(**devito_kwargs)
            except Exception as e:
                if self.device == "cuda":
                    warnings.warn(
                        f"[ShotRecord] Simulation failed on CUDA GPU device ({e}). "
                        "Falling back to CPU execution.",
                        category=UserWarning
                    )
                    self.device = "cpu"
                    configure_devito_device("cpu")
                    print(f"[ShotRecord] Retrying wave simulation on device: CPU (Engine: {self.engine})")
                    if self.engine.lower() == "pylops":
                        self._execute_pylops(ms)
                    elif self.engine.lower() == "devito":
                        self._setup_devito(ms)
                        self._execute_devito(**devito_kwargs)
                else:
                    raise e
            
            if self.snr is not None:
                rms = np.sqrt(np.mean(self.shot_run**2, axis=-1, keepdims=True))
                noise_std = rms / self.snr
                noise = np.random.standard_normal(size=self.shot_run.shape).astype(self.float_type)
                self.shot_run += (noise * noise_std)

            if self.fs_ms is not None and self.fs_ms > 0:
                new_nt = int(np.round(ms / self.fs_ms)) + 1
                t_orig = np.linspace(0, ms, self.shot_run.shape[-1])
                t_new = np.linspace(0, ms, new_nt)
                from scipy.interpolate import interp1d
                f_interp = interp1d(t_orig, self.shot_run, axis=-1, kind="cubic", fill_value="extrapolate")
                resampled_shot_run = f_interp(t_new).astype(self.float_type)
                del self.shot_run
                import gc
                gc.collect()
                self.shot_run = resampled_shot_run

            nelements_time = self.shot_run.shape[-1]
            self.time_vector = np.linspace(0, ms, nelements_time)*(1e-3)
            if len(self.time_vector) > 1:
                self.dt = self.time_vector[1] - self.time_vector[0]
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
            if hasattr(self, 'origin') and self.origin is not None:
                f.create_dataset("origin", data=np.array(self.origin))
            if hasattr(self, 'fs_ms') and self.fs_ms is not None:
                f.create_dataset("fs_ms", data=self.fs_ms)
                
        print(f"Saved simulation files to folder {name}")
    
    def show_shot(self, cmap="seismic", cli=False, hq=False):
        """
        Visualize all generated shot records side-by-side.
        """
        if self.shot_run is not None:
            
            shots_stack = np.hstack([shot.T for shot in self.shot_run])
#            vmax = np.max([np.abs(np.amin(self.shot_run)), np.abs(np.amax(self.shot_run))])
            vmax = np.quantile(self.shot_run, 0.95)
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
            if hq:
                import pathlib
                name = pathlib.Path(__file__).resolve().parents[1] / "examples/shots.png"
                plt.savefig(name)
                subprocess.run(["bash", "-ic", f"open-on-termux '{name}'"])
                #subprocess.run("rm img.png".split())
            if cli:
                plt.savefig("img.png")
                subprocess.run("chafa -w 9 img.png".split())
                time.sleep(0.5)
                subprocess.run("rm img.png".split())
            else:
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
            self.velocity_model, self.dx, self.dz = SegyIO.read_model(vel_path)
        if os.path.exists(v0_path):
            self.smooth_velocity, self.dx, self.dz = SegyIO.read_model(v0_path)
            
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
        
def _inject_headers_if_needed(filepath, dx, dz):
    """
    Open the SEG-Y file and inject DX and DZ into the headers if not already set.
    """
    need_injection = False
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        interval_val = f.bin[segyio.BinField.Interval]
        tracecount = len(f.trace)
        if tracecount == 641601:
            n2 = 801
        else:
            n2 = tracecount
            
        if interval_val != int(dz * 1000):
            need_injection = True
        elif tracecount > 1 and f.header[1][segyio.TraceField.GroupX] != int(dx * 1000):
            need_injection = True
            
    if need_injection:
        print(f"Injecting spatial headers into {filepath}: dx={dx}, dz={dz}")
        with segyio.open(filepath, "r+", ignore_geometry=True) as f:
            f.bin[segyio.BinField.Interval] = int(dz * 1000)
            for i in range(tracecount):
                f.header[i][segyio.TraceField.TRACE_SAMPLE_COUNT] = int(dz * 1000)
                f.header[i][segyio.TraceField.TRACE_SAMPLE_INTERVAL] = int(dz * 1000)
                f.header[i][segyio.TraceField.SourceGroupScalar] = -1000
                
                inline = i % n2
                crossline = i // n2
                f.header[i][segyio.TraceField.GroupX] = int(inline * dx * 1000)
                f.header[i][segyio.TraceField.SourceX] = int(inline * dx * 1000)
                f.header[i][segyio.TraceField.CDP_X] = int(inline * dx * 1000)
                f.header[i][segyio.TraceField.GroupY] = int(crossline * dx * 1000)
                f.header[i][segyio.TraceField.SourceY] = int(crossline * dx * 1000)
                
        # Inject DZ into Binary Header bytes 117-118 for absolute compliance
        with open(filepath, "r+b") as f_raw:
            f_raw.seek(3200 + 117)
            f_raw.write(int(dz * 1000).to_bytes(2, byteorder="big"))

def load_marmousi():
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/vp_marmousi-ii.segy"
    _inject_headers_if_needed(filepath, dx=1.0, dz=1.0)
    
    with open(filepath, "rb") as f_raw:
        f_raw.seek(3200 + 117)
        dz_bin = int.from_bytes(f_raw.read(2), byteorder="big") / 1000.0
        
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:]) * 1000
        dz_trace = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT] / 1000.0
        dz_parsed = dz_bin if dz_bin > 0 else dz_trace
        nx_parsed = len(f.trace)
        nz_parsed = f.bin[segyio.BinField.Samples]
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            mult = 1.0 / abs(scalar)
        elif scalar > 0:
            mult = scalar
        else:
            mult = 1.0
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        metadata = {
            "dx": dx_parsed,
            "dz": dz_parsed,
            "nx": nx_parsed,
            "nz": nz_parsed,
            "origin_x": 0.0,
            "origin_z": 0.0
        }
    return seismic_data, metadata

def load_sigsbee(reflection_coeffs=False):
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/sigsbee2a_stratigraphy.sgy"
    _inject_headers_if_needed(filepath, dx=1.0, dz=1.0)
    
    with open(filepath, "rb") as f_raw:
        f_raw.seek(3200 + 117)
        dz_bin = int.from_bytes(f_raw.read(2), byteorder="big") / 1000.0
        
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:])/3.281
        dz_trace = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT] / 1000.0
        dz_parsed = dz_bin if dz_bin > 0 else dz_trace
        nx_parsed = len(f.trace)
        nz_parsed = f.bin[segyio.BinField.Samples]
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            mult = 1.0 / abs(scalar)
        elif scalar > 0:
            mult = scalar
        else:
            mult = 1.0
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        metadata = {
            "dx": dx_parsed,
            "dz": dz_parsed,
            "nx": nx_parsed,
            "nz": nz_parsed,
            "origin_x": 0.0,
            "origin_z": 0.0
        }
    
    if reflection_coeffs:
        ref_path = pathlib.Path(__file__).resolve().parents[1] / "assets/sigsbee2a_reflection_coefficients.sgy"
        _inject_headers_if_needed(ref_path, dx=1.0, dz=1.0)
        with segyio.open(ref_path, "r", ignore_geometry=True) as f_ref:
            ref_coeffs = np.array(f_ref.trace.raw[:])/3.281
        return (seismic_data, metadata), ref_coeffs
    return seismic_data, metadata

def load_complex_graben():
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/complex_graben.sgy"
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = f.trace.raw[:][:,::-1]
    return seismic_data

def load_bpsalt():
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/vel_z6.25m_x12.5m_exact.segy"
    _inject_headers_if_needed(filepath, dx=12.5, dz=6.5)
    
    with open(filepath, "rb") as f_raw:
        f_raw.seek(3200 + 117)
        dz_bin = int.from_bytes(f_raw.read(2), byteorder="big") / 1000.0
        
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:])
        dz_trace = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT] / 1000.0
        dz_parsed = dz_bin if dz_bin > 0 else dz_trace
        nx_parsed = len(f.trace)
        nz_parsed = f.bin[segyio.BinField.Samples]
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            mult = 1.0 / abs(scalar)
        elif scalar > 0:
            mult = scalar
        else:
            mult = 1.0
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        metadata = {
            "dx": dx_parsed,
            "dz": dz_parsed,
            "nx": nx_parsed,
            "nz": nz_parsed,
            "origin_x": 0.0,
            "origin_z": 0.0
        }
    return seismic_data, metadata

def load_overthrust():
    filepath = pathlib.Path(__file__).resolve().parents[1] / "assets/marine_overthrust_3d.segy"
    n2 = 801
    start_trace = 455 * n2
    end_trace = 456 * n2
    _inject_headers_if_needed(filepath, dx=25.0, dz=25.0)
    
    with open(filepath, "rb") as f_raw:
        f_raw.seek(3200 + 117)
        dz_bin = int.from_bytes(f_raw.read(2), byteorder="big") / 1000.0
        
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[start_trace:end_trace])
        dz_trace = f.header[start_trace][segyio.TraceField.TRACE_SAMPLE_COUNT] / 1000.0
        dz_parsed = dz_bin if dz_bin > 0 else dz_trace
        nx_parsed = end_trace - start_trace
        nz_parsed = f.bin[segyio.BinField.Samples]
        scalar = f.header[start_trace][segyio.TraceField.SourceGroupScalar]
        if scalar < 0:
            mult = 1.0 / abs(scalar)
        elif scalar > 0:
            mult = scalar
        else:
            mult = 1.0
        dx_parsed = (f.header[start_trace + 1][segyio.TraceField.GroupX] - f.header[start_trace][segyio.TraceField.GroupX]) * mult
        metadata = {
            "dx": dx_parsed,
            "dz": dz_parsed,
            "nx": nx_parsed,
            "nz": nz_parsed,
            "origin_x": 0.0,
            "origin_z": 0.0
        }
    return seismic_data, metadata

def map_coordinate_to_index(pos, spacing, origin, max_cells):
    """
    Convert a physical coordinate (in meters) to an exact integer cell index
    using a rounded nearest-neighbor calculation.
    
    Parameters
    ----------
    pos : float or ndarray
        Physical position(s) in meters.
    spacing : float
        Grid spacing (dx or dz) in meters.
    origin : float
        Origin coordinate in meters.
    max_cells : int
        Maximum number of cells (nx or nz).
        
    Returns
    -------
    int or ndarray
        Nearest integer cell index/indices.
    """
    pos_arr = np.atleast_1d(pos)
    max_physical = origin + (max_cells - 1) * spacing
    if np.any(pos_arr < origin) or np.any(pos_arr > max_physical):
        raise ValueError(
            f"Physical position {pos} is out of bounds [{origin}, {max_physical}] "
            f"for grid spacing {spacing} and origin {origin}."
        )
    indices = np.floor((pos_arr - origin) / spacing + 0.5).astype(int)
    if np.isscalar(pos):
        return int(indices[0])
    return indices if len(indices) > 1 else int(indices[0])



