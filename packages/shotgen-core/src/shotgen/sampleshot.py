import pathlib
import os
import shutil
import subprocess
import time
import warnings
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import gaussian_filter
import segyio
import seisplot
from tqdm import tqdm
from joblib import Parallel, delayed

from shotgen.io import SegyIO
from shotgen.geometry import map_coordinate_to_index
from shotgen.runner import run_gpu_simulation_dict


def _find_assets_dir():
    current = pathlib.Path(__file__).resolve()
    for parent in current.parents:
        assets_path = parent / "assets"
        if assets_path.is_dir():
            return assets_path
    return current.parents[3] / "assets"


def detect_device():
    """
    Detect whether CUDA GPU hardware and a suitable Devito GPU compiler (nvc++/nvc) are available.
    """
    has_gpu_compiler = bool(shutil.which("nvc++") or shutil.which("nvc"))
    if not has_gpu_compiler:
        return "cpu"
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if res.returncode == 0:
                return "cuda"
        except Exception:
            pass
    return "cpu"


def configure_devito_device(device="auto", platform=None, compiler=None, language=None):
    """
    Configures Devito environment variables and runtime settings if shotgen-gpu is installed.
    """
    try:
        from shotgen_gpu.config import configure_devito_device as _config_gpu
        return _config_gpu(device=device, platform=platform, compiler=compiler, language=language)
    except ImportError:
        return "cpu"


class ShotRecord:
    """
    A class to generate and manage 2D acoustic wave propagation shot records.
    Handles grid setup, geometry, PyLops modeling, or isolated Devito GPU modeling via subprocess.
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
                raise ValueError(f"Source coordinate X is outside model bounds [{self.origin[0]}, {self.origin[0] + max_x}].")
            if np.any(local_sources[..., 1] < 0.0) or np.any(local_sources[..., 1] > max_z):
                raise ValueError(f"Source coordinate Z is outside model bounds [{self.origin[1]}, {self.origin[1] + max_z}].")
        if hasattr(self, 'recs') and self.recs is not None:
            local_recs = self.recs - np.array(self.origin)
            if np.any(local_recs[..., 0] < 0.0):
                raise ValueError(f"Receiver coordinate X is before model origin {self.origin[0]}.")
            if np.any(local_recs[..., 1] < 0.0) or np.any(local_recs[..., 1] > max_z):
                raise ValueError(f"Receiver coordinate Z is outside model bounds [{self.origin[1]}, {self.origin[1] + max_z}].")

    def __init__(
        self,
        nx,
        nz,
        dx,
        dz,
        n_receivers,
        n_sources,
        f0=25.0,
        src_origin=(0, 0),
        rec_origin=(0, 0),
        origin=(0, 0),
        meters_per_cell=1.0,
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
        self.engine = engine
        self.float_type = float_type
        self.fs_ms = fs_ms

        # Device detection and setup
        if engine.lower() == "pylops" and (device == "auto" or device is None):
            self.device = "cpu"
        elif device == "auto" or device is None:
            self.device = detect_device()
        elif device.lower() in ("cuda", "gpu"):
            detected = detect_device()
            if detected != "cuda":
                warnings.warn(
                    "device='cuda' was explicitly requested, but CUDA hardware or compiler was not detected.",
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
            max_rec_x = self.rec_origin[0] + self.n_sources * self.shot_offset + (self.n_receivers - 1) * self.group_offset
            max_src_x = self.src_origin[0] + (self.n_sources - 1) * self.shot_offset
            new_nx_physical = max(max_src_x, max_rec_x)
            self._set_common_shot()

            if new_nx_physical > self.origin[0] + initial_nx * self.dx:
                self.nx = int(np.ceil((new_nx_physical - self.origin[0]) / self.dx))
                warnings.warn(
                    f"\nThe initial shape ({int(initial_nx)}, {int(self.nz)}) is too small for trailing receivers."
                    f"\nAfter modification, the new shape is ({int(self.nx)}, {int(self.nz)})",
                    category=UserWarning
                )
        elif self.gather == "common midpoint":
            nr = self.n_receivers
            rx = np.linspace(self.rec_origin[0], self.rec_origin[0] + (nr - 1) * self.group_offset, nr, dtype=self.float_type)
            rz = np.ones(nr, dtype=self.float_type) * self.rec_origin[1]
            self.recs = np.vstack((rx, rz)).T

            ns = self.n_sources
            sx = np.linspace(self.src_origin[0], self.src_origin[0] + (ns - 1) * self.shot_offset, ns, dtype=self.float_type)
            sz = np.ones(ns, dtype=self.float_type) * self.src_origin[1]
            sources = np.vstack((sx, sz))
            self.sources = sources.T if sources.ndim >= 2 else sources.reshape((-1, 2))

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
        rec_span = (self.n_receivers - 1) * self.group_offset
        src_span = (self.n_sources - 1) * self.shot_offset

        sx = np.linspace(0, src_span, self.n_sources, dtype=self.float_type) + self.src_origin[0]
        sz = np.ones(self.n_sources, dtype=self.float_type) * self.src_origin[1]
        self.sources = np.vstack([sx, sz]).T

        rx_list = []
        for si in range(self.n_sources):
            src_dx = (si + 1) * self.shot_offset
            rec_x = np.linspace(0, rec_span, self.n_receivers, dtype=self.float_type) + src_dx + self.rec_origin[0]
            rec_z = np.ones(self.n_receivers, dtype=self.float_type) * self.rec_origin[1]
            rx_list.append(np.vstack([rec_x, rec_z]).T)

        self.recs = np.array(rx_list)

    def set_model(self, model, dx_orig=None, dz_orig=None):
        if dx_orig is not None and dz_orig is not None:
            from shotgen.models import resample_velocity_model
            model, nx_new, nz_new, self.dx, self.dz = resample_velocity_model(
                model, dx_orig, dz_orig, target_dx=self.dx, target_dz=self.dz
            )
            self.nx = nx_new
            self.nz = nz_new

        if hasattr(self, 'recs') and self.recs is not None:
            max_rec_x = float(np.max(self.recs[..., 0]))
            required_nx = int(np.ceil((max_rec_x - self.origin[0]) / self.dx))
            if required_nx > model.shape[0]:
                diff_x = required_nx - model.shape[0]
                warnings.warn(
                    f"\nThe trailing receivers extend past the model grid."
                    f"\nPadding velocity model by {diff_x} cells along X edge using edge mode.",
                    category=UserWarning
                )
                model = np.pad(model, ((0, diff_x), (0, 0)), mode="edge")
                self.nx = model.shape[0]

        if model.shape[0] < self.nx or model.shape[1] < self.nz:
            diff_x = max(0, self.nx - model.shape[0])
            diff_z = max(0, self.nz - model.shape[1])
            model = np.pad(model, ((0, diff_x), (0, diff_z)), mode="edge")
            self.nx = model.shape[0]
            self.nz = model.shape[1]

        self.vel = model
        self._model_ready = True


    def show_model(self, draw_recs=True, cli=False, hq=False, **kwargs):
        if not self._model_ready and (not hasattr(self, "vel") or self.vel is None):
            raise ValueError("You need to create a model first")

        if self.gather == "common shot":
            recs_4plot_x = (self.recs.reshape(-1, 2)[:, 0])
            recs_4plot_z = (self.recs.reshape(-1, 2)[:, 1])
        elif self.gather == "common midpoint":
            recs_4plot_x = self.recs[:, 0]
            recs_4plot_z = self.recs[:, 1]

        kwargs.setdefault("cmap", "turbo")
        kwargs.setdefault("aspect", "auto")

        plt.figure(figsize=(10, 5))
        extent = (self.origin[0], self.origin[0] + self.nx * self.dx, self.origin[-1] + self.nz * self.dz, self.origin[-1])
        im = plt.imshow(self.vel.T, extent=extent, **kwargs)
        if draw_recs:
            plt.scatter(recs_4plot_x, recs_4plot_z, marker="v", s=150, c="w", edgecolors="k")
            plt.scatter(self.sources[:, 0], self.sources[:, 1], marker="*", s=150, c="r", edgecolors="k")
        cb = plt.colorbar(im)
        cb.set_label("[m/s]")
        plt.gca().set_aspect("auto")
        plt.axis("tight")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.title("velocity model")
        plt.xlim(self.origin[0], self.origin[0] + self.nx * self.dx)
        plt.tight_layout()

        if hq:
            name = _find_assets_dir().parent / "examples/velmodel.png"
            plt.savefig(name)
            subprocess.run(["bash", "-ic", f"open-on-termux '{name}'"])
        if cli:
            plt.savefig("img.png", dpi=100)
            plt.close()
            subprocess.run(["chafa", "img.png"])
            subprocess.run(["rm", "img.png"])
        else:
            plt.show()

    def set_source_position(self, x_pos, y_pos):
        src_pos = np.vstack([x_pos, y_pos])
        self.sources = src_pos.T if src_pos.ndim >= 2 else src_pos.reshape((-1, 2))
        self.validate_spatial_bounds()

    def set_receiver_position(self, x_pos, y_pos):
        rec_pos = np.vstack([x_pos, y_pos]).T
        self.recs = rec_pos
        self.validate_spatial_bounds()

    def _execute_pylops(self, ms):
        import pylops
        dv = self.vel**(-2) - self.v0**(-2)

        def _get_rec_coords(si):
            if self.gather == "common shot":
                return self.recs[si][:, 0], self.recs[si][:, 1]
            else:
                return self.recs[:, 0], self.recs[:, 1]

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

        run = []
        for si, s in enumerate(tqdm(self.sources, desc="Source")):
            run.append(_process_single_shot(si, s))

        run = np.array(run, dtype=self.float_type)
        self.shot_run = run

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

    def apply_noise(self, snr=None):
        """
        Applies zero-mean Gaussian noise to self.shot_run based on SNR.
        SNR is defined such that RMS(trace) / SNR = std(noise).
        """
        if snr is not None:
            if isinstance(snr, str) and snr.strip().lower() in ("none", "null"):
                self.snr = None
            else:
                self.snr = float(snr)

        if self.snr is not None and self.shot_run is not None:
            safe_snr = float(self.snr)
            if safe_snr > 0:
                rms = np.sqrt(np.mean(self.shot_run**2, axis=-1, keepdims=True))
                noise_std = rms / safe_snr
                noise = np.random.standard_normal(size=self.shot_run.shape).astype(self.float_type)
                self.shot_run = self.shot_run + (noise * noise_std)

    def run(self, ms=1000.0, **kwargs):
        self.tn = ms
        if not self._model_ready:
            raise RuntimeError("Model is not initialized. Call set_model first.")

        self.v0 = gaussian_filter(self.vel, sigma=self.smooth)

        if self.engine.lower() == "devito":
            import tempfile
            with tempfile.TemporaryDirectory() as tmp_dir:
                cfg = {
                    "task": "simulation",
                    "nx": int(self.nx),
                    "nz": int(self.nz),
                    "dx": float(self.dx),
                    "dz": float(self.dz),
                    "n_sources": int(self.n_sources),
                    "n_receivers": int(self.n_receivers),
                    "f0": float(self.f0),
                    "ms": float(ms),
                    "fd_order": int(self.fd_order),
                    "n_damping": int(self.n_damping),
                    "smooth": float(self.smooth),
                    "origin": [float(self.origin[0]), float(self.origin[1])],
                    "src_origin": [float(self.src_origin[0]), float(self.src_origin[1])],
                    "rec_origin": [float(self.rec_origin[0]), float(self.rec_origin[1])],
                    "group_offset": float(self.group_offset),
                    "shot_offset": float(self.shot_offset),
                    "gather": str(self.gather),
                    "meters_per_cell": float(self.meters_per_cell),
                    "snr": float(self.snr) if self.snr is not None else None,
                    "fs_ms": float(self.fs_ms) if self.fs_ms is not None else None,
                    "sources": self.sources.tolist() if hasattr(self, 'sources') and self.sources is not None else None,
                    "receivers": self.recs.tolist() if hasattr(self, 'recs') and self.recs is not None else None,
                    "v_base": float(np.mean(self.vel)),
                    "device": self.device,
                    "output_dir": tmp_dir,
                    "save_wavefield": kwargs.get("save_wavefield", False),
                }

                vel_file = os.path.join(tmp_dir, "vel.npy")
                np.save(vel_file, self.vel)
                cfg["vel_file"] = vel_file

                res_cfg = os.path.join(tmp_dir, "config.json")
                with open(res_cfg, "w") as f:
                    import json
                    json.dump(cfg, f)

                run_gpu_simulation_dict(cfg)

                res_h5 = os.path.join(tmp_dir, "simulation_results.h5")
                if os.path.exists(res_h5):
                    with h5py.File(res_h5, "r") as f:
                        self.shot_run = f["shots"][()]
                        if "wavelet" in f:
                            self.src = f["wavelet"][()]
                else:
                    raise RuntimeError("GPU Simulation failed to write simulation_results.h5")
        else:
            self._execute_pylops(ms)
            if self.snr is not None:
                self.apply_noise()

        if self.fs_ms is not None and self.fs_ms > 0:
            dt_model_ms = ms / (self.shot_run.shape[-1] - 1) if self.shot_run.shape[-1] > 1 else self.fs_ms
            new_nt = int(np.round(ms / self.fs_ms)) + 1
            t_orig = np.linspace(0, ms, self.shot_run.shape[-1])
            t_new = np.linspace(0, ms, new_nt)
            from scipy.interpolate import interp1d
            f_interp = interp1d(t_orig, self.shot_run, axis=-1, kind="linear", fill_value="extrapolate")
            resampled_shot_run = f_interp(t_new).astype(self.float_type)
            del self.shot_run
            import gc
            gc.collect()
            self.shot_run = resampled_shot_run

        nelements_time = self.shot_run.shape[-1]
        self.time_vector = np.linspace(0, ms, nelements_time) * (1e-3)
        if len(self.time_vector) > 1:
            self.dt = self.time_vector[1] - self.time_vector[0]
        return self.shot_run

    def apply_gain(self, factor=2):
        if self.aop is not None:
            return self.shot_run * (self.aop.geometry.time_axis.time_values**factor)
        return self.shot_run

    def save_shot(self, name, overwrite=True):
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
            f.create_dataset("dx", data=self.dx)
            f.create_dataset("dz", data=self.dz)
            f.create_dataset("nx", data=self.nx)
            f.create_dataset("nz", data=self.nz)
            if hasattr(self, 'fs_ms') and self.fs_ms is not None:
                f.create_dataset("fs_ms", data=self.fs_ms)
            if hasattr(self, 'snr') and self.snr is not None:
                f.create_dataset("snr", data=self.snr)


        print(f"Saved simulation files to folder {name}")

    def show_shot(self, cmap="seismic", cli=False, hq=False, source_idx=0):
        if self.shot_run is None:
            print("No simulation data available to show.")
            return

        if isinstance(self.shot_run, (list, tuple)):
            if len(self.shot_run) == 0:
                print("No shot records available.")
                return
            idx = min(source_idx, len(self.shot_run) - 1)
            shot_data = np.array(self.shot_run[idx])
        elif isinstance(self.shot_run, np.ndarray):
            if self.shot_run.ndim == 3:
                idx = min(source_idx, self.shot_run.shape[0] - 1)
                shot_data = self.shot_run[idx]
            else:
                shot_data = self.shot_run
        else:
            shot_data = np.array(self.shot_run)

        if hasattr(self, "time_vector") and self.time_vector is not None:
            n_time = len(self.time_vector)
        else:
            n_time = shot_data.shape[-1] if shot_data.ndim >= 2 else shot_data.shape[0]

        if shot_data.ndim == 2:
            if shot_data.shape[0] == n_time and shot_data.shape[1] != n_time:
                shot_matrix = shot_data
            else:
                shot_matrix = shot_data.T
        else:
            shot_matrix = shot_data

        vmax = float(np.quantile(np.abs(shot_matrix), 0.98))
        if vmax == 0:
            vmax = 1.0
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(10, 6))

        if hasattr(self, "recs") and self.recs is not None and len(self.recs) > 0:
            rec_pos = self.recs[source_idx] if self.recs.ndim == 3 else self.recs
            x_min = float(rec_pos[0, 0])
            x_max = float(rec_pos[-1, 0])
        elif hasattr(self, "receivers") and self.receivers is not None and len(self.receivers) > 0:
            rec_pos = self.receivers[source_idx] if self.receivers.ndim == 3 else self.receivers
            x_min = float(rec_pos[0, 0])
            x_max = float(rec_pos[-1, 0])
        else:
            x_min = 0.0
            x_max = float((self.n_receivers - 1) * self.dx)

        if hasattr(self, "time_vector") and self.time_vector is not None and len(self.time_vector) > 0:
            t_max = float(self.time_vector[-1])
        elif hasattr(self, "tn") and self.tn is not None:
            t_max = float(self.tn) / 1000.0 if float(self.tn) > 10 else float(self.tn)
        else:
            t_max = 1.0
        t_min = 0.0

        im = ax.imshow(
            shot_matrix,
            aspect="auto",
            extent=(x_min, x_max, t_max, t_min),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )

        src_num = source_idx + 1
        ax.set_title(f"Source {src_num}")
        ax.set_xlabel("rec [m]")
        ax.set_ylabel("t [s]")
        plt.colorbar(im, ax=ax, label="Amplitude")
        plt.tight_layout()

        if hq:
            name = _find_assets_dir().parent / "examples/shots.png"
            plt.savefig(name)
            subprocess.run(["bash", "-ic", f"open-on-termux '{name}'"])
        if cli:
            plt.savefig("img.png")
            plt.close(fig)
            subprocess.run("chafa img.png".split())
            time.sleep(0.5)
            subprocess.run("rm img.png".split())
        else:
            plt.show()


class LoadShotRecord:
    """
    A class to load and visualize shot records from SEGY/HDF5 files.
    """

    def __init__(self, path):
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
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a valid directory.")

        traces_path = os.path.join(path, "traces.segy")
        vel_path = os.path.join(path, "velocity_model.segy")
        v0_path = os.path.join(path, "smooth_velocity.segy")
        meta_path = os.path.join(path, "metadata.h5")

        if os.path.exists(traces_path):
            segy_data = SegyIO.read(traces_path)
            self.receivers = segy_data["receivers"]
            self.sources = segy_data["sources"]
            flat_data = segy_data["data"]
            n_traces, n_time = flat_data.shape

            unique_sources = np.unique(np.round(self.sources, 3), axis=0)
            n_sources = unique_sources.shape[0]
            n_receivers = n_traces // n_sources

            if n_sources * n_receivers == n_traces:
                self.shots = flat_data.reshape((n_sources, n_receivers, n_time))
            else:
                self.shots = flat_data

            self.nshots = n_sources

        if os.path.exists(vel_path):
            self.velocity_model, self.dx, self.dz = SegyIO.read_model(vel_path)
        if os.path.exists(v0_path):
            self.smooth_velocity, self.dx, self.dz = SegyIO.read_model(v0_path)

        if os.path.exists(meta_path):
            with h5py.File(meta_path, "r") as f:
                if "time" in f:
                    self.time = f["time"][()]
                if "wavelet" in f:
                    self.wavelet = f["wavelet"][()]
                if "f0" in f:
                    self.f0 = f["f0"][()]
                if "snr" in f:
                    self.snr = f["snr"][()]
                else:
                    self.snr = None

    def plot(self, **kwargs):
        shots_stack = np.hstack([shot.T for shot in self.shots]).T
        fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = seisplot.plot(shots_stack, fig=fig, ax=ax, linewidth=0.1, vaxis=self.time, hlabel="rec (m)", vlabel="Two-way travel time (s)", title="Shot gather", colorbar=True, **kwargs)
        plt.show()
        return plt.gca()


def _inject_headers_if_needed(filepath, dx, dz):
    need_injection = False
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        interval_val = f.bin[segyio.BinField.Interval]
        tracecount = len(f.trace)
        n2 = 801 if tracecount == 641601 else tracecount
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

        with open(filepath, "r+b") as f_raw:
            f_raw.seek(3200 + 117)
            f_raw.write(int(dz * 1000).to_bytes(2, byteorder="big"))


def load_marmousi(target_dx=None, target_dz=None):
    filepath = _find_assets_dir() / "vp_marmousi-ii.segy"
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
        mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        if dx_parsed <= 0:
            dx_parsed = 1.0
        if dz_parsed <= 0:
            dz_parsed = 1.0

    if target_dx is not None or target_dz is not None:
        from shotgen.models import resample_velocity_model
        seismic_data, nx_parsed, nz_parsed, dx_parsed, dz_parsed = resample_velocity_model(
            seismic_data, dx_parsed, dz_parsed, target_dx, target_dz
        )

    metadata = {"dx": dx_parsed, "dz": dz_parsed, "nx": nx_parsed, "nz": nz_parsed, "origin_x": 0.0, "origin_z": 0.0}
    return seismic_data, metadata


def load_sigsbee(reflection_coeffs=False, target_dx=None, target_dz=None):
    filepath = _find_assets_dir() / "sigsbee2a_stratigraphy.sgy"
    _inject_headers_if_needed(filepath, dx=1.0, dz=1.0)
    with open(filepath, "rb") as f_raw:
        f_raw.seek(3200 + 117)
        dz_bin = int.from_bytes(f_raw.read(2), byteorder="big") / 1000.0
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = np.array(f.trace.raw[:]) / 3.281
        dz_trace = f.header[0][segyio.TraceField.TRACE_SAMPLE_COUNT] / 1000.0
        dz_parsed = dz_bin if dz_bin > 0 else dz_trace
        nx_parsed = len(f.trace)
        nz_parsed = f.bin[segyio.BinField.Samples]
        scalar = f.header[0][segyio.TraceField.SourceGroupScalar]
        mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        if dx_parsed <= 0:
            dx_parsed = 1.0
        if dz_parsed <= 0:
            dz_parsed = 1.0

    if target_dx is not None or target_dz is not None:
        from shotgen.models import resample_velocity_model
        seismic_data, nx_parsed, nz_parsed, dx_parsed, dz_parsed = resample_velocity_model(
            seismic_data, dx_parsed, dz_parsed, target_dx, target_dz
        )

    metadata = {"dx": dx_parsed, "dz": dz_parsed, "nx": nx_parsed, "nz": nz_parsed, "origin_x": 0.0, "origin_z": 0.0}
    if reflection_coeffs:
        ref_path = _find_assets_dir() / "sigsbee2a_reflection_coefficients.sgy"
        _inject_headers_if_needed(ref_path, dx=1.0, dz=1.0)
        with segyio.open(ref_path, "r", ignore_geometry=True) as f_ref:
            ref_coeffs = np.array(f_ref.trace.raw[:]) / 3.281
        return (seismic_data, metadata), ref_coeffs
    return seismic_data, metadata


def load_complex_graben():
    filepath = _find_assets_dir() / "complex_graben.sgy"
    with segyio.open(filepath, "r", ignore_geometry=True) as f:
        seismic_data = f.trace.raw[:][:, ::-1]
    return seismic_data


def load_bpsalt(target_dx=None, target_dz=None):
    filepath = _find_assets_dir() / "vel_z6.25m_x12.5m_exact.segy"
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
        mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
        dx_parsed = (f.header[1][segyio.TraceField.GroupX] - f.header[0][segyio.TraceField.GroupX]) * mult
        if dx_parsed <= 0:
            dx_parsed = 12.5
        if dz_parsed <= 0:
            dz_parsed = 6.25

    if target_dx is not None or target_dz is not None:
        from shotgen.models import resample_velocity_model
        seismic_data, nx_parsed, nz_parsed, dx_parsed, dz_parsed = resample_velocity_model(
            seismic_data, dx_parsed, dz_parsed, target_dx, target_dz
        )

    metadata = {"dx": dx_parsed, "dz": dz_parsed, "nx": nx_parsed, "nz": nz_parsed, "origin_x": 0.0, "origin_z": 0.0}
    return seismic_data, metadata


def load_overthrust(target_dx=None, target_dz=None):
    filepath = _find_assets_dir() / "marine_overthrust_3d.segy"
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
        mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
        dx_parsed = (f.header[start_trace + 1][segyio.TraceField.GroupX] - f.header[start_trace][segyio.TraceField.GroupX]) * mult
        if dx_parsed <= 0:
            dx_parsed = 25.0
        if dz_parsed <= 0:
            dz_parsed = 25.0

    if target_dx is not None or target_dz is not None:
        from shotgen.models import resample_velocity_model
        seismic_data, nx_parsed, nz_parsed, dx_parsed, dz_parsed = resample_velocity_model(
            seismic_data, dx_parsed, dz_parsed, target_dx, target_dz
        )

    metadata = {"dx": dx_parsed, "dz": dz_parsed, "nx": nx_parsed, "nz": nz_parsed, "origin_x": 0.0, "origin_z": 0.0}
    return seismic_data, metadata

