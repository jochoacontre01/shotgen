import os
import gc
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from shotgen_gpu.config import configure_devito_device

os.environ["DEVITO_LOGGING"] = "WARNING"
try:
    from devito import configuration
    configuration["log-level"] = "WARNING"
except ImportError:
    pass


def _get_pylops():
    """
    Lazy import helper for PyLops with PyTorch import isolation
    to prevent symbol interposition or OpenACC conflict with nvc++.
    """
    import sys
    import types
    import importlib.machinery

    if "torch" not in sys.modules:
        class DummyTensor:
            pass
        mock_torch = types.ModuleType("torch")
        mock_torch.Tensor = DummyTensor
        mock_torch.__spec__ = importlib.machinery.ModuleSpec("torch", None)
        sys.modules["torch"] = mock_torch

        dummy_torch_op = types.ModuleType("pylops.torchoperator")
        dummy_torch_op.__all__ = []
        sys.modules["pylops.torchoperator"] = dummy_torch_op

    import pylops
    from pylops.waveeqprocessing import AcousticWave2D
    return pylops, AcousticWave2D


class AcousticWaveSolverWrapper:
    """
    Wrapper around PyLops AcousticWave2D for forward acoustic wavefield modeling.
    Enforces GPU device affinity and OpenACC flags via shotgen_gpu.config.
    Devito is reserved exclusively for Reverse Time Migration (RTM).
    """

    def __init__(
        self,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        vel: np.ndarray,
        sources: np.ndarray,
        receivers: np.ndarray,
        f0: float = 25.0,
        origin: tuple = (0.0, 0.0),
        fd_order: int = 4,
        n_damping: int = 100,
        smooth: float = 5.0,
        device: str = "auto",
        float_type=np.float32,
        verbose: bool = False,
    ):
        self.verbose = verbose
        self.device = configure_devito_device(device, verbose=verbose)
        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.vel = vel
        self.sources = sources
        self.receivers = receivers
        self.f0 = f0
        self.origin = origin
        self.fd_order = fd_order
        self.n_damping = n_damping
        self.smooth = smooth
        self.float_type = float_type

        self.n_sources = len(sources)
        self.n_receivers = receivers.shape[1] if receivers.ndim == 3 else len(receivers)

    def run(self, ms: float = 500.0, save_wavefield: bool = False, save_each: int = 20):
        pylops, AcousticWave2D = _get_pylops()

        v0 = gaussian_filter(self.vel, sigma=self.smooth)
        dv = (self.vel**(-2) - v0**(-2)).astype(self.float_type)

        shots = []
        us = []
        wavelet = None

        for i in tqdm(range(self.n_sources), desc="PyLops GPU Source", total=self.n_sources):
            s = self.sources[i]
            rec_positions = self.receivers[i] if self.receivers.ndim == 3 else self.receivers
            rec_x = rec_positions[:, 0]
            rec_z = rec_positions[:, 1]

            aop = AcousticWave2D(
                shape=(self.nx, self.nz),
                origin=self.origin,
                spacing=(self.dx, self.dz),
                vp=v0,
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
                dtype=str(np.dtype(self.float_type)),
            )

            dobs = aop @ dv
            dobs_shot = dobs.reshape(self.n_receivers, -1)
            shots.append(dobs_shot)

            if save_wavefield:
                aop.srcillumination_allshots(savewav=True)
                if hasattr(aop, "src_wavefield") and len(aop.src_wavefield) > 0:
                    u0_data = aop.src_wavefield[0].data.copy()
                    us.append(u0_data[::save_each])

            if wavelet is None and hasattr(aop.geometry, "src"):
                wavelet = aop.geometry.src.wavelet

            del aop, dobs
            gc.collect()

        shot_run = np.array(shots, dtype=self.float_type)
        us_arr = np.array(us, dtype=self.float_type) if save_wavefield and len(us) > 0 else None
        return shot_run, us_arr, wavelet
