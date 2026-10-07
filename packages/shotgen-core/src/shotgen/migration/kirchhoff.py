import os
import logging
import warnings
from time import sleep
import multiprocessing as mp
from multiprocessing import get_context, current_process
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import skfmm
from tqdm import tqdm
from shotgen.io import SegyIO
from shotgen.pylogger import get_logger, log_execution_time
from scipy.ndimage import gaussian_filter
import subprocess
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning, module="multiprocessing")

logger = get_logger("shotgen.migration.kirchhoff")


def load_dataset_dir(dataset_dir, require_f0=False, provided_f0=None, verbose=True):
    with log_execution_time(logger, f"Load dataset SEGY files from '{dataset_dir}'", level=logging.INFO if verbose else logging.DEBUG):
        if verbose:
            logger.info(f"Starting SEGY dataset loading phase for: '{dataset_dir}'")

        traces_path = os.path.join(dataset_dir, "traces.segy")
        v0_path = os.path.join(dataset_dir, "smooth_velocity.segy")
        vel_path = os.path.join(dataset_dir, "velocity_model.segy")
        meta_path = os.path.join(dataset_dir, "metadata.h5")

        if os.path.exists(v0_path):
            logger.debug(f"Reading smooth velocity model from: {v0_path}")
            vp, dx, dz = SegyIO.read_model(v0_path, verbose=verbose)
        elif os.path.exists(vel_path):
            logger.debug(f"Reading true velocity model from: {vel_path}")
            vp, dx, dz = SegyIO.read_model(vel_path, verbose=verbose)
        else:
            raise FileNotFoundError(f"smooth_velocity.segy not found (nor velocity_model.segy) in {dataset_dir}")

        segy_data = SegyIO.read(traces_path, verbose=verbose)
        receivers = segy_data["receivers"]
        sources = segy_data["sources"]
        time = segy_data["time"]

        flat_data = segy_data["data"]
        n_traces, n_time = flat_data.shape

        unique_src = np.unique(np.round(sources, 3), axis=0)
        n_sources = unique_src.shape[0]
        n_receivers = n_traces // n_sources

        shots = flat_data.reshape((n_sources, n_receivers, n_time))
        sources = sources.reshape((n_sources, n_receivers, 2))[:, 0, :]
        receivers = receivers.reshape((n_sources, n_receivers, 2))

        f0 = provided_f0
        if os.path.exists(meta_path) and f0 is None:
            import h5py
            with h5py.File(meta_path, "r") as f:
                if "f0" in f:
                    f0 = f["f0"][()]
                    logger.debug(f"Loaded peak frequency f0={f0} Hz from {meta_path}")

        if require_f0 and f0 is None:
            raise ValueError(f"f0 is required for RTM but was not found in {meta_path} and not provided as an argument.")

        if verbose:
            logger.info(f"Loaded dataset '{dataset_dir}': {n_sources} shots x {n_receivers} receivers ({n_traces:,} traces) | Velocity grid {vp.shape[0]}x{vp.shape[1]}")

        return vp, sources, receivers, shots, time, f0, dx, dz


def _compute_single_traveltime_task(args):
    """Computes an Eikonal traveltime field using Fast Marching Method in an isolated process worker."""
    coord, vp, dx, dz, dtype = args
    phi = np.ones_like(vp)
    idx_x = int(np.clip(np.round(coord[0]), 0, vp.shape[0] - 1))
    idx_z = int(np.clip(np.round(coord[1]), 0, vp.shape[1] - 1))
    phi[idx_x, idx_z] = 0
    field = skfmm.travel_time(phi, vp, dx=[dx, dz]).astype(dtype)
    return coord, field


def _process_single_shot(
    si: int,
    source: np.ndarray,
    receivers: np.ndarray,
    shot_data: np.ndarray,
    vp_shape: tuple,
    origin: tuple,
    spacing: tuple,
    time: np.ndarray,
    traveltime_dict: dict,
    X: np.ndarray = None,
    Z: np.ndarray = None,
    dtype: type = np.float32,
) -> np.ndarray:
    """Computes depth migration image contribution for a single shot."""
    nx, nz = vp_shape
    eps = np.float32(1e-4) if dtype == np.float32 else 1e-4

    if X is None or Z is None:
        x_coords = (origin[0] + np.arange(nx) * spacing[0]).astype(dtype)
        z_coords = (origin[1] + np.arange(nz) * spacing[1]).astype(dtype)
        X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')

    shot_output = np.zeros(vp_shape, dtype=dtype)

    sx_idx = int(np.clip(np.round(source[0]), 0, nx - 1))
    sz_idx = int(np.clip(np.round(source[1]), 0, nz - 1))
    sx = (origin[0] + source[0] * spacing[0]).astype(dtype) if hasattr(source[0], 'astype') else dtype(origin[0] + source[0] * spacing[0])
    sz = (origin[1] + source[1] * spacing[1]).astype(dtype) if hasattr(source[1], 'astype') else dtype(origin[1] + source[1] * spacing[1])

    Rs = np.sqrt((X - sx)**2 + (Z - sz)**2).astype(dtype) + eps
    traveltime_s = traveltime_dict[(sx_idx, sz_idx)]

    n_receivers = receivers.shape[1] if receivers.ndim == 3 else receivers.shape[0]

    for ri in range(n_receivers):
        rec_coord = receivers[si, ri] if receivers.ndim == 3 else receivers[ri]
        rx_idx = int(np.clip(np.round(rec_coord[0]), 0, nx - 1))
        rz_idx = int(np.clip(np.round(rec_coord[1]), 0, nz - 1))
        traveltime_r = traveltime_dict[(rx_idx, rz_idx)]
        total_traveltime = traveltime_s + traveltime_r

        trace = shot_data[ri]

        rx = dtype(origin[0] + rec_coord[0] * spacing[0])
        rz = dtype(origin[1] + rec_coord[1] * spacing[1])

        Rr = np.sqrt((X - rx)**2 + (Z - rz)**2).astype(dtype) + eps
        spreading = 1.0 / np.sqrt(Rs * Rr)
        obliquity = np.abs(Z - rz) / Rr
        weight = spreading * obliquity

        amplitudes = np.interp(total_traveltime.ravel(), time, trace, left=0.0, right=0.0).reshape(total_traveltime.shape).astype(dtype)
        shot_output += amplitudes * weight

    return shot_output


class KirchhoffModel:
    """Lightweight model container representing grid shape and origin for Kirchhoff migration."""
    def __init__(self, shape, origin=(0.0, 0.0)):
        self.shape = shape
        self.origin = origin


class KirchhoffMigration:
    """
    Kirchhoff pre-stack depth migration (PSDM) for 2D seismic data.
    Computes depth-migrated image using traveltimes from Fast Marching Method (FMM).
    Optimized for low RAM footprint via active-window lifecycle caching and float32 precision.
    """

    def __init__(
        self,
        vp: np.ndarray = None,
        sources: np.ndarray = None,
        receivers: np.ndarray = None,
        shots: np.ndarray = None,
        time: np.ndarray = None,
        spacing: tuple = (1.0, 1.0),
        dataset_dir: str = None,
        n_jobs: int = -1,
        backend: str = "threading",
        smooth_sigma: float = None,
        cli: bool = False,
        batch_size: int = None,
        dtype: type = np.float32,
    ):
        origin = (0.0, 0.0)
        if dataset_dir is not None:
            vp, sources, receivers, shots, time, _, dx, dz = load_dataset_dir(dataset_dir, require_f0=False)
            spacing = (dx, dz)

            meta_path = os.path.join(dataset_dir, "metadata.h5")
            if os.path.exists(meta_path):
                import h5py
                with h5py.File(meta_path, "r") as f:
                    if "origin" in f:
                        origin = tuple(f["origin"][()])

            sources = sources.copy()
            sources[..., 0] = (sources[..., 0] - origin[0]) / dx
            sources[..., 1] = (sources[..., 1] - origin[1]) / dz
            receivers = receivers.copy()
            receivers[..., 0] = (receivers[..., 0] - origin[0]) / dx
            receivers[..., 1] = (receivers[..., 1] - origin[1]) / dz

        if smooth_sigma is not None:
            self.vp = gaussian_filter(vp, sigma=smooth_sigma).astype(dtype)
        else:
            self.vp = vp.astype(dtype) if vp is not None else None

        self.sources = sources
        self.receivers = receivers
        self.shots = shots.astype(dtype) if shots is not None else None
        self.time = time.astype(dtype) if time is not None else None
        self.spacing = spacing
        self.origin = origin
        self.n_jobs = n_jobs
        self.backend = backend
        self.cli = cli
        self.batch_size = batch_size
        self.dtype = dtype
        self.traveltime_dict = {}

        if self.vp is not None:
            self.model = KirchhoffModel(shape=self.vp.shape, origin=self.origin)
            self._gather_unique_coords()

        if self.cli and self.vp is not None and current_process().name == 'MainProcess':
            plt.figure(figsize=(10, 5))
            plt.imshow(self.vp.T, cmap="gray", aspect="auto")
            plt.colorbar()
            plt.tight_layout()
            temp_img = "/tmp/kirchhoff_model.png"
            plt.savefig(temp_img)
            plt.close()
            subprocess.run(f"chafa {temp_img}".split())
            sleep(0.25)
            try:
                os.remove(temp_img)
            except Exception:
                pass

    def _gather_unique_coords(self):
        """Analyzes acquisition geometry and schedules coordinate lifecycle per shot."""
        nx, nz = self.vp.shape
        n_sources = len(self.sources)

        self.shot_coords = []
        self.coord_first = {}
        self.coord_last = {}
        all_coords = []

        for si in range(n_sources):
            cur_coords = []
            s = self.sources[si]
            sx = int(np.clip(np.round(s[0]), 0, nx - 1))
            sz = int(np.clip(np.round(s[1]), 0, nz - 1))
            cur_coords.append((sx, sz))
            all_coords.append((sx, sz))

            if self.receivers.ndim == 3:
                for ri in range(self.receivers.shape[1]):
                    rx = int(np.clip(np.round(self.receivers[si, ri, 0]), 0, nx - 1))
                    rz = int(np.clip(np.round(self.receivers[si, ri, 1]), 0, nz - 1))
                    cur_coords.append((rx, rz))
                    all_coords.append((rx, rz))
            else:
                for r in self.receivers:
                    rx = int(np.clip(np.round(r[0]), 0, nx - 1))
                    rz = int(np.clip(np.round(r[1]), 0, nz - 1))
                    cur_coords.append((rx, rz))
                    all_coords.append((rx, rz))

            for c in cur_coords:
                if c not in self.coord_first:
                    self.coord_first[c] = si
                self.coord_last[c] = si

            self.shot_coords.append(cur_coords)

        self.unique_coords = list(set(all_coords))
        logger.debug(f"Collected {len(self.unique_coords)} unique coordinate points across {n_sources} shots.")

    def compute_single_traveltime_field(self, coord, vp, dx, dz):
        phi = np.ones_like(vp)
        idx_x = int(np.clip(np.round(coord[0]), 0, vp.shape[0] - 1))
        idx_z = int(np.clip(np.round(coord[1]), 0, vp.shape[1] - 1))
        phi[idx_x, idx_z] = 0
        return skfmm.travel_time(phi, vp, dx=[dx, dz]).astype(self.dtype)

    def run(self, n_jobs: int = None, backend: str = None, show_progress: bool = True, batch_size: int = None):
        """
        Executes Kirchhoff PSDM using batch-of-shots parallel migration and active-window caching.
        Uses multiprocessing (fork) for GIL-free parallel Eikonal traveltimes, ThreadPoolExecutor for shot migration,
        and pinned transient progress bars (position=1, leave=False) for clean terminal feedback.
        """
        effective_n_jobs = self.n_jobs if n_jobs is None else n_jobs
        if effective_n_jobs is None or effective_n_jobs < 0:
            effective_n_jobs = os.cpu_count() or 4

        effective_batch_size = self.batch_size if batch_size is None else batch_size
        if effective_batch_size is None or effective_batch_size < 1:
            effective_batch_size = 8

        # Cap worker count to maximize throughput while bounding memory allocations
        eikonal_workers = max(1, min(8, effective_n_jobs))
        shot_workers = max(1, min(effective_batch_size, effective_n_jobs))

        nx, nz = self.vp.shape
        x_coords = (self.origin[0] + np.arange(nx) * self.spacing[0]).astype(self.dtype)
        z_coords = (self.origin[1] + np.arange(nz) * self.spacing[1]).astype(self.dtype)
        X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')

        n_sources = len(self.sources)
        self.output = np.zeros(self.vp.shape, dtype=self.dtype)
        batch_starts = list(range(0, n_sources, effective_batch_size))
        total_batches = len(batch_starts)

        ctx = get_context("fork")

        with ctx.Pool(processes=eikonal_workers) as eikonal_pool:
            with log_execution_time(logger, f"Kirchhoff PSDM ({n_sources} shots, batch_size={effective_batch_size}, eikonal_workers={eikonal_workers}, shot_workers={shot_workers})"):
                batch_iterator = enumerate(batch_starts)
                pbar_batches = tqdm(
                    batch_iterator,
                    desc="Kirchhoff PSDM Batches",
                    total=total_batches,
                    position=0,
                    leave=True,
                ) if show_progress else batch_iterator

                for b_idx, b_start in pbar_batches:
                    b_end = min(b_start + effective_batch_size, n_sources)
                    batch_shots = list(range(b_start, b_end))

                    # Identify all unique coordinates needed for this batch
                    batch_needed = set()
                    for si in batch_shots:
                        batch_needed.update(self.shot_coords[si])

                    missing_coords = [c for c in batch_needed if c not in self.traveltime_dict]
                    if missing_coords:
                        eikonal_tasks = [
                            (c, self.vp, self.spacing[0], self.spacing[1], self.dtype)
                            for c in missing_coords
                        ]
                        pbar_eikonal = tqdm(
                            total=len(missing_coords),
                            desc=f"  [Batch {b_idx + 1}/{total_batches}] Eikonal fields",
                            position=1,
                            leave=False,
                            ncols=85,
                        ) if show_progress else None

                        for c, field in eikonal_pool.imap_unordered(_compute_single_traveltime_task, eikonal_tasks):
                            self.traveltime_dict[c] = field
                            if pbar_eikonal is not None:
                                pbar_eikonal.update(1)

                        if pbar_eikonal is not None:
                            pbar_eikonal.close()

                    # Process batch of shots
                    if len(batch_shots) == 1:
                        si = batch_shots[0]
                        shot_img = _process_single_shot(
                            si, self.sources[si], self.receivers, self.shots[si],
                            self.vp.shape, self.origin, self.spacing, self.time,
                            self.traveltime_dict, X, Z, dtype=self.dtype
                        )
                        self.output += shot_img
                    else:
                        pbar_shots = tqdm(
                            total=len(batch_shots),
                            desc=f"  [Batch {b_idx + 1}/{total_batches}] Migrating shots",
                            position=1,
                            leave=False,
                            ncols=85,
                        ) if show_progress else None

                        with ThreadPoolExecutor(max_workers=shot_workers) as executor:
                            futures = [
                                executor.submit(
                                    _process_single_shot,
                                    si, self.sources[si], self.receivers, self.shots[si],
                                    self.vp.shape, self.origin, self.spacing, self.time,
                                    self.traveltime_dict, X, Z, self.dtype
                                )
                                for si in batch_shots
                            ]
                            for f in as_completed(futures):
                                self.output += f.result()
                                if pbar_shots is not None:
                                    pbar_shots.update(1)

                        if pbar_shots is not None:
                            pbar_shots.close()

                    # Evict coordinates whose last required shot is in or before this batch
                    expired_coords = [c for c in list(self.traveltime_dict.keys()) if self.coord_last.get(c, -1) < b_end]
                    for c in expired_coords:
                        del self.traveltime_dict[c]

            return self.output
