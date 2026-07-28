import os
import numpy as np
import skfmm
from tqdm import tqdm
from joblib import Parallel, delayed
from shotgen.io import SegyIO


def load_dataset_dir(dataset_dir, require_f0=False, provided_f0=None):
    traces_path = os.path.join(dataset_dir, "traces.segy")
    v0_path = os.path.join(dataset_dir, "smooth_velocity.segy")
    vel_path = os.path.join(dataset_dir, "velocity_model.segy")
    meta_path = os.path.join(dataset_dir, "metadata.h5")

    if os.path.exists(v0_path):
        vp, dx, dz = SegyIO.read_model(v0_path)
    elif os.path.exists(vel_path):
        vp, dx, dz = SegyIO.read_model(vel_path)
    else:
        raise FileNotFoundError(f"smooth_velocity.segy not found (nor velocity_model.segy) in {dataset_dir}")

    segy_data = SegyIO.read(traces_path)
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

    if require_f0 and f0 is None:
        raise ValueError(f"f0 is required for RTM but was not found in {meta_path} and not provided as an argument.")

    return vp, sources, receivers, shots, time, f0, dx, dz


class KirchhoffModel:
    """Lightweight model container representing grid shape and origin for Kirchhoff migration."""
    def __init__(self, shape, origin=(0.0, 0.0)):
        self.shape = shape
        self.origin = origin


class KirchhoffMigration:
    """
    Kirchhoff pre-stack depth migration (PSDM) for 2D seismic data.
    Computes depth-migrated image using traveltimes from Fast Marching Method (FMM).
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

        self.vp = vp
        self.sources = sources
        self.receivers = receivers
        self.shots = shots
        self.time = time
        self.spacing = spacing
        self.origin = origin
        self.model = KirchhoffModel(shape=self.vp.shape, origin=self.origin)

        self._gather_unique_coords()
        self._setup_solver()

    def _gather_unique_coords(self):
        all_coords = []
        nx, nz = self.vp.shape
        for s in self.sources:
            sx = int(np.clip(np.round(s[0]), 0, nx - 1))
            sz = int(np.clip(np.round(s[1]), 0, nz - 1))
            all_coords.append((sx, sz))

        if self.receivers.ndim == 3:
            for i in range(self.receivers.shape[0]):
                for j in range(self.receivers.shape[1]):
                    rx = int(np.clip(np.round(self.receivers[i, j, 0]), 0, nx - 1))
                    rz = int(np.clip(np.round(self.receivers[i, j, 1]), 0, nz - 1))
                    all_coords.append((rx, rz))
        else:
            for r in self.receivers:
                rx = int(np.clip(np.round(r[0]), 0, nx - 1))
                rz = int(np.clip(np.round(r[1]), 0, nz - 1))
                all_coords.append((rx, rz))

        self.unique_coords = list(set(all_coords))

    def _setup_solver(self):
        computed_fields = Parallel(n_jobs=-1)(
            delayed(self.compute_single_traveltime_field)(c, self.vp, self.spacing[0], self.spacing[1])
            for c in self.unique_coords
        )
        self.traveltime_dict = {coord: field for coord, field in zip(self.unique_coords, computed_fields)}
        self.output = np.zeros_like(self.vp)

    def compute_single_traveltime_field(self, coord, vp, dx, dz):
        phi = np.ones_like(vp)
        idx_x = int(np.clip(np.round(coord[0]), 0, vp.shape[0] - 1))
        idx_z = int(np.clip(np.round(coord[1]), 0, vp.shape[1] - 1))
        phi[idx_x, idx_z] = 0
        return skfmm.travel_time(phi, vp, dx=[dx, dz])

    def run(self):
        nx, nz = self.vp.shape
        x_coords = self.origin[0] + np.arange(nx) * self.spacing[0]
        z_coords = self.origin[1] + np.arange(nz) * self.spacing[1]
        X, Z = np.meshgrid(x_coords, z_coords, indexing='ij')
        eps = 1e-4

        for si, source in enumerate(tqdm(self.sources, desc="Kirchhoff PSDM Source", total=len(self.sources))):
            sx_idx = int(np.clip(np.round(source[0]), 0, nx - 1))
            sz_idx = int(np.clip(np.round(source[1]), 0, nz - 1))
            sx = self.origin[0] + source[0] * self.spacing[0]
            sz = self.origin[1] + source[1] * self.spacing[1]

            Rs = np.sqrt((X - sx)**2 + (Z - sz)**2) + eps
            traveltime_s = self.traveltime_dict[(sx_idx, sz_idx)]

            n_receivers = self.receivers.shape[1] if self.receivers.ndim == 3 else self.receivers.shape[0]

            for ri in range(n_receivers):
                rec_coord = self.receivers[si, ri] if self.receivers.ndim == 3 else self.receivers[ri]
                rx_idx = int(np.clip(np.round(rec_coord[0]), 0, nx - 1))
                rz_idx = int(np.clip(np.round(rec_coord[1]), 0, nz - 1))
                traveltime_r = self.traveltime_dict[(rx_idx, rz_idx)]
                total_traveltime = traveltime_s + traveltime_r

                trace = self.shots[si, ri]

                rx = self.origin[0] + rec_coord[0] * self.spacing[0]
                rz = self.origin[1] + rec_coord[1] * self.spacing[1]

                Rr = np.sqrt((X - rx)**2 + (Z - rz)**2) + eps
                spreading = 1.0 / np.sqrt(Rs * Rr)
                obliquity = np.abs(Z - rz) / Rr
                weight = spreading * obliquity

                amplitudes = np.interp(total_traveltime.ravel(), self.time, trace, left=0.0, right=0.0).reshape(total_traveltime.shape)
                self.output += amplitudes * weight

        return self.output

