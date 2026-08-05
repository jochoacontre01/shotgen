import os
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from devito import TimeFunction, Operator, Eq, solve, Function
from examples.seismic import AcquisitionGeometry, PointSource, Model
from examples.seismic.acoustic import AcousticWaveSolver
from shotgen_gpu.config import configure_devito_device


def load_dataset_dir_gpu(dataset_dir, require_f0=False, provided_f0=None):
    """Helper to load dataset for Devito GPU RTM without importing shotgen-core."""
    import segyio
    import h5py

    traces_path = os.path.join(dataset_dir, "traces.segy")
    v0_path = os.path.join(dataset_dir, "smooth_velocity.segy")
    meta_path = os.path.join(dataset_dir, "metadata.h5")

    if os.path.exists(v0_path):
        with segyio.open(v0_path, "r", strict=False) as f:
            vp = segyio.tools.collect(f.trace)
            dz = f.bin[segyio.BinField.Interval] / 1000.0
            nx = len(f.trace)
            dx = (f.header[1][segyio.TraceField.CDP_X] - f.header[0][segyio.TraceField.CDP_X]) / 1000.0 if nx > 1 else 1.0
            if dx <= 0.0: dx = 1.0
            if dz <= 0.0: dz = 1.0
    else:
        raise FileNotFoundError(f"smooth_velocity.segy not found in {dataset_dir}")

    with segyio.open(traces_path, "r", strict=False) as f:
        n_traces = f.tracecount
        flat_data = segyio.tools.collect(f.trace)
        sources = np.zeros((n_traces, 2))
        receivers = np.zeros((n_traces, 2))
        for i in range(n_traces):
            header = f.header[i]
            scalar = header[segyio.TraceField.SourceGroupScalar]
            mult = 1.0 / abs(scalar) if scalar < 0 else (scalar if scalar > 0 else 1.0)
            sources[i, 0] = header[segyio.TraceField.SourceX] * mult
            sources[i, 1] = header[segyio.TraceField.SourceY] * mult
            receivers[i, 0] = header[segyio.TraceField.GroupX] * mult
            receivers[i, 1] = header[segyio.TraceField.GroupY] * mult
        time = f.samples / 1000.0

    n_traces, n_time = flat_data.shape
    unique_src = np.unique(np.round(sources, 3), axis=0)
    n_sources = unique_src.shape[0]
    n_receivers = n_traces // n_sources

    shots = flat_data.reshape((n_sources, n_receivers, n_time))
    sources = sources.reshape((n_sources, n_receivers, 2))[:, 0, :]
    receivers = receivers.reshape((n_sources, n_receivers, 2))

    f0 = provided_f0
    if os.path.exists(meta_path) and f0 is None:
        with h5py.File(meta_path, "r") as f:
            if "f0" in f:
                f0 = f["f0"][()]

    if require_f0 and f0 is None:
        raise ValueError(f"f0 is required for RTM but was not found in {meta_path} and not provided as an argument.")

    return vp, sources, receivers, shots, time, f0, dx, dz


class ReverseTimeMigration:
    """
    Reverse Time Migration (RTM) implementation using Devito JIT code generation.
    Supports OpenACC acceleration on NVIDIA GPUs.
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
        space_order: int = 8,
        time_order: int = 2,
        dtype=np.float32,
        device="auto",
        verbose: bool = False,
        **kwargs,
    ):
        self.verbose = verbose
        configure_devito_device(device, verbose=verbose)
        if dataset_dir is not None:
            vp, sources, receivers, shots, time, f0, dx, dz = load_dataset_dir_gpu(dataset_dir, require_f0=True, provided_f0=f0)
            spacing = (dx, dz)
            self.from_data = True
        elif sources is not None:
            self.from_data = True
        else:
            self.from_data = False
            if f0 is None:
                f0 = 25.0

        if self.from_data:
            origin = kwargs.get("origin")
            if origin is None and dataset_dir is not None:
                meta_path = os.path.join(dataset_dir, "metadata.h5")
                if os.path.exists(meta_path):
                    import h5py
                    with h5py.File(meta_path, "r") as f:
                        if "origin" in f:
                            origin = tuple(f["origin"][()])
            if origin is None:
                origin = (0.0, 0.0)

            n_sources = sources.shape[0]
            n_receivers = receivers.shape[1] if receivers.ndim == 3 else receivers.shape[0]

            t0 = time[0] * 1000.0 if time[0] <= 10.0 else time[0]
            tn = time[-1] * 1000.0 if time[-1] <= 10.0 else time[-1]
        else:
            n_sources = args[0] if len(args) > 0 else kwargs.get("n_sources")
            n_receivers = args[1] if len(args) > 1 else kwargs.get("n_receivers")
            origin = args[2] if len(args) > 2 else kwargs.get("origin")

            if len(args) > 3: spacing = args[3]
            if len(args) > 4: nbl = args[4]
            t0 = args[5] if len(args) > 5 else kwargs.get("t0")
            tn = args[6] if len(args) > 6 else kwargs.get("tn")
            if len(args) > 7: f0 = args[7]
            if len(args) > 8: smooth_sigma = args[8]
            if len(args) > 9: dtype = args[9]
            if len(args) > 10: space_order = args[10]
            if len(args) > 11: time_order = args[11]

        if dataset_dir is not None:
            self.v0 = vp / 1000.0
            self.vp = self.v0.copy()
        else:
            self.vp = vp / 1000.0
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
        self.f0 = f0 / 1000.0 if (f0 is not None and f0 > 1.0) else f0
        self.dtype = dtype

        self._create_model()

        if self.from_data:
            self.sources = sources
            self.receivers = receivers
            self.shots = np.transpose(shots, (0, 2, 1)).astype(self.dtype)
            self._create_geometry()
            dt_data = (time[1] - time[0]) * 1000.0 if (time[1] - time[0]) <= 1.0 else (time[1] - time[0])
            self.geometry.resample(dt_data)

            nt_target = self.geometry.nt
            nt_actual = self.shots.shape[1]
            if nt_actual != nt_target:
                if nt_actual < nt_target:
                    padding = np.zeros((self.shots.shape[0], nt_target - nt_actual, self.shots.shape[2]), dtype=self.shots.dtype)
                    self.shots = np.concatenate([self.shots, padding], axis=1)
                else:
                    self.shots = self.shots[:, :nt_target, :]
        else:
            source_locations = np.empty((self.n_sources, 2), dtype=self.dtype)
            source_locations[:, 0] = self.origin[0] + np.linspace(0, self.model.domain_size[0], num=self.n_sources)
            source_locations[:, 1] = self.origin[1] + 0.
            self.sources = source_locations

            rec_locations = np.empty((self.n_receivers, 2))
            rec_locations[:, 0] = self.origin[0] + np.linspace(0, self.model.domain_size[0], num=self.n_receivers)
            rec_locations[:, 1] = self.origin[1] + 0.
            self.receivers = rec_locations

            self._create_geometry()

        self._setup_solver()

    def _create_model(self):
        self.model = Model(
            vp=self.vp, origin=self.origin, shape=self.vp.shape,
            spacing=self.spacing, space_order=self.space_order,
            nbl=self.nbl, bcs="damp", dtype=self.dtype, grid=None
        )
        self.model0 = Model(
            vp=self.v0, origin=self.origin, shape=self.vp.shape,
            spacing=self.spacing, space_order=self.space_order,
            nbl=self.nbl, bcs="damp", dtype=self.dtype, grid=None
        )

    def _create_geometry(self):
        src_coordinates = np.empty((1, 2))
        src_coordinates[0, 0] = self.origin[0] + self.model.domain_size[0] * 0.5
        src_coordinates[0, 1] = self.origin[1] + 0.0
        rec_positions = self.receivers[0] if (self.from_data and self.receivers.ndim == 3) else self.receivers
        self.geometry = AcquisitionGeometry(
            model=self.model, rec_positions=rec_positions,
            src_positions=src_coordinates, t0=self.t0, tn=self.tn,
            f0=self.f0, src_type="Ricker"
        )

    def _setup_solver(self):
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=self.space_order, time_order=self.time_order)

    def run(self, save_wavefield=False, save_each=5):
        if self.from_data:
            us, vs = [], []
            image = Function(name="image", grid=self.model.grid)
            operator = self._imaging_operator(self.model, image, save_wavefield=save_wavefield)

            for i in tqdm(range(self.n_sources), desc="Source", total=self.n_sources):
                self.geometry.src_positions[0, :] = self.sources[i, :]
                self.geometry.src.coordinates.data[0, :] = self.sources[i, :]

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
            shots, us, vs = [], [], []
            image = Function(name="image", grid=self.model.grid)
            operator = self._imaging_operator(self.model, image, save_wavefield=save_wavefield)

            for i in tqdm(range(self.n_sources), desc="Source", total=self.n_sources):
                self.geometry.src_positions[0, :] = self.sources[i, :]
                self.geometry.src.coordinates.data[0, :] = self.sources[i, :]

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

    def _imaging_operator(self, model, image, save_wavefield=False):
        if save_wavefield:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        else:
            v = TimeFunction(name="v", grid=model.grid, time_order=self.time_order, space_order=self.space_order)

        u = TimeFunction(name="u", grid=model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
        eqn = model.m * v.dt2 - v.laplace + model.damp * v.dt
        stencil = Eq(v.backward, solve(eqn, v.backward))
        dt = model.grid.stepping_dim.spacing

        self.residual_source = PointSource(name="residual", grid=model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        res_term = self.residual_source.inject(field=v.backward, expr=self.residual_source*dt**2/model.m)
        image_update = Eq(image, image+u*v)

        return Operator([stencil] + res_term + [image_update], subs=model.spacing_map)

    @property
    def src(self):
        return self.geometry.src.wavelet
