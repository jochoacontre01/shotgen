from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from devito import configuration, TimeFunction, Operator, Eq, solve, Function
from examples.seismic import demo_model, AcquisitionGeometry, PointSource, plot_image, Model, plot_velocity, plot_shotrecord
from examples.seismic.acoustic import AcousticWaveSolver
import torch
from deepwave import scalar_born
from shotgen.sampleshot import load_marmousi
from shotgen.sampleshot import GeoModel, LoadShot
from scipy.ndimage import gaussian_filter, laplace
from tqdm import tqdm
import scienceplots
import matplotlib
import sys
import pathlib
configuration["log-level"] = "WARNING"
plt.style.use(['science','no-latex'])
matplotlib.rcParams.update({"font.size":14})


class ReverseTimeMigration:
    """
    The `ReverseTimeMigration` class build on top of `Devito`, `PyLops` and `shotgen` to create an RTM image.
    
    It reads a shot record from `shotgen` as a HDF5 file and pass all the necessary parameters to `Devito`'s `Model` class to generate a forward model. `Devito` then performs the backward simulation and generates the image.
    
    For a full example on RTM using `Devito`, see [the example](https://www.devitoproject.org/examples/seismic/tutorials/02_rtm.html) in their
    """
    
    def __init__(
        self,
        vp:np.ndarray,
        n_sources:int,
        n_receivers:int,
        origin:tuple,
        spacing:tuple,
        nbl:int,
        t0:float,
        tn:float,
        f0:float,
        smooth_sigma:float,
        dtype=np.float32,
        space_order:int=4,
        time_order:int=2,
    ):
        
        self.vp = vp/1000
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
        self.f0 = f0/1000
        self.dtype = dtype
        
        self._create_model()
        
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
        self.geometry = AcquisitionGeometry(
            model=self.model,
            rec_positions=self.receivers,
            src_positions=src_coordinates,
            t0=self.t0,
            tn=self.tn,
            f0=self.f0,
            src_type="Ricker"
        )
    
    def _setup_solver(self):
        self.solver = AcousticWaveSolver(self.model, self.geometry, space_order=self.space_order, time_order=self.time_order)

    def run(self, save_wavefield=False, save_each=5):
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
        us = []
        vs = []
        image = Function(name="image", grid=self.model.grid)
        operator = self._imaging_operator(self.model, image)
        nshots = shot_records.shape[0]
        
        for i in tqdm(range(nshots), desc="source", total=nshots):
            self.geometry.src_positions[0, :] = self.sources[i, :]
            
            # true_d, _, _ = self.solver.forward(vp=self.model.vp)
            _, u0, _ = self.solver.forward(vp=self.model0.vp, save=True)
            
            if save_wavefield:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order, save=self.geometry.nt)
            else:
                v = TimeFunction(name="v", grid=self.model.grid, time_order=self.time_order, space_order=self.space_order)
            
            operator(u=u0, v=v, vp=self.model0.vp, dt=self.model0.critical_dt, residual=shot_records[i])
            
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
        residual = PointSource(name="residual", grid=model.grid, time_range=self.geometry.time_axis, coordinates=self.geometry.rec_positions)
        res_term = residual.inject(field=v.backward, expr=residual*dt**2/model.m)
        
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
        sources=None,
        receivers=None,
        wavelet=None,
        f0=None,
        time=None,
    ):
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