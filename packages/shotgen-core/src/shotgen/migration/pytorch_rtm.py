import os
import numpy as np
from tqdm import tqdm
from shotgen.migration.kirchhoff import load_dataset_dir


class ReverseTimeMigrationGPU:
    """
    PyTorch/Deepwave-based Reverse Time Migration (RTM).
    """

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
            velocity_model, sources, receivers, shot_record, time, f0, dx, dz = load_dataset_dir(dataset_dir, require_f0=True, provided_f0=f0)
            self.dx = dx
            self.dz = dz
        else:
            self.dx = 1.0
            self.dz = 1.0
            if dataset_dir is not None:
                import h5py
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
        self.nshots = self.shot_record.shape[0]
        self.nreceivers = len(self.receivers)

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.acquisition_params = self.set_acquisition_params()

    def set_acquisition_params(self):
        import torch
        nsource_per_shot = 1
        source_locations = torch.from_numpy(self.sources[:, np.newaxis, :]).long().to(self.device)
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
        import torch
        from deepwave import scalar_born
        if self.acquisition_params is None:
            raise ValueError("Acquisition params have not been set")

        scatter = torch.zeros_like(self.acquisition_params["init_velocity_model"])
        scatter.requires_grad_()
        optimizer = torch.optim.LBFGS([scatter])
        loss_fn = torch.nn.MSELoss()

        observed = torch.from_numpy(self.shot_record).float().to(self.device)
        for _ in tqdm(range(epochs), desc="Epoch", total=epochs):
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

        return scatter.detach().cpu().numpy()
