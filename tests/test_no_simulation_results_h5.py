import numpy as np
import pytest
import shotgen.sampleshot as sampleshot_mod
import shotgen_gpu.cli as gpu_cli
from shotgen.sampleshot import LoadShotRecord, ShotRecord


def test_gpu_cli_does_not_write_simulation_results_h5(tmp_path, monkeypatch):
    """Verify shotgen-gpu CLI writes SEG-Y + metadata.h5 and omits simulation_results.h5."""
    n_sources, n_receivers, n_time = 2, 5, 26
    synthetic_shots = np.arange(n_sources * n_receivers * n_time, dtype=np.float32).reshape(
        (n_sources, n_receivers, n_time)
    )
    synthetic_wavelet = np.linspace(-1.0, 1.0, n_time, dtype=np.float32)

    class _DummySolver:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ms=1000.0, save_wavefield=False):
            return synthetic_shots.copy(), None, synthetic_wavelet.copy()

    monkeypatch.setattr(gpu_cli, "AcousticWaveSolverWrapper", _DummySolver)
    monkeypatch.setattr(gpu_cli, "configure_devito_device", lambda *args, **kwargs: "cpu")

    out_dir = tmp_path / "gpu_sim_out"
    vel_path = tmp_path / "vel.npy"
    np.save(vel_path, np.full((30, 20), 2000.0, dtype=np.float32))

    cfg = {
        "task": "simulation",
        "vel_file": str(vel_path),
        "dx": 10.0,
        "dz": 10.0,
        "nx": 30,
        "nz": 20,
        "n_sources": n_sources,
        "n_receivers": n_receivers,
        "ms": 50.0,
        "fs_ms": 2.0,
        "f0": 20.0,
        "group_offset": 10.0,
        "shot_offset": 20.0,
        "gather": "common shot",
        "smooth": 2.0,
        "device": "cpu",
        "output_dir": str(out_dir),
    }

    gpu_cli._run_simulation_cli(cfg)

    assert (out_dir / "traces.segy").exists()
    assert (out_dir / "velocity_model.segy").exists()
    assert (out_dir / "smooth_velocity.segy").exists()
    assert (out_dir / "metadata.h5").exists()
    assert not (out_dir / "simulation_results.h5").exists()

    loaded = LoadShotRecord(str(out_dir))
    assert loaded.shots.shape == (n_sources, n_receivers, n_time)
    assert np.allclose(loaded.shots, synthetic_shots)
    assert np.allclose(loaded.wavelet, synthetic_wavelet)


def test_shotrecord_devito_reads_traces_segy_and_metadata_h5(tmp_path, monkeypatch):
    """Verify ShotRecord(engine='devito').run() reads back traces.segy and metadata.h5 without simulation_results.h5."""
    n_sources, n_receivers, n_time = 2, 4, 26
    synthetic_shots = np.linspace(0.1, 10.0, n_sources * n_receivers * n_time, dtype=np.float32).reshape(
        (n_sources, n_receivers, n_time)
    )
    synthetic_wavelet = np.sin(np.linspace(0.0, np.pi, n_time, dtype=np.float32))

    class _DummySolver:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, ms=1000.0, save_wavefield=False):
            return synthetic_shots.copy(), None, synthetic_wavelet.copy()

    monkeypatch.setattr(gpu_cli, "AcousticWaveSolverWrapper", _DummySolver)
    monkeypatch.setattr(gpu_cli, "configure_devito_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(sampleshot_mod, "configure_devito_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(sampleshot_mod, "run_gpu_simulation_dict", lambda cfg: gpu_cli._run_simulation_cli(cfg))

    shot = ShotRecord(
        nx=30,
        nz=20,
        dx=10.0,
        dz=10.0,
        n_sources=n_sources,
        n_receivers=n_receivers,
        f0=20.0,
        engine="devito",
        device="cpu",
        fs_ms=2.0,
    )
    shot.set_model(np.full((30, 20), 2000.0, dtype=np.float32))
    res = shot.run(ms=50.0)

    assert res.shape == (n_sources, n_receivers, n_time)
    assert np.allclose(res, synthetic_shots)
    assert shot.src is not None
    assert np.allclose(shot.src, synthetic_wavelet)


def test_shotrecord_devito_raises_when_traces_segy_missing(monkeypatch):
    """Verify ShotRecord(engine='devito').run() raises a clear error if traces.segy is missing."""
    monkeypatch.setattr(sampleshot_mod, "configure_devito_device", lambda *args, **kwargs: "cpu")
    monkeypatch.setattr(sampleshot_mod, "run_gpu_simulation_dict", lambda cfg: None)

    shot = ShotRecord(
        nx=30,
        nz=20,
        dx=10.0,
        dz=10.0,
        n_sources=1,
        n_receivers=4,
        f0=20.0,
        engine="devito",
        device="cpu",
    )
    shot.set_model(np.full((30, 20), 2000.0, dtype=np.float32))

    with pytest.raises(RuntimeError, match="traces.segy"):
        shot.run(ms=50.0)
