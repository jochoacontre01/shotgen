import pytest
import numpy as np
from shotgen_gpu.engine.rtm import ReverseTimeMigration


def test_rtm_time_weights_default():
    """Verify that when no t_start and t_end are provided, all weights remain 1.0 (backward compatible)."""
    # Create minimal RTM instance
    vp = 1500.0 * np.ones((20, 20), dtype=np.float32)
    rtm = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=2,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=1000.0,
        f0=20.0,
        space_order=2,
        device="cpu",
    )
    weights = rtm.compute_time_weights()
    assert len(weights) == rtm.geometry.nt
    assert np.all(weights == 1.0)


def test_rtm_time_weights_windowing_and_tapers():
    """Verify time windowing math with cosine tapers."""
    vp = 1500.0 * np.ones((20, 20), dtype=np.float32)
    rtm = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=2,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=1000.0,
        f0=20.0,
        space_order=2,
        device="cpu",
    )
    # Window between 200 and 800 ms with 50 ms taper
    weights = rtm.compute_time_weights(t_start=200.0, t_end=800.0, t_taper=50.0)
    t_vals = np.linspace(rtm.t0, rtm.tn, rtm.geometry.nt)

    # Below 200 ms: zero
    assert np.all(weights[t_vals < 200.0] == 0.0)
    # Above 800 ms: zero
    assert np.all(weights[t_vals > 800.0] == 0.0)
    # Plateau between 250 and 750 ms: exactly 1.0
    plateau = weights[(t_vals >= 250.0) & (t_vals <= 750.0)]
    assert np.allclose(plateau, 1.0)
    # Midpoint of ramps should be ~0.5
    idx_mid_up = np.argmin(np.abs(t_vals - 225.0))
    idx_mid_down = np.argmin(np.abs(t_vals - 775.0))
    assert np.isclose(weights[idx_mid_up], 0.5, atol=0.05)
    assert np.isclose(weights[idx_mid_down], 0.5, atol=0.05)


def test_rtm_time_weights_boxcar():
    """Verify boxcar behavior when t_taper=0."""
    vp = 1500.0 * np.ones((20, 20), dtype=np.float32)
    rtm = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=2,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=500.0,
        f0=20.0,
        space_order=2,
        device="cpu",
    )
    weights = rtm.compute_time_weights(t_start=100.0, t_end=400.0, t_taper=0.0)
    t_vals = np.linspace(rtm.t0, rtm.tn, rtm.geometry.nt)
    assert np.all(weights[t_vals < 100.0] == 0.0)
    assert np.all(weights[(t_vals >= 100.0) & (t_vals <= 400.0)] == 1.0)
    assert np.all(weights[t_vals > 400.0] == 0.0)


def test_rtm_time_weights_validation():
    """Verify error checking on invalid input bounds."""
    vp = 1500.0 * np.ones((20, 20), dtype=np.float32)
    rtm = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=2,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=500.0,
        f0=20.0,
        space_order=2,
        device="cpu",
    )
    # t_start >= t_end
    with pytest.raises(ValueError, match="must be strictly less than"):
        rtm.compute_time_weights(t_start=400.0, t_end=200.0)

    # negative taper
    with pytest.raises(ValueError, match="cannot be negative"):
        rtm.compute_time_weights(t_start=100.0, t_end=400.0, t_taper=-10.0)


def test_rtm_migration_with_time_window():
    """Run synthetic RTM with time windowing and verify image validity and surface artifact suppression."""
    nbl = 10
    vp = 1500.0 * np.ones((30, 30), dtype=np.float32)
    vp[:, 15:] = 2500.0

    # 1. Full migration without windowing
    rtm_full = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=4,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=nbl,
        t0=0.0,
        tn=200.0,
        f0=25.0,
        space_order=2,
        device="cpu",
    )
    img_full = rtm_full.run(save_wavefield=False)

    # 2. Windowed migration starting at 60 ms (cutting off early direct arrivals)
    rtm_win = ReverseTimeMigration(
        vp=vp,
        n_sources=1,
        n_receivers=4,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        nbl=nbl,
        t0=0.0,
        tn=200.0,
        f0=25.0,
        space_order=2,
        device="cpu",
        t_start=60.0,
        t_taper=15.0,
    )
    img_win = rtm_win.run(save_wavefield=False)

    assert img_win is not None
    assert img_win.shape == img_full.shape
    assert np.any(img_win != 0.0)
    assert not np.all(np.isnan(img_win))

    # Verify that the two images differ (windowing altered wavefield cross-correlations)
    assert not np.allclose(img_win, img_full)

    # Compare energy at the very top of the model (first few depth layers inside physical domain)
    top_energy_full = np.mean(np.abs(img_full[nbl:nbl + 3, nbl:-nbl]))
    top_energy_win = np.mean(np.abs(img_win[nbl:nbl + 3, nbl:-nbl]))
    assert top_energy_win < top_energy_full


def test_rtm_cli_flags(monkeypatch, tmp_path):
    """Test that CLI flags --t-start, --t-end, and --t-taper correctly pass into ReverseTimeMigration."""
    from shotgen_gpu.cli import main
    import json
    import sys

    # Write a dummy config
    cfg = {
        "task": "rtm",
        "dataset_dir": str(tmp_path),
        "t_start": 50.0,
        "t_end": 500.0,
    }
    cfg_file = tmp_path / "rtm_config.json"
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    captured_kwargs = {}

    class MockRTM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def run(self, **kwargs):
            return np.zeros((10, 10))

    monkeypatch.setattr("shotgen_gpu.cli.ReverseTimeMigration", MockRTM)

    # 1. Run with CLI flags overriding config
    test_args = ["shotgen-gpu", "-c", str(cfg_file), "--t-start", "150.0", "--t-end", "400.0", "--t-taper", "25.0"]
    monkeypatch.setattr(sys, "argv", test_args)
    main()

    assert captured_kwargs["t_start"] == 150.0
    assert captured_kwargs["t_end"] == 400.0
    assert captured_kwargs["t_taper"] == 25.0

    # 2. Run without CLI flags, should fallback to config file values
    captured_kwargs.clear()
    test_args_no_flags = ["shotgen-gpu", "-c", str(cfg_file)]
    monkeypatch.setattr(sys, "argv", test_args_no_flags)
    main()

    assert captured_kwargs["t_start"] == 50.0
    assert captured_kwargs["t_end"] == 500.0
    assert captured_kwargs["t_taper"] == 50.0

