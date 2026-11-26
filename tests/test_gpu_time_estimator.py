"""Unit tests for shotgen-gpu runtime estimation and calibration."""

from __future__ import annotations

import json

import numpy as np
import pytest
from shotgen_gpu.estimator import (
    GPUInfo,
    compute_cfl_dt_nt,
    estimate_runtime,
    format_duration,
    format_estimate_report,
    save_calibration_db,
)


@pytest.mark.parametrize("space_order", [4, 8])
@pytest.mark.parametrize("dx,dz", [(10.0, 10.0), (5.0, 10.0)])
def test_cfl_dt_nt_matches_devito(space_order, dx, dz):
    """compute_cfl_dt_nt should exactly match Devito Model.critical_dt and TimeAxis."""
    from examples.seismic import Model, TimeAxis

    vp = np.full((30, 30), 3500.0, dtype=np.float32)
    model = Model(
        vp=vp * 1e-3,
        origin=(0.0, 0.0),
        shape=vp.shape,
        spacing=(dx, dz),
        space_order=space_order,
        nbl=10,
        bcs="damp",
        dtype=np.float32,
    )
    ms = 500.0
    ta = TimeAxis(start=0.0, stop=ms, step=model.critical_dt)

    dt_ms, nt = compute_cfl_dt_nt(
        max_vp_m_s=float(np.max(vp)),
        dx=dx,
        dz=dz,
        space_order=space_order,
        ms=ms,
        dtype=np.float32,
    )
    assert dt_ms == pytest.approx(float(model.critical_dt), rel=1e-6)
    assert nt == ta.num


def test_estimate_runtime_simulation_scaling(tmp_path, monkeypatch):
    """Runtime estimate should scale with n_sources, fd_order, float64, and save_wavefield."""
    fake_gpu = GPUInfo(
        name="Test RTX GPU",
        vram_mib=20480.0,
        compute_cap="8.9",
        is_gpu=True,
    )
    monkeypatch.setattr("shotgen_gpu.estimator.detect_gpu_info", lambda _dev="auto": fake_gpu)

    cal_path = tmp_path / "calibration.json"
    base_cfg = {
        "task": "simulation",
        "nx": 100,
        "nz": 100,
        "dx": 10.0,
        "dz": 10.0,
        "n_sources": 4,
        "n_receivers": 16,
        "ms": 500.0,
        "fd_order": 4,
        "n_damping": 20,
        "smooth": 1.0,
        "float_type": "float32",
        "save_wavefield": False,
    }

    # Uncalibrated fallback
    est_fallback = estimate_runtime(base_cfg, calibration_path=cal_path)
    assert "fallback" in est_fallback.calibration_source
    assert est_fallback.uncertainty_frac == pytest.approx(0.40)

    # Now save a calibration entry for Test RTX GPU
    save_calibration_db(
        {
            "schema_version": 1,
            "gpus": {
                "Test RTX GPU": {
                    "vram_mib": 20480.0,
                    "compute_cap": "8.9",
                    "calibrated_at": "2026-11-26T14:00:00",
                    "simulation": {
                        "jit_compile_s": 3.0,
                        "per_shot_overhead_s": 0.2,
                        "gcells_per_s_order4": 1.0,
                        "order_exponent": 1.0,
                        "float64_factor": 2.0,
                        "save_wavefield_multiplier": 1.6,
                    },
                }
            },
        },
        calibration_path=cal_path,
    )

    est_base = estimate_runtime(base_cfg, calibration_path=cal_path)
    assert "calibrated" in est_base.calibration_source
    assert est_base.uncertainty_frac == pytest.approx(0.20)

    # More sources -> longer runtime
    est_more_src = estimate_runtime({**base_cfg, "n_sources": 12}, calibration_path=cal_path)
    assert est_more_src.compute_s == pytest.approx(est_base.compute_s * 3.0, rel=1e-6)
    assert est_more_src.setup_overhead_s == pytest.approx(est_base.setup_overhead_s * 3.0, rel=1e-6)

    # Higher fd_order -> higher compute time (both smaller CFL dt and slower stencil)
    est_o8 = estimate_runtime({**base_cfg, "fd_order": 8}, calibration_path=cal_path)
    assert est_o8.compute_s > est_base.compute_s

    # float64 -> slower compute and higher VRAM
    est_f64 = estimate_runtime({**base_cfg, "float_type": "float64"}, calibration_path=cal_path)
    assert est_f64.compute_s > est_base.compute_s
    assert est_f64.estimated_vram_mib > est_base.estimated_vram_mib

    # save_wavefield -> slower compute and much higher VRAM
    est_wav = estimate_runtime({**base_cfg, "save_wavefield": True}, calibration_path=cal_path)
    assert est_wav.compute_s > est_base.compute_s
    assert est_wav.estimated_vram_mib > est_base.estimated_vram_mib

    report = format_estimate_report(est_base)
    assert "Test RTX GPU" in report
    assert "SIMULATION" in report
    assert "Estimated Runtime (warm)" in report


def test_estimate_runtime_rtm_synthetic(tmp_path, monkeypatch):
    """estimate_runtime should support task='rtm' configs."""
    fake_gpu = GPUInfo(name="Test RTX GPU", vram_mib=20480.0, compute_cap="8.9", is_gpu=True)
    monkeypatch.setattr("shotgen_gpu.estimator.detect_gpu_info", lambda _dev="auto": fake_gpu)

    cfg = {
        "task": "rtm",
        "nx": 80,
        "nz": 80,
        "dx": 10.0,
        "dz": 10.0,
        "n_sources": 5,
        "n_receivers": 20,
        "ms": 400.0,
        "space_order": 4,
        "nbl": 30,
    }
    est = estimate_runtime(cfg, calibration_path=tmp_path / "missing.json")
    assert est.workload.task == "rtm"
    assert est.workload.padded_nx == 140
    assert est.workload.padded_nz == 140
    assert est.total_warm_s > 0.0
    assert "RTM" in format_estimate_report(est)


def test_vram_oom_warning(tmp_path, monkeypatch):
    """Huge save_wavefield workload on small VRAM GPU should trigger CRITICAL OOM warning."""
    small_gpu = GPUInfo(name="Small GPU", vram_mib=1024.0, compute_cap="8.0", is_gpu=True)
    monkeypatch.setattr("shotgen_gpu.estimator.detect_gpu_info", lambda _dev="auto": small_gpu)

    cfg = {
        "task": "simulation",
        "nx": 500,
        "nz": 500,
        "dx": 5.0,
        "dz": 5.0,
        "n_sources": 1,
        "n_receivers": 10,
        "ms": 2000.0,
        "fd_order": 4,
        "n_damping": 50,
        "save_wavefield": True,
    }
    est = estimate_runtime(cfg, calibration_path=tmp_path / "missing.json")
    assert est.estimated_vram_mib > small_gpu.vram_mib
    assert est.vram_warning is not None
    assert "CRITICAL" in est.vram_warning


def test_format_duration():
    assert format_duration(12.34) == "12.3 s"
    assert format_duration(125.0) == "2m 05.0s"
    assert format_duration(3665.0) == "1h 01m 05s"


def test_cli_estimate_time_dry_run(tmp_path, monkeypatch, capsys):
    """shotgen-gpu --config ... --estimate-time prints estimate and does not run simulation."""
    from shotgen_gpu.cli import main

    out_dir = tmp_path / "should_not_exist"
    cfg_file = tmp_path / "cfg.json"
    cfg = {
        "task": "simulation",
        "nx": 40,
        "nz": 40,
        "dx": 10.0,
        "dz": 10.0,
        "n_sources": 2,
        "n_receivers": 6,
        "ms": 100.0,
        "fd_order": 4,
        "n_damping": 20,
        "output_dir": str(out_dir),
    }
    cfg_file.write_text(json.dumps(cfg), encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["shotgen-gpu", "--config", str(cfg_file), "--estimate-time"])
    main()

    captured = capsys.readouterr()
    assert "shotgen-gpu Runtime Estimate (SIMULATION)" in captured.out
    assert "Estimated Runtime (warm)" in captured.out
    assert not out_dir.exists()


def test_cli_calibrate_standalone(monkeypatch):
    """shotgen-gpu --calibrate works without --config and calls run_calibration."""
    from shotgen_gpu.cli import main

    called = {}

    def _fake_calibrate(device="auto", verbose=False):
        called["device"] = device
        called["verbose"] = verbose
        return {}

    monkeypatch.setattr("shotgen_gpu.estimator.run_calibration", _fake_calibrate)
    monkeypatch.setattr("sys.argv", ["shotgen-gpu", "--calibrate"])
    main()

    assert called == {"device": "auto", "verbose": False}


def test_cli_requires_config_without_calibrate(monkeypatch):
    """shotgen-gpu without --config and without --calibrate should exit with error."""
    from shotgen_gpu.cli import main

    monkeypatch.setattr("sys.argv", ["shotgen-gpu"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code != 0
