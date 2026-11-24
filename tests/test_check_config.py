import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import check_config


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_common_shot_reach_and_max_sources_fit(tmp_path):
    # Model X extent: 0 .. 1000 m (nx=100, dx=10)
    # CS: rec_origin=0, shot_offset=50, group_offset=20, n_receivers=11 -> rec_span=200
    # For N=5: latest_rec_x = 0 + 5*50 + 200 = 450 m, last_src_x = 0 + 4*50 = 200 m
    # Max sources: floor((1000 - 0 - 200) / 50) = 16
    cfg_path = _write_json(
        tmp_path / "cs_ok.json",
        {
            "task": "simulation",
            "nx": 100,
            "nz": 80,
            "dx": 10.0,
            "dz": 10.0,
            "n_sources": 5,
            "n_receivers": 11,
            "gather": "common shot",
            "src_origin": [0.0, 10.0],
            "rec_origin": [0.0, 10.0],
            "shot_offset": 50.0,
            "group_offset": 20.0,
            "f0": 15.0,
            "ms": 1000.0,
            "fs_ms": 2.0,
            "v_base": 2500.0,
            "n_damping": 40,
            "fd_order": 4,
        },
    )

    rep = check_config.analyze_config(cfg_path, fast=True)
    assert rep.model_max_x == pytest.approx(1000.0)
    assert rep.last_src_x == pytest.approx(200.0)
    assert rep.latest_rec_x == pytest.approx(450.0)
    assert rep.margin_x_m == pytest.approx(550.0)
    assert rep.overflow_x_m == pytest.approx(0.0)
    assert rep.overflow_x_cells == 0
    assert rep.max_sources_fit == 16
    assert rep.max_sources_by_rec == 16
    assert rep.max_sources_by_src == 21
    assert rep.spread_length_m == pytest.approx(200.0)
    assert rep.nominal_cmp_fold == pytest.approx(11 * 20.0 / (2.0 * 50.0))
    assert rep.overall_status == "PASS"


def test_common_shot_receiver_overflow_warns_and_counts_cells(tmp_path):
    # Model X extent: 1000 m; 18 sources -> latest_rec_x = 18*50 + 200 = 1100 m (+100 m, +10 cells)
    cfg_path = _write_json(
        tmp_path / "cs_overflow.json",
        {
            "nx": 100,
            "nz": 50,
            "dx": 10.0,
            "dz": 10.0,
            "n_sources": 18,
            "n_receivers": 11,
            "gather": "cs",
            "src_origin": [0.0, 10.0],
            "rec_origin": [0.0, 10.0],
            "shot_offset": 50.0,
            "group_offset": 20.0,
            "f0": 15.0,
            "ms": 800.0,
            "v_base": 2500.0,
            "n_damping": 40,
        },
    )

    rep = check_config.analyze_config(cfg_path, fast=True)
    assert rep.latest_rec_x == pytest.approx(1100.0)
    assert rep.overflow_x_m == pytest.approx(100.0)
    assert rep.overflow_x_cells == 10
    assert rep.max_sources_fit == 16
    rec_check = next(c for c in rep.checks if c.name == "receiver_x_bounds")
    assert rec_check.status == "WARN"


def test_cmp_gather_and_yaml_support(tmp_path):
    cfg_path = tmp_path / "cmp_cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "nx": 50,
                "nz": 40,
                "dx": 10.0,
                "dz": 10.0,
                "n_sources": 4,
                "n_receivers": 10,
                "gather": "cmp",
                "src_origin": [50.0, 5.0],
                "rec_origin": [20.0, 5.0],
                "shot_offset": 100.0,
                "group_offset": 30.0,
                "f0": 15.0,
                "ms": 500.0,
                "v_base": 2000.0,
                "n_damping": 30,
            }
        ),
        encoding="utf-8",
    )

    rep = check_config.analyze_config(cfg_path, fast=True)
    assert rep.gather == "common midpoint"
    assert rep.last_src_x == pytest.approx(350.0)
    assert rep.latest_rec_x == pytest.approx(290.0)
    # floor((500 - 50) / 100) + 1 = 5
    assert rep.max_sources_fit == 5


def test_loads_npy_model_with_resample_and_crop(tmp_path):
    # Create a small 40x30 velocity model with v_min=1500, v_max=4000
    vel = np.linspace(1500.0, 4000.0, 40 * 30, dtype=np.float32).reshape(40, 30)
    vel_path = tmp_path / "mini_vel.npy"
    np.save(vel_path, vel)

    cfg_path = _write_json(
        tmp_path / "with_model.json",
        {
            "velocity_model": str(vel_path),
            "dx_orig": 10.0,
            "dz_orig": 10.0,
            "dx": 10.0,
            "dz": 10.0,
            "max_size": [300.0, 200.0],
            "n_sources": 3,
            "n_receivers": 5,
            "gather": "cs",
            "shot_offset": 40.0,
            "group_offset": 10.0,
            "f0": 20.0,
            "ms": 500.0,
            "fs_ms": 4.0,
            "n_damping": 20,
        },
    )

    rep = check_config.analyze_config(cfg_path, fast=False)
    assert rep.velocity_source == "loaded velocity model"
    assert rep.nx == 30
    assert rep.nz == 20
    assert rep.model_max_x == pytest.approx(300.0)
    assert rep.v_min == pytest.approx(1500.0, rel=1e-3)
    assert rep.v_max <= 4000.0 + 1e-3
    # rec_span = 40 m; max_sources = floor((300 - 40) / 40) = 6
    assert rep.max_sources_fit == 6


def test_cli_json_and_strict_exit_code(tmp_path, capsys):
    # Source out of bounds -> FAIL
    bad_cfg = _write_json(
        tmp_path / "bad.json",
        {
            "nx": 20,
            "nz": 20,
            "dx": 10.0,
            "dz": 10.0,
            "n_sources": 10,
            "n_receivers": 5,
            "gather": "cs",
            "shot_offset": 50.0,
            "group_offset": 10.0,
            "f0": 20.0,
            "ms": 200.0,
            "v_base": 2500.0,
        },
    )

    rc = check_config.main([str(bad_cfg), "--fast", "--json", "--strict"])
    assert rc == 1
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["overall_status"] == "FAIL"
    assert parsed["max_sources_fit"] == 3


def test_cli_directory_summary_table(tmp_path, capsys):
    for i in (1, 2):
        _write_json(
            tmp_path / f"cfg_{i}.json",
            {
                "nx": 60,
                "nz": 40,
                "dx": 10.0,
                "dz": 10.0,
                "n_sources": i * 2,
                "n_receivers": 6,
                "gather": "cs",
                "shot_offset": 40.0,
                "group_offset": 10.0,
                "f0": 15.0,
                "ms": 500.0,
                "v_base": 2500.0,
            },
        )

    rc = check_config.main([str(tmp_path), "--fast"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUMMARY" in out
    assert "cfg_1.json" in out
    assert "cfg_2.json" in out
