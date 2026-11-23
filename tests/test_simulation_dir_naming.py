import json
import re

from shotgen.cli import main as core_main
from shotgen.utils import (
    MAX_SIM_DIR_NAME_LEN,
    canonicalize_model_name,
    generate_simulation_dir_name,
    resolve_simulation_output_dir,
)


def test_naming_with_model_and_fs_ms():
    cfg = {
        "velocity_model": "marmousi",
        "gather": "common shot",
        "nx": 120,
        "nz": 80,
        "dx": 12.5,
        "dz": 10.0,
        "n_sources": 3,
        "n_receivers": 24,
        "ms": 500.0,
        "fs_ms": 6.0,
        "f0": 20.0,
        "group_offset": 5.0,
        "shot_offset": 10.0,
        "snr": None,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir="data")
    assert dir_name.parent.name == "data"
    assert re.match(r"^marmousi_cs_3s_24r_20hz_[0-9a-f]{8}$", dir_name.name)
    assert len(dir_name.name) <= MAX_SIM_DIR_NAME_LEN

    # Deterministic across identical configs and distinct when fs_ms changes
    assert generate_simulation_dir_name(cfg, base_dir="data") == dir_name
    cfg_diff_fs = dict(cfg, fs_ms=2.0)
    assert generate_simulation_dir_name(cfg_diff_fs, base_dir="data") != dir_name


def test_naming_native_fs_and_filepath_model_and_snr():
    cfg = {
        "velocity_model": "assets/complex_graben.sgy",
        "gather": "cmp",
        "nx": 50,
        "nz": 50,
        "dx": 5.0,
        "dz": 5.0,
        "n_sources": 1,
        "n_receivers": 10,
        "ms": 100.0,
        "f0": 30.0,
        "snr": 15.5,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir=None)
    assert re.match(r"^graben_cmp_1s_10r_30hz_[0-9a-f]{8}$", dir_name.name)
    assert len(dir_name.name) <= MAX_SIM_DIR_NAME_LEN

    # Changing SNR must change the hash
    cfg_noiseless = dict(cfg, snr=None)
    assert generate_simulation_dir_name(cfg_noiseless, base_dir=None) != dir_name


def test_naming_fallback_to_synth_when_model_missing():
    cfg = {
        "nx": 30,
        "nz": 30,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir=None)
    assert str(dir_name).startswith("synth_")
    assert len(dir_name.name) <= MAX_SIM_DIR_NAME_LEN


def test_naming_long_model_stays_under_max_path():
    cfg = {
        "velocity_model": "CA-3000991-GOA_PSDM_Vel_RL_IL10249_XL4923_to_IL7796_XL7908_moho_discretized.segy",
        "gather": "common shot",
        "nx": 2986,
        "nz": 1750,
        "dx": 15.0,
        "dz": 20.0,
        "n_sources": 23,
        "n_receivers": 48,
        "ms": 12000.0,
        "fs_ms": 2.0,
        "f0": 10.0,
        "group_offset": 12.5,
        "shot_offset": 25.0,
        "snr": None,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir="data/simulations")
    assert re.match(r"^ca30009_moho_disc_cs_23s_48r_10hz_[0-9a-f]{8}$", dir_name.name)
    assert len(dir_name.name) <= MAX_SIM_DIR_NAME_LEN
    # Verify worst-case Windows path with a 120-char repo root and 45-char log file stays < 260
    worst_case_len = 120 + len(str(dir_name)) + 1 + 45
    assert worst_case_len < 260


def test_canonicalize_model_name():
    assert canonicalize_model_name("vp_marmousi-ii.segy") == "marmousi"
    assert canonicalize_model_name("bpsalt") == "bpsalt"
    assert canonicalize_model_name("sigsbee2a_stratigraphy_meters.sgy") == "sigsbee2a"
    assert canonicalize_model_name("overthrust_2d.segy") == "overthrust2d"
    assert canonicalize_model_name("wavy_inclined_layer_2.segy") == "wavy_inc2"
    assert canonicalize_model_name("simple_sinusoidal_fault.segy") == "sin_fault"
    assert canonicalize_model_name("two-layer.segy") == "2layer"
    assert canonicalize_model_name("three-layer.segy") == "3layer"
    assert len(canonicalize_model_name("a_very_long_arbitrary_custom_geological_velocity_model_name.segy")) <= 18


def test_resolve_simulation_output_dir():
    cfg = {"velocity_model": "marmousi", "nx": 10, "nz": 10, "fs_ms": 4.0}

    # 1. Omitted output_dir -> uses data/<generated>
    resolved_default = resolve_simulation_output_dir(cfg)
    assert str(resolved_default).startswith("data/marmousi_")
    assert len(resolved_default.name) <= MAX_SIM_DIR_NAME_LEN

    # 2. Parent directory with trailing slash -> nests inside it
    cfg_parent = dict(cfg, output_dir="custom_sims/")
    resolved_parent = resolve_simulation_output_dir(cfg_parent)
    assert str(resolved_parent).startswith("custom_sims/marmousi_")

    # 3. Known parent directory name without slash -> nests inside it
    cfg_data = dict(cfg, output_dir="simulations")
    resolved_data = resolve_simulation_output_dir(cfg_data)
    assert str(resolved_data).startswith("simulations/marmousi_")

    # 4. Explicit specific leaf folder -> used directly
    cfg_explicit = dict(cfg, output_dir="/tmp/my_exact_run")
    resolved_explicit = resolve_simulation_output_dir(cfg_explicit)
    assert str(resolved_explicit) == "/tmp/my_exact_run"


def test_cli_execution_with_auto_generated_name_and_config_copy(tmp_path, monkeypatch):
    """Verify that end-to-end CLI without output_dir generates the named folder and copies the config into it."""
    cfg_file = tmp_path / "survey_test.json"
    cfg = {
        "velocity_model": "simple_sinusoidal",
        "dx": 20.0,
        "dz": 20.0,
        "nx": 30,
        "nz": 30,
        "n_sources": 1,
        "n_receivers": 4,
        "ms": 20.0,
        "fs_ms": 4.0,
        "f0": 20.0,
        "engine": "pylops",
        "device": "cpu",
        "output_dir": str(tmp_path) + "/",  # Trailing slash indicates parent container
    }
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    monkeypatch.setattr("sys.argv", ["shotgen-core", "--config", str(cfg_file)])
    core_main()

    # Find the created directory inside tmp_path
    subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert len(subdirs) == 1
    dataset_dir = subdirs[0]

    # Verify folder name follows compact naming convention
    assert re.match(r"^sinusoidal_cmp_1s_4r_20hz_[0-9a-f]{8}$", dataset_dir.name)
    assert len(dataset_dir.name) <= MAX_SIM_DIR_NAME_LEN

    # Verify dataset files and copied config exist inside
    assert (dataset_dir / "traces.segy").exists()
    assert (dataset_dir / "survey_test.json").exists()
