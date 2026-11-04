import json

from shotgen.cli import main as core_main
from shotgen.utils import generate_simulation_dir_name, resolve_simulation_output_dir


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
    expected = (
        "data/marmousi_commonshot_120nx_80nz_12_5dx_10dz_3src_24rec_"
        "500ms_fs_6ms_20Hz_5groupoffset_10shotoffset_noiseless"
    )
    assert str(dir_name) == expected


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
    expected = (
        "complex_graben_cmp_50nx_50nz_5dx_5dz_1src_10rec_"
        "100ms_fs_native_30Hz_1groupoffset_1shotoffset_15_5snr"
    )
    assert str(dir_name) == expected


def test_naming_fallback_to_synthetic_when_model_missing():
    cfg = {
        "nx": 30,
        "nz": 30,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir=None)
    assert str(dir_name).startswith("synthetic_")


def test_resolve_simulation_output_dir():
    cfg = {"velocity_model": "marmousi", "nx": 10, "nz": 10, "fs_ms": 4.0}
    
    # 1. Omitted output_dir -> uses data/<generated>
    resolved_default = resolve_simulation_output_dir(cfg)
    assert str(resolved_default).startswith("data/marmousi_")
    assert "fs_4ms" in str(resolved_default)

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

    # Verify folder name has the comprehensive tokens
    assert dataset_dir.name.startswith("simple_sinusoidal_")
    assert "fs_4ms" in dataset_dir.name
    assert "noiseless" in dataset_dir.name

    # Verify dataset files and copied config exist inside
    assert (dataset_dir / "traces.segy").exists()
    assert (dataset_dir / "survey_test.json").exists()
