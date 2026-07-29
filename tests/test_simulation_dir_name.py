import os
from pathlib import Path
from shotgen.utils import generate_simulation_dir_name, _format_num_for_filename


def test_format_num_for_filename():
    assert _format_num_for_filename(100) == "100"
    assert _format_num_for_filename(100.0) == "100"
    assert _format_num_for_filename(12.5) == "12_5"
    assert _format_num_for_filename("12.5") == "12_5"
    assert _format_num_for_filename(None) == "None"


def test_generate_simulation_dir_name():
    cfg = {
        "gather": "common shot",
        "nx": 100,
        "nz": 100,
        "dx": 1.0,
        "dz": 1.0,
        "n_sources": 100,
        "n_receivers": 50,
        "ms": 100,
        "f0": 8.0,
        "group_offset": 50.0,
        "shot_offset": 12.5,
        "snr": 5,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir="data")
    expected = Path("data/commonshot_100nx_100nz_1dx_1dz_100src_50rec_100ms_8Hz_50groupoffset_12_5shotoffset_5snr")
    assert dir_name == expected


def test_generate_simulation_dir_name_defaults_and_none():
    cfg = {
        "gather": "cmp",
        "nx": 50,
        "nz": 50,
        "dx": 5.0,
        "dz": 5.0,
        "n_sources": 2,
        "n_receivers": 10,
        "ms": 300.0,
        "f0": 25.0,
        "group_offset": 1.5,
        "shot_offset": 2.0,
        "snr": None,
    }
    dir_name = generate_simulation_dir_name(cfg, base_dir="data")
    expected = Path("data/cmp_50nx_50nz_5dx_5dz_2src_10rec_300ms_25Hz_1_5groupoffset_2shotoffset_Nonesnr")
    assert dir_name == expected
