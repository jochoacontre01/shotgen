import os
from pathlib import Path


def _format_num_for_filename(val) -> str:
    """
    Format a number for inclusion in filenames.
    If None, returns 'None'.
    If float is convertible to int without precision loss (e.g. 100.0 or 12.0),
    convert to int string ('100', '12').
    If it has decimal places (e.g. 12.5), replace period '.' with underscore '_'.
    """
    if val is None:
        return "None"
    if isinstance(val, (int, float)):
        # If float equals integer value, return integer representation
        if float(val).is_integer():
            return str(int(val))
        s = str(val)
        return s.replace(".", "_")
    # For strings or other types
    s = str(val).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s.replace(".", "_")
    except ValueError:
        return s.replace(".", "_")


def generate_simulation_dir_name(cfg: dict, base_dir: str = "data") -> Path:
    """
    Generates a unique simulation output directory name based on parameters from a JSON/YAML config.

    Format order:
    gather, nx, nz, dx, dz, n_sources, n_receivers, ms, f0, group_offset, shot_offset, snr

    Example:
    data/commonshot_100nx_100nz_1dx_1dz_100src_50rec_100ms_8Hz_50groupoffset_12_5shotoffset_5snr
    """
    raw_gather = str(cfg.get("gather", "common shot"))
    gather_clean = raw_gather.replace(" ", "")

    nx = _format_num_for_filename(cfg.get("nx", 100))
    nz = _format_num_for_filename(cfg.get("nz", 100))
    dx = _format_num_for_filename(cfg.get("dx", 10.0))
    dz = _format_num_for_filename(cfg.get("dz", 10.0))
    n_sources = _format_num_for_filename(cfg.get("n_sources", 2))
    n_receivers = _format_num_for_filename(cfg.get("n_receivers", 10))
    ms = _format_num_for_filename(cfg.get("ms", cfg.get("ntime", 300.0)))
    f0 = _format_num_for_filename(cfg.get("f0", 25.0))
    group_offset = _format_num_for_filename(cfg.get("group_offset", 1.0))
    shot_offset = _format_num_for_filename(cfg.get("shot_offset", 1.0))
    snr = _format_num_for_filename(cfg.get("snr", None))

    dir_name = f"{gather_clean}_{nx}nx_{nz}nz_{dx}dx_{dz}dz_{n_sources}src_{n_receivers}rec_{ms}ms_{f0}Hz_{group_offset}groupoffset_{shot_offset}shotoffset_{snr}snr"

    if base_dir:
        return Path(base_dir) / dir_name
    return Path(dir_name)
